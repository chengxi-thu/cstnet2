"""Run the ten coaxis/offset binary wheel classification experiments."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_utils.constraint_dataset_common import (
    CONSTRAINT_POINT_COLUMNS,
    cstnet_attributes,
    discover_txt_files,
    load_constraint_point_file,
    sample_without_replacement,
    split_constraint_columns,
)
from functional.checkpoint_io import safe_json_save
from gen_cst_pred import Stage1Predictor
from networks.classification_models import (
    classification_model_config,
    classification_model_uses_constraints,
    classification_run_name,
)
from train_cls import add_classification_training_arguments, train_classification


DATA_ROOT = r"/opt/data/private/data_set/pcd_cstnet2/tmcad_pcd_cstnet"
DATA_ROOT_CSTNET = r"/opt/data/private/data_set/pcd_cstnet2/tmcad_pcd"
SPLIT_SEED = 2026
TEST_RATIO = 0.2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=[
            "constraint_aware",
            "pointnet",
            "pointnet2",
            "dgcnn",
            "attn3dgcn",
            "pointtransformer",
            "pointmamba",
            "pointnext",
            "pointmlp",
            "cstnet",
        ],
        default="pointnet",
        help="Wheel classification architecture (default: %(default)s)",
    )
    parser.add_argument("--baseline_use_constraints", action="store_true", default=False, 
                        help="Add 12D constraints to XYZ for baseline models. CSTNet and constraint_aware always use constraints.")
    add_classification_training_arguments(parser)
    parser.add_argument("--data_root", default=DATA_ROOT,
                        help="12-column wheel dataset for constraint_aware and other baselines (default: %(default)s)")
    parser.add_argument("--data_root_cstnet", default=DATA_ROOT_CSTNET,
                        help="9-column wheel dataset for original CSTNet (default: %(default)s)")
    parser.add_argument("--output_dir", default="model_trained/wheels_binary")
    parser.add_argument("--stage1_checkpoint",
                        help="Optionally predict 12D constraints with frozen Stage 1 instead of reading file fields")
    args = parser.parse_args(argv)
    if args.epoch < 1:
        parser.error("--epoch must be positive")
    if args.cstnet_weights and (args.model != "cstnet" or args.resume):
        parser.error("--cstnet_weights requires --model cstnet and a fresh run without --resume")
    if args.model in ("cstnet", "constraint_aware"):
        args.baseline_use_constraints = False  # Both architectures always use their own constraint representation.
    return args


def wheel_splits(root: Path) -> dict[str, dict[str, list[Path]]]:
    offsets = sorted(
        (p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"offset-\d+(?:\.\d+)?", p.name)),
        key=lambda p: float(p.name.removeprefix("offset-")),
    )
    if len(offsets) != 10:
        raise ValueError(f"expected ten offset-XX directories in {root}, found {len(offsets)}")
    splits = {}
    for directory in [root / "coaxis", *offsets]:
        paths = discover_txt_files(directory)
        if len(paths) < 2:
            raise ValueError(f"need at least two point clouds to split {directory}")
        # Each class has a stable partition even if another class changes size.
        random.Random(SPLIT_SEED).shuffle(paths)
        n_test = math.ceil(len(paths) * TEST_RATIO)
        splits[directory.name] = {"train": paths[n_test:], "test": paths[:n_test]}
    return splits


class WheelPairDataset(Dataset):
    def __init__(self, splits, offset, split, n_points, cstnet_inputs, *, use_stored_constraints=False):
        self.classes = {"coaxis": 0, offset: 1}
        self.samples = [
            (path, label) for name, label in self.classes.items() for path in splits[name][split]
        ]
        self.n_points = n_points
        self.cstnet_inputs = cstnet_inputs
        self.use_stored_constraints = use_stored_constraints
        self.is_test = split == "test"

    def __len__(self):
        return len(self.samples)

    def n_classes(self):
        return len(self.classes)

    def __getitem__(self, index):
        path, label = self.samples[index]
        columns = 3
        if self.cstnet_inputs:
            columns = 9
        elif self.use_stored_constraints:
            columns = CONSTRAINT_POINT_COLUMNS
        cloud = load_constraint_point_file(
            path, task_name="wheel classification", expected_columns=columns, allow_extra_columns=columns == 3,
        )
        rng = np.random.default_rng(SPLIT_SEED + index) if self.is_test else None
        cloud = sample_without_replacement(cloud, self.n_points, path=path, rng=rng)
        xyz = cloud[:, :3]
        if self.cstnet_inputs:
            return xyz, label, cstnet_attributes(cloud, source=path)
        if self.use_stored_constraints:
            xyz, pmt, direction, dimension, location, _ = split_constraint_columns(cloud)
            return xyz, label, pmt, direction, dimension, location
        return xyz, label


def main(args):
    cstnet_inputs = args.model == "cstnet"
    root = Path(args.data_root_cstnet if cstnet_inputs else args.data_root)
    splits = wheel_splits(root)
    config = classification_model_config(args)
    uses_12d_constraints = not cstnet_inputs and classification_model_uses_constraints(config)
    use_stored_constraints = uses_12d_constraints and not args.stage1_checkpoint
    model_name = classification_run_name(config)
    output = Path(args.output_dir) / model_name
    output.mkdir(parents=True, exist_ok=True)
    files_by_class = {
        name: {split: [path.relative_to(root).as_posix() for path in paths]
               for split, paths in partitions.items()}
        for name, partitions in splits.items()
    }
    manifest = {"seed": SPLIT_SEED, "test_ratio": TEST_RATIO, "classes": files_by_class}
    if use_stored_constraints:
        manifest["constraint_source"] = "point_file_fields"
    manifest_file = output / "splits.json"
    if args.resume and any(output.glob("coaxis_vs_offset-*.pth")):
        if not manifest_file.is_file() or json.loads(manifest_file.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"wheel file lists or constraint source differ from the saved run in {output}; use a fresh --output_dir")
    elif not safe_json_save(manifest, manifest_file):
        raise OSError(f"could not save the wheel split manifest to {manifest_file}")

    extract_constraints = None
    if uses_12d_constraints and args.stage1_checkpoint:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        extract_constraints = Stage1Predictor(args.stage1_checkpoint, device).predict_batch

    results = {}
    offsets = [name for name in splits if name != "coaxis"]
    for i, offset in enumerate(offsets, start=1):
        train_set, test_set = (
            WheelPairDataset(splits, offset, split, args.n_points, cstnet_inputs,
                             use_stored_constraints=use_stored_constraints)
            for split in ("train", "test")
        )
        print(f"[{i}/{len(offsets)}] coaxis vs {offset} | train={len(train_set)} test={len(test_set)}")
        # Restart each pair from the same initialization and sampler seed.
        random.seed(SPLIT_SEED)
        np.random.seed(SPLIT_SEED)
        torch.manual_seed(SPLIT_SEED)
        loader_kwargs = dict(
            batch_size=args.bs, num_workers=args.workers,
            pin_memory=torch.cuda.is_available(), persistent_workers=args.workers > 0,
        )
        stem = f"coaxis_vs_{offset}"
        run_args = argparse.Namespace(**vars(args))
        run_args.resume = args.resume and (output / f"{stem}.pth").is_file()
        run_args.swanlab_run_name = f"{args.swanlab_run_name or model_name}_{stem}"
        run_args.offset = offset
        run_args.split_seed = SPLIT_SEED
        run_args.test_ratio = TEST_RATIO
        metrics = train_classification(
            run_args, DataLoader(train_set, shuffle=True, **loader_kwargs),
            DataLoader(test_set, shuffle=False, **loader_kwargs),
            str(root), stem, output, constraint_extractor=extract_constraints,
        )
        results[offset] = {"train_samples": len(train_set), "test_samples": len(test_set), **metrics}
        if not safe_json_save(results, output / "results.json"):
            raise OSError(f"could not save experiment results in {output}")
    return results


if __name__ == "__main__":
    main(parse_args())
