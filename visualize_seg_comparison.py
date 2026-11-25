"""Compare MFCAD++ segmenters in an editable PowerPoint grid (requires python-pptx).

Select rows by test filename (10131.txt, then 1002.txt):
    python visualize_seg_comparison.py --part-ids "[10131, 1002]"
Randomly sample eight aligned test parts:
    python visualize_seg_comparison.py --part-ids "[]" --count 8 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import secrets
from dataclasses import dataclass
from importlib.util import find_spec
from math import ceil
from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

from data_utils.mfcad_seg_dataset import Stage2SegmentationDataset
from networks.segmentation_models import build_segmentation_model, segmentation_model_config
from tools.visualize_mfcad_seg import colors_from_label_map


DATA = Path("F:/DataSet/cstnet2")
# Folder names are explicit so the *_constraints baseline weights cannot enter the figure.
MODELS = {
    "pointnet": "PointNet",
    "pointnet2": "PointNet++",
    "dgcnn": "DGCNN",
    "attn3dgcn": "Attn-3DGCN",
    "pointtransformer": "Point Transformer",
    "pointmamba": "PointMamba",
    "pointnext": "PointNeXt",
    "pointmlp": "PointMLP",
    "cstnet": "CSTNet",
    "cstnet++": "CSTNet++",
}


@dataclass
class Part:
    name: str
    xyz: torch.Tensor
    labels: torch.Tensor
    constraints: torch.Tensor
    masks: torch.Tensor
    legacy_constraints: torch.Tensor


def first_xyz(path):
    # Most same-name exports have independent surface samples. Reject those
    # cheaply before parsing 2 x 2048 rows, then check every point below.
    with path.open("rb") as handle:
        return tuple(map(np.float32, handle.readline().split()[:3]))


def parse_part_ids(value):
    try:
        ids = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("use a JSON array such as [10131, 1002] or []") from exc
    if not isinstance(ids, list) or any(type(part_id) is not int or part_id < 0 for part_id in ids):
        raise argparse.ArgumentTypeError("part IDs must be an array of nonnegative integers")
    return ids


def match_selected_rows(item, other, *, cad_constraints):
    """Map standard attributes onto the legacy XYZ without interpolating point predictions."""
    # Same points in a different order: match exact coordinates, face IDs and labels.
    def point_order(sample):
        return np.lexsort((sample["labels"], sample["face_ids"], *sample["xyz"].numpy().T))

    src, dst = point_order(item), point_order(other)
    sample_id = item["sample_id"]
    if torch.equal(item["xyz"][src], other["xyz"][dst]):
        if not all(torch.equal(item[key][src], other[key][dst]) for key in ("face_ids", "labels")):
            raise ValueError(f"Part {sample_id}: different face IDs or labels at identical XYZ")
        rows = np.empty_like(dst)
        rows[dst] = src
        return rows

    if not cad_constraints:
        raise ValueError(f"Part {sample_id}: predicted constraints and legacy attributes have different "
                         "XYZ samples. Supply Stage 1 predictions for the legacy points.")
    # The two CAD exports can sample different points on the same B-rep face.
    # Only the standard CAD attributes are face-constant. Keep XYZ and the
    # point-specific nearby/edge flag together from the original legacy row.
    faces, first, inverse = np.unique(item["face_ids"].numpy(), return_index=True, return_inverse=True)
    target_faces = other["face_ids"].numpy()
    constraints = item["constraints"]
    if not np.isin(target_faces, faces).all() or not torch.equal(constraints[first][inverse], constraints):
        raise ValueError(f"Part {sample_id}: missing CAD faces or varying constraints within a face")
    rows = first[np.searchsorted(faces, target_faces)]
    if (not torch.equal(item["labels"][rows], other["labels"])
            or not torch.equal(constraints[rows, :5], other["constraints"][:, 5:])):
        raise ValueError(f"Part {sample_id}: CAD face labels or primitive types differ between exports")
    return rows


def sample_parts(root, legacy_root, count, seed, part_ids=(), *, cad_constraints=True):
    standard = Stage2SegmentationDataset(root, "test", n_points=None, storage_format="txt")
    legacy = Stage2SegmentationDataset(legacy_root, "test", n_points=None, cstnet_inputs=True)
    legacy_index = {path.stem: i for i, path in enumerate(legacy.files)}
    if part_ids:
        standard_index = {path.stem: i for i, path in enumerate(standard.files)}
        names = [str(part_id) for part_id in part_ids]
        missing = [name for name in names if name not in standard_index or name not in legacy_index]
        if missing:
            raise ValueError("Part IDs missing from one or both test splits: " + ", ".join(missing))
        candidates = [standard_index[name] for name in names]
        count = len(names)
    else:
        candidates = [i for i, path in enumerate(standard.files) if path.stem in legacy_index]
        random.Random(seed).shuffle(candidates)
    parts = []
    for i in candidates:
        path = standard.files[i]
        j = legacy_index[path.stem]
        if not part_ids and first_xyz(path) != first_xyz(legacy.files[j]):
            continue
        item, other = standard[i], legacy[j]
        rows = slice(None)
        if not all(torch.equal(item[key], other[key]) for key in ("xyz", "labels", "face_ids")):
            if not part_ids:
                continue
            rows = match_selected_rows(item, other, cad_constraints=cad_constraints)
        parts.append(Part(path.stem, other["xyz"], other["labels"], item["constraints"][rows],
                          item["constraint_masks"][rows], other["constraints"]))
        if len(parts) == count:
            return parts, standard.label_map
    raise ValueError(f"Requested {count} test parts, but only {len(parts)} of "
                     f"{len(candidates)} common IDs have identical ordered XYZ, face IDs and labels.")


def load_model(folder, weights_root, label_map):
    arch = "constraint_aware" if folder == "cstnet++" else folder
    if arch == "cstnet":
        path = weights_root / folder / "ca_dataset_part_seg_face_v2.pth"
        state = torch.load(path, map_location="cpu", weights_only=True)
        config = {"model": arch, "cstnet_n_metatype": state["preprocess.linear_layers.0.weight"].shape[1] - 8}
    else:
        path = weights_root / folder / "best_point_miou.pth"
        # Local trainer checkpoints also contain optimizer and NumPy RNG state.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = segmentation_model_config(checkpoint.get("model_config") or checkpoint["args"])
        if config["model"] != arch or config.get("baseline_use_constraints", False):
            raise ValueError(f"Incompatible model configuration in {path}: {config}")
        if checkpoint["label_map"] != label_map:
            raise ValueError(f"Segmentation label map differs in {path}")
        state = checkpoint["model"]
    model = build_segmentation_model(len(label_map["labels"]), config)
    model.load_state_dict(state, strict=True)
    return model.eval()


@torch.inference_mode()
def predict(parts, weights_root, label_map, device, seed):
    predictions = {}
    for folder, title in MODELS.items():
        model = load_model(folder, weights_root, label_map).to(device)
        rows = []
        for row, part in enumerate(parts):
            torch.manual_seed(seed + row)  # Some evaluation backbones use random FPS starts.
            xyz = part.xyz.unsqueeze(0).to(device)
            if folder == "cstnet":
                logits = model(xyz, part.legacy_constraints.unsqueeze(0).to(device))
            elif folder == "cstnet++":
                logits = model(xyz, part.constraints.unsqueeze(0).to(device),
                               part.masks.unsqueeze(0).to(device))
            else:
                logits = model(xyz)
            if logits.shape != (1, len(part.xyz), len(label_map["labels"])) or not logits.isfinite().all():
                raise ValueError(f"Invalid per-point scores from {title} for part {part.name}")
            rows.append(logits[0].argmax(-1).cpu().numpy())
        predictions[title] = rows
        del model
        print(f"{title}: {len(rows)} predictions", flush=True)
    return predictions


def project(xyz, azimuth, elevation):
    """Project once per row, retaining the draw order and a geometry-only lighting term."""
    az, el = np.radians([azimuth, elevation])
    right = np.array([-np.sin(az), np.cos(az), 0.])
    view = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    up = np.cross(view, right)
    xy = np.column_stack((xyz @ right, xyz @ up))
    xy -= (xy.max(0) + xy.min(0)) / 2
    depth = xyz @ view
    order = np.argsort(depth, kind="stable")

    # Local PCA normals give the uncoloured scan readable faces without RGB input.
    _, neighbors = cKDTree(xyz).query(xyz, k=min(16, len(xyz)))
    patch = xyz[neighbors] - xyz[neighbors].mean(axis=1, keepdims=True)
    normals = np.linalg.eigh(patch.transpose(0, 2, 1) @ patch)[1][:, :, 0]
    normals[normals @ view < 0] *= -1
    light = 0.60 * view + 0.60 * up - 0.53 * right
    illumination = np.clip(normals @ light, 0, 1)
    return xy[order], order, illumination[order, None]


def _comparison_figure(parts, predictions, label_map, args, seed, page_rows):
    titles = ["Input XYZ", "Ground truth", *predictions]
    palette = colors_from_label_map(label_map) / 255.
    cell_w, cell_h = 1.38, 1.13  # physical inches
    left, right, top, bottom = 0.60, 0.12, 0.62, 1.66
    width = left + cell_w * len(titles) + right
    height = top + cell_h * page_rows + bottom
    fig = plt.figure(figsize=(width, height), facecolor="white")
    ink, muted, rule = "#192936", "#485d69", "#bbc5ca"

    def label(x, y, value, **kwargs):
        fig.text(x / width, y / height, value, color=kwargs.pop("color", ink),
                 va="center", **kwargs)

    def line(x1, y1, x2, y2, color=rule, lw=0.55):
        fig.add_artist(Line2D([x1 / width, x2 / width], [y1 / height, y2 / height],
                              transform=fig.transFigure, color=color, lw=lw))

    # label(0.16, height - 0.26, "MFCAD++  /  Point cloud segmentation", fontsize=15, weight="bold")
    # label(width - right, height - 0.26, f"Test split   /   {len(parts)} parts   /   {len(predictions)} models", ha="right", fontsize=10, color=muted)
    # line(0.66, height - 0.49, width - right, height - 0.49)

    for row, part in enumerate(parts):
        y = height - top - (row + 1) * cell_h
        # label(left - 0.05, y + cell_h / 2, part.name, ha="right", fontsize=8, color=muted)  # 零件编号
        xy, order, light = project(part.xyz.numpy(), args.azimuth, args.elevation)
        scale = max(np.ptp(xy[:, 0]) / cell_w, np.ptp(xy[:, 1]) / cell_h, 1e-8) / 0.89
        colors = [None, part.labels.numpy(), *(rows[row] for rows in predictions.values())]
        for col, ids in enumerate(colors):
            x = left + col * cell_w
            ax = fig.add_axes([x / width, y / height, cell_w / width, cell_h / height],
                              label=f"{part.name} - {titles[col]}")
            # Soft light on label colours preserves the same class hues in every panel.
            if ids is None:
                rgb = np.array([0.66, 0.72, 0.75]) * (0.47 + 0.53 * light)
            else:
                rgb = palette[ids[order]] * (0.82 + 0.18 * light)
            ax.scatter(xy[:, 0], xy[:, 1], c=rgb, s=5.2 * 2048 / len(xy),
                       linewidths=0, rasterized=True)
            ax.set(xlim=(-cell_w * scale / 2, cell_w * scale / 2),
                   ylim=(-cell_h * scale / 2, cell_h * scale / 2))
            ax.set_axis_off()

    for col, title in enumerate(titles):
        x = left + col * cell_w
        if col:
            line(x, bottom - 0.24, x, height - top - 0.13)
        label(x + cell_w / 2, bottom - 0.12, title, ha="center", fontsize=9,
              weight="bold" if title == "CSTNet++" else "normal")
    line(0.66, bottom - 0.34, width - right - 0.2, bottom - 0.34)

    # Put the key in label-ID order; keep swatches and names as vector artwork.
    for i, entry in enumerate(label_map["labels"]):
        col, row = divmod(i, 5)
        x = left + col * (width - left - right) / 5 + 0.05
        y = bottom - 0.51 - row * 0.17
        label(x, y, "●", fontsize=9, color=palette[i])
        label(x + 0.15, y, entry["display_name"], fontsize=8)
    source = "precomputed predicted" if args.predicted_root else "CAD"
    # label(0.16, 0.17, f"CSTNet++ uses {source} constraints.  |  CSTNet uses the legacy direction, edge and primitive attributes.", fontsize=8, color=muted)
    # seed_label = "Inference seed" if args.part_ids else "Random seed"
    # label(width - right, 0.17, f"{seed_label} {seed}", ha="right", fontsize=8, color=muted)
    return fig


def draw_figure(parts, predictions, label_map, args, seed):
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".pdf":
        fig = _comparison_figure(parts, predictions, label_map, args, seed, len(parts))
        try:
            fig.savefig(output, format="pdf", dpi=350)
        finally:
            plt.close(fig)
        return

    from pptx import Presentation
    from tools.segmentation_pptx import add_figure_slide

    deck = Presentation()
    deck.core_properties.title = "MFCAD++ point cloud segmentation"
    # Keep larger selections readable. Balance the page lengths and give every
    # slide the same physical size, including the final slide.
    pages = np.array_split(np.arange(len(parts)), ceil(len(parts) / 12))
    for page in pages:
        selection = slice(int(page[0]), int(page[-1]) + 1)
        page_predictions = {title: rows[selection] for title, rows in predictions.items()}
        fig = _comparison_figure(parts[selection], page_predictions, label_map, args, seed, len(pages[0]))
        try:
            add_figure_slide(deck, fig)
        finally:
            plt.close(fig)
    with NamedTemporaryFile(dir=output.parent, suffix=".pptx", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        deck.save(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights-root", type=Path, default=DATA / "seg_weights")
    parser.add_argument("--data-root", type=Path, default=DATA / "mfcad_pcd", help="14-column CAD constraint dataset")
    parser.add_argument("--cstnet-root", type=Path, default=DATA / "mfcad_pcd_cstnet", help="11-column legacy dataset")
    parser.add_argument("--predicted-root", type=Path, help="Use a 14-column Stage 1 prediction dataset in place of --data-root")
    parser.add_argument("--part-ids", type=parse_part_ids, default=[57259,10096,57436,10007,1000,1013], metavar="JSON",
                        help="Ordered test part IDs (filename stems), e.g. '[10131, 1002]'; [] selects randomly")
    parser.add_argument("--count", type=int, default=9, help="Number of random parts when --part-ids is empty (default: 9)")
    parser.add_argument("--seed", type=int, default=13, help="Seed for random sampling and inference (default: 13)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--azimuth", type=float, default=-55)
    parser.add_argument("--elevation", type=float, default=28)
    parser.add_argument("--output", type=Path, default=Path("output/pptx/mfcad_seg_comparison.pptx"),
                        help="Output .pptx (editable, default) or .pdf")
    args = parser.parse_args()
    if not args.part_ids and args.count < 1:
        parser.error("--count must be positive")
    if args.output.suffix.lower() not in (".pptx", ".pdf"):
        parser.error("--output must end in .pptx or .pdf")
    if args.output.suffix.lower() == ".pptx" and find_spec("pptx") is None:
        parser.error("PowerPoint export requires python-pptx. Install with: python -m pip install python-pptx")
    if args.seed is not None and not 0 <= args.seed < 2**63:
        parser.error("--seed must be in [0, 2^63)")
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    if not args.part_ids:
        print(f"Selecting aligned test parts with seed {seed} ...", flush=True)
    try:
        parts, label_map = sample_parts(args.predicted_root or args.data_root, args.cstnet_root,
                                        args.count, seed, args.part_ids, cad_constraints=args.predicted_root is None)
    except ValueError as exc:
        parser.error(str(exc))
    print("Parts:", ", ".join(part.name for part in parts), flush=True)
    predictions = predict(parts, args.weights_root, label_map, args.device, seed)
    draw_figure(parts, predictions, label_map, args, seed)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
