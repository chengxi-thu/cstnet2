"""Train constraint-aware or baseline Stage 2 point-cloud classifiers."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from colorama import init

from data_utils.classification_dataset import Stage2ClassificationDataset
from functional.constraints import constraints_to_tensor, ground_truth_constraints_to_tensor
from functional.checkpoint_io import safe_torch_save
from functional.cuda_runtime import preload_cuda_nvrtc
from functional.swanlab_utils import (
    add_swanlab_arguments,
    flatten_swanlab_summary_metrics,
    initialize_swanlab_run,
    swanlab_confusion_matrix,
    swanlab_run_id,
)
from networks.classification_models import (
    CLASSIFICATION_MODEL_NAMES,
    DEFAULT_CLASSIFICATION_MODEL,
    build_classification_model,
    classification_model_config,
    classification_model_uses_constraints,
    classification_run_name,
)
from networks.cstnet import add_cstnet_arguments, load_cstnet_weights
from networks.utils import all_metric_cls


def add_classification_training_arguments(parser):
    parser.add_argument("--bs", "--batch_size", dest="bs", type=int, default=20)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n_points", "--n_point", dest="n_points", type=int, default=2048)

    parser.add_argument("--stage2_norm", choices=["ln", "bn"], default="ln")
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--aux_loss_weight", type=float, default=0.1)
    parser.add_argument("--token_dim", type=int, default=256)
    parser.add_argument("--transformer_layers", type=int, default=3)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--token_dropout", type=float, default=0.1)
    parser.add_argument("--stream_dropout", type=float, default=0.1)
    parser.add_argument("--use_stats_token", action="store_true", default=False)

    add_swanlab_arguments(parser, default_project="cstnet2-s2")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Load the selected model's existing weight file before training",
    )
    add_cstnet_arguments(parser)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser("Stage 2 classification training")
    parser.add_argument(
        "--baseline_use_constraints",
        action="store_true",
        default=False,
        help="Use the 12D per-point constraint as baseline point features",
    )
    parser.add_argument(
        "--model",
        choices=CLASSIFICATION_MODEL_NAMES,
        default=DEFAULT_CLASSIFICATION_MODEL,
        help="Stage 2 classification architecture",
    )
    add_classification_training_arguments(parser)
    parser.add_argument("--is_sample", action="store_true", default=False)
    parser.add_argument("--local", action="store_true", default=False)
    parser.add_argument("--data_root", type=str, default=None,
                        help="TXT/HDF5 dataset root for models other than CSTNet")
    parser.add_argument("--data_root_cstnet", type=str, default=None,
                        help="CSTNet classification dataset root with 9-column legacy TXT files")
    parser.add_argument("--data_format", choices=("auto", "txt", "h5"), default="auto",
                        help="CSTNet reads 9-column legacy TXT with binary nearby labels; other models support TXT/HDF5")
    parser.add_argument("--root_sever", type=str)
    parser.add_argument("--root_local", type=str)
    parser.add_argument("--save_name", type=str, default="stage2_cls_cstpred")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Test fraction used only when first creating split_file.json")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Random seed used only when first creating split_file.json")
    args = parser.parse_args(argv)
    fallback_root = args.root_local if args.local else args.root_sever
    if args.data_root is None:
        args.data_root = fallback_root or (
            r"D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_Extend" if args.local
            else "/opt/data/private/data_set/pcd_cstnet2/tmcad_pcd_pred"
        )
    if args.data_root_cstnet is None:
        args.data_root_cstnet = fallback_root or (
            r"F:\DataSet\tmcad_pcd_cstnet" if args.local
            else "/root/my_data/data_set/pcd_cstnet2/tmcad_pcd_cstnet"
        )
    if args.cstnet_weights and (args.model != "cstnet" or args.resume):
        parser.error("--cstnet_weights requires --model cstnet and a fresh run without --resume")
    return args


def constraints_from_dataset_batch(data, device: torch.device) -> torch.Tensor:
    """Assemble constraints already stored in the selected point-cloud files."""
    if len(data) == 3:  # CSTNet dataset supplies its original packed inputs.
        return data[2].float().to(device, non_blocking=True)
    return ground_truth_constraints_to_tensor(
        pmt=data[2].to(device, non_blocking=True),
        direction=data[3].float().to(device, non_blocking=True),
        dimension=data[4].float().to(device, non_blocking=True),
        location=data[5].float().to(device, non_blocking=True),
    )


def build_stage2_classifier(args, n_classes: int) -> torch.nn.Module:
    return build_classification_model(n_classes, classification_model_config(args))


def classification_loss(model, xyz, constraints, target, args, model_config):
    use_aux = (
        model_config["model"] == DEFAULT_CLASSIFICATION_MODEL
        and args.aux_loss_weight > 0.0
    )
    if not use_aux:
        log_probs = model(xyz, constraints)
        if args.label_smoothing > 0.0:
            loss = F.cross_entropy(log_probs, target, label_smoothing=args.label_smoothing)
        else:
            loss = F.nll_loss(log_probs, target)
        return log_probs, loss

    output = model(xyz, constraints, return_aux=True)
    loss = F.cross_entropy(
        output["main_logits"],
        target,
        label_smoothing=args.label_smoothing,
    )

    for key in ("aux_final_constraint_logits", "aux_component_token_logits"):
        loss = loss + args.aux_loss_weight * F.cross_entropy(
            output[key],
            target,
            label_smoothing=args.label_smoothing,
        )

    return output["log_probs"], loss


def classification_batch(data, device, use_constraints, constraint_extractor):
    xyz = data[0].float().to(device, non_blocking=True)
    target = data[1].long().to(device, non_blocking=True)
    constraints = None
    if constraint_extractor is not None:
        constraints = constraints_to_tensor(constraint_extractor(xyz))
    elif use_constraints:
        constraints = constraints_from_dataset_batch(data, device)
    return xyz, target, constraints


def evaluate(model, loader, device, use_constraints, constraint_extractor=None):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for data in tqdm(loader, total=len(loader), desc="eval"):
            xyz, target, constraints = classification_batch(
                data, device, use_constraints, constraint_extractor
            )
            pred = model(xyz, constraints)
            total_loss += F.nll_loss(pred, target).item()
            all_preds.append(pred.cpu().numpy())
            all_labels.append(target.cpu().numpy())
    metrics = all_metric_cls(all_preds, all_labels)
    return total_loss / max(len(loader), 1), metrics


def save_classification_checkpoints(
    checkpoint_payload,
    save_path: str,
    best_path: str,
    save_best: bool,
) -> dict[str, bool]:
    status = {"last": safe_torch_save(checkpoint_payload, save_path)}
    status["best"] = (
        safe_torch_save(checkpoint_payload, best_path) if save_best else True
    )
    return status


def load_classification_checkpoint(path: str, map_location) -> object:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was added
        return torch.load(path, map_location=map_location)


def restore_classification_training_state(
    checkpoint,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    requested_model_config: dict,
) -> tuple[int, float, str, bool]:
    """Restore a full checkpoint, while accepting legacy model-only weights."""
    if not (isinstance(checkpoint, Mapping) and "model" in checkpoint):
        state = (
            checkpoint["state_dict"]
            if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state, strict=True)
        return 0, 0.0, "", True

    required = {"epoch", "model", "optimizer", "scheduler", "best_acc"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(
            f"incomplete classification checkpoint; missing fields: {missing}"
        )
    saved_model_config = classification_model_config(
        checkpoint.get("model_config", requested_model_config)
    )
    if saved_model_config != requested_model_config:
        raise ValueError(
            "classification checkpoint model configuration mismatch: "
            f"saved={saved_model_config}, requested={requested_model_config}"
        )
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint["best_acc"]),
        str(checkpoint.get("swanlab_run_id", "") or "").strip(),
        False,
    )


def main(args):
    data_root = args.data_root_cstnet if args.model == "cstnet" else args.data_root
    train_loader, test_loader = Stage2ClassificationDataset.create_dataloaders(
        root=data_root,
        bs=args.bs,
        n_points=args.n_points,
        num_workers=args.workers,
        is_sample=args.is_sample,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        storage_format=args.data_format,
        cstnet_inputs=args.model == "cstnet",
    )
    save_stem = args.save_name
    if save_stem in {"stage2_cls", "stage2_cls_cstpred"}:
        save_stem += "_" + classification_run_name(classification_model_config(args))
    train_classification(args, train_loader, test_loader, data_root, save_stem)


def train_classification(args, train_loader, test_loader, data_root, save_stem,
                         output_dir="model_trained", constraint_extractor=None):
    os.makedirs(output_dir, exist_ok=True)
    n_classes = train_loader.dataset.n_classes()
    class_names = [f"class_{index}" for index in range(n_classes)]
    for name, index in train_loader.dataset.classes.items():
        class_names[int(index)] = str(name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = classification_model_config(args)
    use_constraints = classification_model_uses_constraints(model_config)

    nvrtc_library = None
    if (
        device.type == "cuda"
        and model_config["model"] == DEFAULT_CLASSIFICATION_MODEL
    ):
        nvrtc_library = preload_cuda_nvrtc()

    model = build_classification_model(n_classes, model_config).to(device)
    if args.cstnet_weights:
        load_cstnet_weights(model, args.cstnet_weights)
    save_path = os.path.join(output_dir, f"{save_stem}.pth")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"classification model: {model_config}; parameters={parameter_count:,}")
    print(f"constraints required by model: {use_constraints}")
    if nvrtc_library is not None:
        print(f"CUDA NVRTC: preloaded {nvrtc_library}")
    if args.resume and not os.path.exists(save_path):
        raise FileNotFoundError(
            f"resume=true but no classification weights were found: {save_path}"
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.decay_rate,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    start_epoch = 0
    best_acc = 0.0
    test_acc = None
    swanlab_resume_id = ""
    if args.resume:
        checkpoint = load_classification_checkpoint(save_path, map_location=device)
        start_epoch, best_acc, swanlab_resume_id, legacy_checkpoint = (
            restore_classification_training_state(
                checkpoint, model, optimizer, scheduler, model_config
            )
        )
        if legacy_checkpoint:
            print(
                "WARNING: loaded a legacy model-only classification checkpoint; "
                "optimizer/epoch/SwanLab state cannot be resumed, so training and "
                "SwanLab logging restart from epoch 1"
            )
        else:
            test_acc = checkpoint.get("test_acc")
            print(
                f"checkpoint: resumed {save_path}; next_epoch={start_epoch + 1}, "
                f"best_acc={best_acc:.4f}, swanlab_run_id={swanlab_resume_id or '<missing>'}"
            )
            if not swanlab_resume_id:
                print(
                    "WARNING: resume checkpoint has no swanlab_run_id; "
                    "a new SwanLab Run will be created"
                )
    elif args.cstnet_weights:
        print(f"Initialized CSTNet classification from {args.cstnet_weights}")
    else:
        print(f"checkpoint: none provided; starting a new training run ({save_path})")

    swanlab_run = initialize_swanlab_run(
        project=args.swanlab_project,
        workspace=args.swanlab_workspace,
        name=args.swanlab_run_name or save_stem,
        run_id=swanlab_resume_id,
        config={
            **vars(args),
            "model_config": model_config,
            "parameter_count": parameter_count,
            "num_classes": n_classes,
            "class_names": class_names,
            "selected_data_root": data_root,
            "constraint_storage": "frozen_stage1" if constraint_extractor is not None else "point_file_fields",
            "device": str(device),
        },
    )
    for epoch in range(start_epoch, args.epoch):
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        model.train()
        all_preds, all_labels = [], []
        running_loss = 0.0
        gradient_norm_sum = 0.0
        gradient_norm_max = 0.0
        gradient_norm_count = 0
        for data in tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"train {epoch + 1}/{args.epoch}",
        ):
            xyz, target, constraints = classification_batch(
                data, device, use_constraints, constraint_extractor
            )

            optimizer.zero_grad(set_to_none=True)
            pred, loss = classification_loss(
                model,
                xyz,
                constraints,
                target,
                args,
                model_config,
            )
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            )
            if np.isfinite(gradient_norm):
                gradient_norm_sum += gradient_norm
                gradient_norm_max = max(gradient_norm_max, gradient_norm)
                gradient_norm_count += 1
            optimizer.step()

            running_loss += loss.item()
            all_preds.append(pred.detach().cpu().numpy())
            all_labels.append(target.detach().cpu().numpy())

        train_metrics = all_metric_cls(all_preds, all_labels)
        test_loss, test_metrics = evaluate(
            model,
            test_loader,
            device,
            use_constraints,
            constraint_extractor,
        )
        train_loss = running_loss / max(len(train_loader), 1)

        test_acc = test_metrics["instance_accuracy"]
        improved = test_acc >= best_acc
        if improved:
            best_acc = test_acc
        scheduler.step()
        checkpoint_payload = {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": float(best_acc),
            "test_acc": test_acc,
            "args": vars(args),
            "model_config": model_config,
            "swanlab_run_id": swanlab_run_id(swanlab_run),
        }
        checkpoint_status = save_classification_checkpoints(
            checkpoint_payload,
            save_path,
            os.path.join(output_dir, f"{save_stem}_best.pth"),
            save_best=improved,
        )

        swanlab_payload = {
            "epoch": epoch + 1,
            "learning_rate": epoch_learning_rate,
            "loss/train": train_loss,
            "loss/test": test_loss,
            "best/test_instance_accuracy": best_acc,
            "checkpoint/last_saved": int(checkpoint_status["last"]),
            "checkpoint/best_saved": int(checkpoint_status["best"]),
            "train/optimization/gradient_norm_mean": (
                gradient_norm_sum / max(gradient_norm_count, 1)
            ),
            "train/optimization/gradient_norm_max": gradient_norm_max,
        }
        swanlab_payload.update(
            flatten_swanlab_summary_metrics("train/metric", train_metrics)
        )
        swanlab_payload.update(
            flatten_swanlab_summary_metrics("test/metric", test_metrics)
        )
        swanlab_payload["train/confusion_matrix"] = swanlab_confusion_matrix(
            train_metrics["confusion_matrix"],
            class_names,
            title="Train Classification Confusion Matrix",
        )
        swanlab_payload["test/confusion_matrix"] = swanlab_confusion_matrix(
            test_metrics["confusion_matrix"],
            class_names,
            title="Test Classification Confusion Matrix",
        )
        swanlab_run.log(swanlab_payload, step=epoch)

        print(
            f"epoch {epoch + 1}/{args.epoch} "
            f"train_loss={train_loss:.6f} test_loss={test_loss:.6f} "
            f"train_acc={train_metrics['instance_accuracy']:.4f} "
            f"test_acc={test_metrics['instance_accuracy']:.4f} "
            f"test_top5={test_metrics['top_5_accuracy']:.4f} "
            f"best={best_acc:.4f}"
        )

    swanlab_run.finish()
    return {"test_accuracy": test_acc, "best_test_accuracy": best_acc}


if __name__ == "__main__":
    init(autoreset=True)
    main(parse_args())
