"""Required SwanLab setup shared by all training entry points."""

from __future__ import annotations

import os
from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
INVALID_API_KEYS = {
    "",
    "your_swanlab_api_key_here",
    "replace_with_your_swanlab_api_key",
}


def add_swanlab_arguments(parser: ArgumentParser, *, default_project: str) -> None:
    parser.add_argument("--swanlab_project", "--wandb_project", default=default_project)
    parser.add_argument("--swanlab_workspace", "--wandb_entity", default="",
                        help="SwanLab workspace (defaults to your personal workspace)")
    parser.add_argument("--swanlab_run_name", "--wandb_run_name", default="",
                        help="Override the generated SwanLab experiment name")


def read_env_file(path: str | os.PathLike[str] = DEFAULT_ENV_PATH) -> dict[str, str]:
    """Read a small dotenv file without adding a python-dotenv dependency."""
    env_path = Path(path).expanduser().resolve()
    if not env_path.is_file():
        raise FileNotFoundError(
            f"SwanLab configuration file was not found: {env_path}. "
            "Create the file and set SWANLAB_API_KEY."
        )

    values: dict[str, str] = {}
    with env_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            if "=" not in line:
                raise ValueError(
                    f"invalid .env entry at {env_path}:{line_number}; expected KEY=VALUE"
                )
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if not key:
                raise ValueError(f"empty .env key at {env_path}:{line_number}")
            values[key] = value
    return values


def require_swanlab_api_key(
    path: str | os.PathLike[str] = DEFAULT_ENV_PATH,
) -> str:
    """Read the SwanLab credential from the project .env."""
    values = read_env_file(path)
    api_key = values.get("SWANLAB_API_KEY", "").strip()
    if api_key.lower() in INVALID_API_KEYS:
        raise ValueError(
            f"SWANLAB_API_KEY is missing or still a placeholder in {Path(path).resolve()}"
        )
    return api_key


def initialize_swanlab_run(
    *,
    project: str,
    name: str,
    config: Mapping[str, Any],
    workspace: str = "",
    run_id: str = "",
    env_path: str | os.PathLike[str] = DEFAULT_ENV_PATH,
):
    """Authenticate from .env and create the mandatory online SwanLab run."""
    api_key = require_swanlab_api_key(env_path)
    try:
        import swanlab
    except ImportError as exc:
        raise RuntimeError(
            "swanlab is required for training; install it with `python -m pip install swanlab`"
        ) from exc

    swanlab.login(api_key=api_key, relogin=True, save=False)
    init_kwargs = dict(
        project=project,
        workspace=workspace or None,
        experiment_name=name,
        config=dict(config),
        mode="online",
        on_online_error="raise",
    )
    if run_id:
        init_kwargs.update(id=str(run_id), resume="must")
    return swanlab.init(**init_kwargs)


def read_swanlab_run_id_from_checkpoint(
    path: str | os.PathLike[str] | None,
) -> str:
    """Read a persisted SwanLab Run ID without constructing a training model."""
    if not path:
        return ""
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to inspect a checkpoint") from exc
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:  # PyTorch versions before weights_only was added
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        return ""
    return str(checkpoint.get("swanlab_run_id", "") or "").strip()


def swanlab_run_id(run: Any) -> str:
    """Return a live run's stable ID, or an empty string for test doubles."""
    return str(getattr(run, "id", "") or "").strip()


def flatten_swanlab_metrics(prefix: str, data: Any) -> dict[str, float]:
    """Flatten nested metric dicts, per-class vectors, and matrices to scalars."""
    output: dict[str, float] = {}

    def visit(path: str, value: Any) -> None:
        if hasattr(value, "detach") and hasattr(value, "numel"):
            value = value.detach().float().cpu()
            if value.numel() == 1:
                output[path] = float(value.item())
                return
            visit(path, value.tolist())
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(f"{path}/{key}", item)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, item in enumerate(value):
                visit(f"{path}/{index}", item)
            return
        if isinstance(value, bool):
            output[path] = float(value)
        elif isinstance(value, (int, float)):
            output[path] = float(value)

    visit(prefix.rstrip("/"), data)
    return output


def flatten_swanlab_summary_metrics(prefix: str, data: Any) -> dict[str, float]:
    """Flatten scalar epoch summaries while skipping vectors and matrices."""
    output: dict[str, float] = {}

    def visit(path: str, value: Any) -> None:
        if hasattr(value, "detach") and hasattr(value, "numel"):
            value = value.detach().float().cpu()
            if value.numel() == 1:
                output[path] = float(value.item())
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(f"{path}/{key}", item)
            return
        if isinstance(value, bool):
            output[path] = float(value)
        elif isinstance(value, Real):
            output[path] = float(value)

    visit(prefix.rstrip("/"), data)
    return output


def swanlab_confusion_matrix(
    confusion_matrix: Any,
    class_names: Sequence[str],
    *,
    title: str,
):
    """Build one native SwanLab confusion-matrix panel from aggregated counts."""
    try:
        import swanlab
    except ImportError as exc:
        raise RuntimeError(
            "swanlab is required for confusion-matrix logging; "
            "install it with `python -m pip install swanlab`"
        ) from exc

    if hasattr(confusion_matrix, "detach"):
        confusion_matrix = confusion_matrix.detach().cpu()
    if hasattr(confusion_matrix, "tolist"):
        confusion_matrix = confusion_matrix.tolist()
    if not isinstance(confusion_matrix, Sequence) or isinstance(
        confusion_matrix, (str, bytes)
    ):
        raise TypeError("confusion_matrix must be a square numeric matrix")

    matrix = [list(row) for row in confusion_matrix]
    size = len(matrix)
    names = [str(name) for name in class_names]
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("confusion_matrix must be non-empty and square")
    if len(names) != size:
        raise ValueError(
            f"expected {size} class names for confusion matrix, got {len(names)}"
        )

    data = []
    for actual_index, row in enumerate(matrix):
        for predicted_index, value in enumerate(row):
            if hasattr(value, "item"):
                value = value.item()
            if not isinstance(value, Real):
                raise TypeError("confusion_matrix values must be numeric")
            count = float(value)
            if count < 0:
                raise ValueError("confusion_matrix values must be non-negative")
            if count.is_integer():
                count = int(count)
            data.append([predicted_index, actual_index, count])

    chart = swanlab.echarts.HeatMap()
    chart.add_xaxis(names)
    chart.add_yaxis("Count", names, data)
    chart.set_global_opts(
        title_opts={"text": title},
        legend_opts={"show": False},
        xaxis_opts={"name": "Predicted", "type": "category"},
        yaxis_opts={"name": "Actual", "type": "category", "inverse": True},
        visualmap_opts={"min": 0, "max": max(1, max(row[2] for row in data))},
    )
    return chart


__all__ = [
    "DEFAULT_ENV_PATH",
    "add_swanlab_arguments",
    "flatten_swanlab_metrics",
    "flatten_swanlab_summary_metrics",
    "initialize_swanlab_run",
    "read_swanlab_run_id_from_checkpoint",
    "read_env_file",
    "require_swanlab_api_key",
    "swanlab_confusion_matrix",
    "swanlab_run_id",
]
