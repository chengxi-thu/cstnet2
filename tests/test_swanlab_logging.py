from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from functional.swanlab_utils import (
    flatten_swanlab_metrics,
    flatten_swanlab_summary_metrics,
    initialize_swanlab_run,
    read_swanlab_run_id_from_checkpoint,
    read_env_file,
    require_swanlab_api_key,
    swanlab_confusion_matrix,
)
from networks.utils import all_metric_cls


class SwanLabLoggingTest(unittest.TestCase):
    def test_all_training_clis_require_swanlab_without_disable_flag(self):
        import train_cls
        import train_cls_ablation
        import train_cls_wheels
        import train_cst_pred
        import train_seg
        import train_stage1_direct_baseline

        for script, project in (
            (train_cst_pred, "cstnet2-s1-abcmf30"),
            (train_stage1_direct_baseline, "cstnet2-s1-baseline-abcmf30"),
            (train_cls, "cstnet2-s2"), (train_cls_wheels, "cstnet2-s2"),
            (train_cls_ablation, "cstnet2-s2-ablation"), (train_seg, "cstnet2-s2"),
        ):
            with self.subTest(script=script.__name__):
                args = script.parse_args([])
                self.assertEqual(args.swanlab_project, project)
                self.assertEqual(args.swanlab_workspace, "")
                override = script.parse_args(["--swanlab_workspace", "team", "--wandb_run_name", "trial"])
                self.assertEqual(override.swanlab_workspace, "team")
                self.assertEqual(override.swanlab_run_name, "trial")

    def test_stage1_boolean_options_are_flag_based(self):
        import train_cst_pred

        defaults = train_cst_pred.parse_args(["--data_root", "dataset"])
        self.assertFalse(hasattr(defaults, "stage1_mode"))
        self.assertFalse(defaults.is_sample)
        self.assertEqual(defaults.data_root, "dataset")
        self.assertFalse(hasattr(defaults, "local"))
        self.assertFalse(hasattr(defaults, "root_local"))
        self.assertFalse(hasattr(defaults, "root_sever"))
        self.assertFalse(defaults.use_extra_features)
        self.assertFalse(defaults.overfit_one_batch)
        self.assertFalse(hasattr(defaults, "use_amp"))
        self.assertFalse(hasattr(defaults, "enable_grad_diagnostics"))
        self.assertFalse(hasattr(defaults, "cluster_metric_interval"))
        self.assertEqual(defaults.grad_clip, 1.0)
        self.assertEqual(defaults.checkpoint_policy, "auto")
        self.assertFalse(hasattr(defaults, "resume_checkpoint"))
        self.assertFalse(hasattr(defaults, "init_from_checkpoint"))

        enabled = train_cst_pred.parse_args([
            "--is_sample",
            "--data_root",
            "dataset",
            "--use_extra_features",
            "--overfit_one_batch",
        ])
        self.assertTrue(enabled.is_sample)
        self.assertEqual(enabled.data_root, "dataset")
        self.assertTrue(enabled.use_extra_features)
        self.assertTrue(enabled.overfit_one_batch)

    def test_env_reader_and_required_key(self):
        with patch.dict(os.environ, {"SWANLAB_API_KEY": "different-key"}):
            with tempfile.TemporaryDirectory() as directory:
                env_path = Path(directory) / ".env"
                env_path.write_text(
                    "# comment\nexport SWANLAB_API_KEY='test-secret-key'\n",
                    encoding="utf-8",
                )
                self.assertEqual(
                    read_env_file(env_path),
                    {"SWANLAB_API_KEY": "test-secret-key"},
                )
                self.assertEqual(require_swanlab_api_key(env_path), "test-secret-key")

                env_path.write_text(
                    "SWANLAB_API_KEY=your_swanlab_api_key_here\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "placeholder"):
                    require_swanlab_api_key(env_path)
                env_path.write_text("WANDB_API_KEY=old-key\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "SWANLAB_API_KEY"):
                    require_swanlab_api_key(env_path)

    def test_nested_metrics_are_fully_flattened(self):
        flattened = flatten_swanlab_metrics(
            "val/metric",
            {
                "mean_iou": 0.5,
                "per_class_iou": [0.25, 0.75],
                "confusion_matrix": [[3, 1], [2, 4]],
            },
        )
        self.assertEqual(flattened["val/metric/mean_iou"], 0.5)
        self.assertEqual(flattened["val/metric/per_class_iou/1"], 0.75)
        self.assertEqual(flattened["val/metric/confusion_matrix/1/0"], 2.0)

    def test_summary_metrics_skip_vectors_and_matrices(self):
        summaries = flatten_swanlab_summary_metrics(
            "val/metric",
            {
                "mean_iou": 0.5,
                "optimization": {"gradient_norm": 1.25},
                "per_class_iou": [0.25, 0.75],
                "confusion_matrix": [[3, 1], [2, 4]],
                "tensor_vector": np.asarray([1.0, 2.0]),
            },
        )
        self.assertEqual(
            summaries,
            {
                "val/metric/mean_iou": 0.5,
                "val/metric/optimization/gradient_norm": 1.25,
            },
        )

    def test_confusion_matrix_is_logged_as_one_native_panel(self):
        chart = Mock()
        fake_swanlab = SimpleNamespace(echarts=SimpleNamespace(HeatMap=lambda: chart))
        with patch.dict(sys.modules, {"swanlab": fake_swanlab}):
            result = swanlab_confusion_matrix(
                [[3, 1], [2, 4]],
                ["plane", "cylinder"],
                title="Primitive Confusion Matrix",
            )

        self.assertIs(result, chart)
        chart.add_xaxis.assert_called_once_with(["plane", "cylinder"])
        chart.add_yaxis.assert_called_once_with(
            "Count", ["plane", "cylinder"], [[0, 0, 3], [1, 0, 1], [0, 1, 2], [1, 1, 4]]
        )
        self.assertEqual(chart.set_global_opts.call_args.kwargs["visualmap_opts"]["max"], 4)

    def test_initializer_authenticates_from_env_and_forces_online_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("SWANLAB_API_KEY=test-key\n", encoding="utf-8")
            fake_run = object()
            fake_swanlab = SimpleNamespace(
                login=Mock(),
                init=Mock(return_value=fake_run),
            )
            with (
                patch.dict(sys.modules, {"swanlab": fake_swanlab}),
                patch.dict(os.environ, {}, clear=False),
            ):
                result = initialize_swanlab_run(
                    project="project",
                    workspace="team",
                    name="run",
                    config={"epochs": 2},
                    env_path=env_path,
                )

        self.assertIs(result, fake_run)
        fake_swanlab.login.assert_called_once_with(api_key="test-key", relogin=True, save=False)
        fake_swanlab.init.assert_called_once_with(
            project="project",
            workspace="team",
            experiment_name="run",
            config={"epochs": 2},
            mode="online",
            on_online_error="raise",
        )

    def test_initializer_resumes_exact_swanlab_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("SWANLAB_API_KEY=test-key\n", encoding="utf-8")
            fake_swanlab = SimpleNamespace(
                login=Mock(),
                init=Mock(return_value=SimpleNamespace(id="run-123")),
            )
            with patch.dict(sys.modules, {"swanlab": fake_swanlab}), patch.dict(os.environ):
                initialize_swanlab_run(
                    project="project",
                    workspace="team",
                    name="pointnet2",
                    config={"model": "pointnet2"},
                    run_id="run-123",
                    env_path=env_path,
                )

        fake_swanlab.init.assert_called_once_with(
            project="project",
            workspace="team",
            experiment_name="pointnet2",
            config={"model": "pointnet2"},
            mode="online",
            on_online_error="raise",
            id="run-123",
            resume="must",
        )

    def test_checkpoint_swanlab_run_id_reader_supports_new_and_legacy_files(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "last.pth"
            torch.save({"model": {}, "swanlab_run_id": "stable-id"}, checkpoint)
            self.assertEqual(
                read_swanlab_run_id_from_checkpoint(checkpoint), "stable-id"
            )
            torch.save({"model": {}, "wandb_run_id": "old-id"}, checkpoint)
            self.assertEqual(read_swanlab_run_id_from_checkpoint(checkpoint), "")

    def test_classification_metrics_include_per_class_and_confusion_data(self):
        scores = [
            np.asarray(
                [
                    [0.8, 0.1, 0.1],
                    [0.2, 0.7, 0.1],
                    [0.1, 0.2, 0.7],
                    [0.2, 0.6, 0.2],
                ],
                dtype=np.float32,
            )
        ]
        labels = [np.asarray([0, 1, 2, 2], dtype=np.int64)]
        metrics = all_metric_cls(scores, labels)

        self.assertIn("instance_accuracy", metrics)
        self.assertAlmostEqual(metrics["top_1_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["top_5_accuracy"], 1.0)
        self.assertEqual(metrics["top_5_effective_k"], 3)
        self.assertEqual(len(metrics["per_class_precision"]), 3)
        self.assertEqual(len(metrics["per_class_average_precision"]), 3)
        self.assertEqual(np.asarray(metrics["confusion_matrix"]).shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
