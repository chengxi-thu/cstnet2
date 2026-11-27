import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from data_utils.constraint_dataset_common import cstnet_attributes
from data_utils.classification_dataset import Stage2ClassificationDataset
from data_utils.mfcad_seg_dataset import Stage2SegmentationDataset
from networks.classification_models import build_classification_model, classification_model_uses_constraints
from networks.cstnet import _stage2_inputs
from networks.segmentation_models import build_segmentation_model


class CstNetCompatibilityTest(unittest.TestCase):
    def test_training_entry_points_select_the_model_specific_root(self):
        import train_cls
        import train_seg

        for trainer, dataset, options in (
            (train_cls, Stage2ClassificationDataset, []),
            (train_seg, Stage2SegmentationDataset, ["--not_resume"]),
        ):
            for model, expected in (("cstnet", "legacy"), ("pointnet", "standard")):
                with self.subTest(script=trainer.__name__, model=model):
                    args = trainer.parse_args([
                        "--model", model, "--data_root", "standard",
                        "--data_root_cstnet", "legacy", *options,
                    ])
                    with patch.object(dataset, "create_dataloaders", side_effect=StopIteration) as load:
                        with self.assertRaises(StopIteration):
                            trainer.main(args)
                    self.assertEqual(load.call_args.kwargs["root"], expected)
                    self.assertEqual(load.call_args.kwargs["cstnet_inputs"], model == "cstnet")

    def test_legacy_edges_and_direction_reach_the_encoder(self):
        rows = np.zeros((2, 9), dtype=np.float32)
        rows[:, 3] = (1, 4)
        rows[:, 4:7] = ((1, 2, 3), (4, 5, 6))
        rows[:, 7] = (1, 0)
        packed = torch.from_numpy(cstnet_attributes(rows))[None]
        xyz = torch.zeros(1, 2, 3)
        _, direction, edge, primitive = _stage2_inputs(xyz, packed, 5)
        torch.testing.assert_close(direction[0], torch.from_numpy(rows[:, 4:7]))
        torch.testing.assert_close(edge[0], torch.tensor([[0., 1.], [1., 0.]]))
        self.assertEqual(primitive.argmax(-1).tolist(), [[1, 4]])
        rows[0, 7] = 0.5
        with self.assertRaisesRegex(ValueError, "nearby"):
            cstnet_attributes(rows)

    def test_model_adapter_requires_legacy_attributes(self):
        with self.assertRaisesRegex(ValueError, "nearby"):
            _stage2_inputs(torch.zeros(1, 2, 3), torch.zeros(1, 2, 12), 5)

    def test_training_datasets_require_recorded_binary_edge_labels(self):
        for dataset_type, width, subdir in (
            (Stage2ClassificationDataset, 8, "test/gear"),
            (Stage2ClassificationDataset, 9, "test/gear"),
            (Stage2SegmentationDataset, 11, "val"),
        ):
            with self.subTest(width=width), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "train").mkdir()
                sample = root / subdir / "0.txt"
                sample.parent.mkdir(parents=True)
                rows = np.zeros((4, width), dtype=np.float32)
                rows[:, 0] = np.arange(4)
                rows[:, 7] = (0, 1, 0, 1)
                np.savetxt(sample, rows)
                # Auto selection must still use legacy TXT in a mixed directory.
                (root / "stage2.h5").touch()
                split = subdir.split("/")[0]
                dataset = dataset_type(root, split, n_points=4, cstnet_inputs=True)
                item = dataset[0]
                xyz, attributes = (item["xyz"], item["constraints"]) if width == 11 else (item[0], item[2])
                np.testing.assert_array_equal(attributes[:, 3:5], np.eye(2)[np.asarray(xyz[:, 0], int) % 2])
                dataset.n_points = 1
                rows[-1, 7] = -1
                np.savetxt(sample, rows)
                with self.assertRaisesRegex(ValueError, "nearby labels must be 0 or 1"):
                    dataset[0]
                # Even a dimension containing only 0/1 is not a recorded edge label.
                np.savetxt(sample, np.zeros((4, width + 3), dtype=np.float32))
                expected_width = "8 or 9" if dataset_type is Stage2ClassificationDataset else str(width)
                with self.assertRaisesRegex(ValueError, f"expected {expected_width} columns"):
                    dataset[0]
                with self.assertRaisesRegex(ValueError, "legacy TXT"):
                    dataset_type(root, split, storage_format="h5", cstnet_inputs=True)

    def test_task_factories_keep_unprefixed_checkpoint_keys(self):
        for build, width, classes, head in (
            (build_classification_model, 4, 68, "fc3.weight"),
            (build_segmentation_model, 5, 25, "afterprocess.outlayer.weight"),
        ):
            config = {"model": "cstnet", "cstnet_n_metatype": width}
            self.assertTrue(classification_model_uses_constraints(config))
            state = build(classes, config).state_dict()
            self.assertEqual(state["preprocess.linear_layers.0.weight"].shape, (16, 8 + width, 1))
            self.assertEqual(state[head].shape[0], classes)
            self.assertIn("sa5.attention_points.gamma.outlayer.weight", state)
