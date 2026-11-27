import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from networks.classification_models import classification_model_config, classification_model_uses_constraints
from train_cls import classification_batch
from train_cls_wheels import WheelPairDataset, main, parse_args, wheel_splits


class WheelClassificationTest(unittest.TestCase):
    def test_training_and_test_loaders_use_the_selected_representation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roots = {9: root / "legacy", 12: root / "standard"}
            for columns, directory in roots.items():
                cloud = np.zeros((4, columns), dtype=np.float32)
                cloud[:, 0] = columns  # Identify which root supplied XYZ.
                cloud[:, 3] = 1        # Cylinder
                cloud[:, 6] = 1        # Z axis
                cloud[:, 7] = 1 if columns == 9 else 2.5  # nearby or radius
                if columns == 12:
                    cloud[:, 8:11] = [3, 4, 0]
                cloud[:, -1] = 42      # Instance ID is never a model feature.
                for name in ["coaxis", *(f"offset-{i}" for i in range(1, 11))]:
                    (directory / name).mkdir(parents=True)
                    for i in range(2):
                        rows = cloud[:, :8] if columns == 9 and i == 0 else cloud
                        np.savetxt(directory / name / f"{i}.txt", rows)

            for model, flag, columns, channels in (
                ("cstnet", False, 9, 10), ("cstnet", True, 9, 10),
                ("constraint_aware", False, 12, 12), ("constraint_aware", True, 12, 12),
                ("pointnet", False, 12, 0), ("pointnet", True, 12, 12),
            ):
                with self.subTest(model=model, baseline_use_constraints=flag):
                    command = ["train_cls_wheels.py", "--model", model, "--data_root", str(roots[12]),
                               "--data_root_cstnet", str(roots[9]), "--output_dir", str(root / "out"),
                               "--workers", "0", "--n_points", "4"]
                    if flag:
                        command.append("--baseline_use_constraints")
                    with patch("sys.argv", command):
                        args = parse_args()
                    with patch("train_cls_wheels.train_classification", return_value={}) as train:
                        main(args)
                    self.assertEqual(train.call_count, 10)
                    for call in train.call_args_list:
                        args, train_loader, test_loader, selected_root, *_ = call.args
                        uses_constraints = classification_model_uses_constraints(classification_model_config(args))
                        self.assertEqual(uses_constraints, bool(channels))
                        self.assertEqual(Path(selected_root), roots[columns])
                        for loader in (train_loader, test_loader):
                            data = next(iter(loader))
                            xyz, _, attributes = classification_batch(data, torch.device("cpu"), uses_constraints)
                            self.assertTrue((xyz[..., 0] == columns).all())
                            if channels == 0:
                                self.assertEqual(len(data), 2)
                                self.assertIsNone(attributes)
                            else:
                                expected = ([0, 0, 1, 0, 1, 0, 1, 0, 0, 0] if columns == 9 else
                                            [0, 1, 0, 0, 0, 0, 0, 1, 2.5, 3, 4, 0])
                                torch.testing.assert_close(attributes, attributes.new_tensor(expected).expand_as(attributes))

    def test_reproducible_binary_splits_with_unequal_class_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            offsets = [f"offset-{i}" for i in range(1, 11)]
            for name in ["coaxis", *offsets]:
                directory = root / name
                directory.mkdir()
                for i in range(10 if name == "coaxis" else 5):
                    (directory / f"{name}_aug_{i:04}.txt").touch()

            splits = wheel_splits(root)
            self.assertEqual(wheel_splits(root), splits)
            self.assertEqual(list(splits), ["coaxis", *offsets])
            for name in offsets:
                train, test = [
                    WheelPairDataset(splits, name, part, 16, False)
                    for part in ("train", "test")
                ]
                self.assertEqual(train.classes, {"coaxis": 0, name: 1})
                self.assertEqual(len(train), 12)
                self.assertEqual(len(test), 3)
                self.assertEqual([p for p, label in test.samples if label == 0], splits["coaxis"]["test"])
                self.assertTrue(set(train.samples).isdisjoint(test.samples))
                self.assertEqual({p.parent.name for p, label in train.samples if label == 1}, {name})

    def test_legacy_wheel_inputs_for_xyz_and_cstnet(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wheel.txt"
            cloud = np.zeros((32, 9), dtype=np.float32)
            cloud[:, 0] = np.arange(32)
            cloud[:, 3] = 4
            cloud[:, 7] = 1
            cloud[:, 8] = 42  # Primitive instance ID must never be used as a feature.
            np.savetxt(path, cloud)
            splits = {"coaxis": {"test": [path]}, "offset-1": {"test": []}}
            xyz, label = WheelPairDataset(splits, "offset-1", "test", 16, False)[0]
            xyz_again, _, features = WheelPairDataset(splits, "offset-1", "test", 16, True)[0]
            np.testing.assert_array_equal(xyz_again, xyz)
            self.assertEqual(label, 0)
            self.assertEqual(features.shape, (16, 10))
            np.testing.assert_array_equal(features[:, [4, 9]], np.ones((16, 2)))
            np.savetxt(path, cloud[:, :8])
            short_xyz, short_label, short_features = WheelPairDataset(splits, "offset-1", "test", 16, True)[0]
            np.testing.assert_array_equal(short_xyz, xyz_again)
            self.assertEqual(short_label, label)
            np.testing.assert_array_equal(short_features, features)

    def test_same_xyz_uses_constraints_from_each_selected_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            for source, radius in (("gt", 2.5), ("predicted", 1.75)):
                with self.subTest(source=source):
                    path = Path(tmp) / f"{source}.txt"
                    rows = np.zeros((4, 12), dtype=np.float32)
                    rows[:, 3] = 1  # cylinder
                    rows[:, 6] = 1  # z axis
                    rows[:, 7] = radius
                    np.savetxt(path, rows)
                    splits = {"coaxis": {"test": [path]}, "offset-1": {"test": []}}
                    dataset = WheelPairDataset(splits, "offset-1", "test", 4, False, use_stored_constraints=True)
                    data = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
                    _, _, constraints = classification_batch(data, torch.device("cpu"), True)
                    torch.testing.assert_close(constraints[..., 8], torch.full((1, 4), radius))
