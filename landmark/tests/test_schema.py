from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import yaml

from landmark import train as train_module
from landmark.data import schema
from landmark.data.prepare import prepare_dataset
from landmark.utils.api import KneePose

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HIP_YAML = REPOSITORY_ROOT / "landmark" / "cfg" / "datasets" / "hip_point.yaml"
MODEL_ROOT = REPOSITORY_ROOT / "landmark" / "cfg" / "models"
HIP_COUNTS = (59, 7, 9, 10)


def _run_with_schema(code: str, schema_yaml: Path) -> subprocess.CompletedProcess:
    """The schema is fixed at import, so each non-default schema needs a fresh interpreter."""
    env = {**os.environ, schema.SCHEMA_ENV: str(schema_yaml), "PYTHONPATH": str(REPOSITORY_ROOT)}
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=REPOSITORY_ROOT
    )


def _write_hip_dataset(root: Path, images: int = 4) -> None:
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    for index in range(images):
        (root / "images" / "train" / f"case{index}.png").write_bytes(b"\x89PNG")
        rows = []
        for class_id, count in enumerate(HIP_COUNTS):
            keypoints = [0.0] * (59 * 3)
            for point in range(count):
                keypoints[point * 3 : point * 3 + 3] = [0.1 + 0.01 * point, 0.2, 2.0]
            rows.append(" ".join(map(str, [class_id, 0.5, 0.5, 0.2, 0.2, *keypoints])))
        (root / "labels" / "train" / f"case{index}.txt").write_text("\n".join(rows), encoding="utf-8")


class KneeDefaultSchemaTests(unittest.TestCase):
    def test_default_schema_is_the_original_knee_layout(self):
        self.assertEqual(schema.REGION_NAMES, ("femur", "tibia", "fibula", "patella"))
        self.assertEqual(schema.REGION_KEYPOINT_COUNTS, (45, 51, 24, 9))
        self.assertEqual(schema.REGION_OFFSETS, (0, 45, 96, 120))
        self.assertEqual((schema.MAX_REGION_KEYPOINTS, schema.NUM_LANDMARKS), (51, 129))
        self.assertEqual(
            schema.LANDMARK_PATH_RANGES,
            ((0, 45), (45, 86), (86, 91), (91, 96), (96, 120), (120, 129)),
        )
        self.assertEqual(
            schema.REGION_LOCAL_PATHS,
            (((0, 45),), ((0, 41), (41, 46), (46, 51)), ((0, 24),), ((0, 9),)),
        )
        self.assertEqual(int(schema.POINT_REGION_IDS.numel()), 129)

    def test_knee_yaml_matches_the_default_schema(self):
        knee_yaml = REPOSITORY_ROOT / "landmark" / "cfg" / "datasets" / "mesko4gf2.yaml"
        self.assertEqual(schema.load_schema(knee_yaml), schema.KNEE_SCHEMA)

    def test_tibia_paths_never_connect_across_boundaries(self):
        masks = schema.class_path_masks(torch.tensor([1]), order=2)[0]
        self.assertTrue(masks[39] and not masks[40] and masks[41] and not masks[45] and masks[46])

    def test_schema_env_name_is_shared_with_train_bootstrap(self):
        self.assertEqual(train_module.SCHEMA_ENV, schema.SCHEMA_ENV)

    def test_knee_models_build_and_hip_models_are_rejected_under_knee_schema(self):
        self.assertEqual(list(KneePose(MODEL_ROOT / "rtmo-pose.yaml").model.kpt_shape), [51, 3])
        head = KneePose(MODEL_ROOT / "yolo26-pose-v9.yaml").model.model[-1]
        self.assertEqual((list(head.kpt_shape), head.heatmap_channels), ([51, 3], 129))
        with self.assertRaisesRegex(ValueError, "rtmo-pose-hip.yaml does not match the active landmark schema"):
            KneePose(MODEL_ROOT / "rtmo-pose-hip.yaml")
        with self.assertRaisesRegex(ValueError, r"kpt_shape=\[51, 3\], got \[59, 3\].*yolo26-pose-hip"):
            KneePose(MODEL_ROOT / "yolo26-pose-hip.yaml")


class SchemaValidationTests(unittest.TestCase):
    def test_hip_yaml_defines_a_valid_schema(self):
        hip = schema.load_schema(HIP_YAML)
        self.assertEqual(hip.counts, HIP_COUNTS)
        self.assertEqual(hip.offsets, (0, 59, 66, 75))
        self.assertEqual((hip.max_keypoints, hip.total), (59, 85))
        self.assertEqual(hip.local_paths, (((0, 59),), ((0, 7),), ((0, 9),), ((0, 10),)))

    def test_yaml_without_region_counts_declares_no_schema(self):
        self.assertIsNone(schema.schema_from_metadata({"dataset_name": "x"}))

    def test_inconsistent_schemas_fail_fast(self):
        names = ("a", "b", "c", "d")
        with self.assertRaisesRegex(ValueError, "exactly 4 regions"):
            schema.build_schema(names[:3], (5, 5, 5))
        with self.assertRaisesRegex(ValueError, "kpt_shape"):
            schema.build_schema(names, (5, 7, 9, 10), kpt_shape=(59, 3))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            schema.build_schema(names, (5, 7, 9, 10), path_ranges=[(0, 5), (6, 12), (12, 21), (21, 31)])
        with self.assertRaisesRegex(ValueError, "crosses a region boundary"):
            schema.build_schema(names, (5, 7, 9, 10), path_ranges=[(0, 8), (8, 12), (12, 21), (21, 31)])
        with self.assertRaisesRegex(ValueError, "end at"):
            schema.build_schema(names, (5, 7, 9, 10), path_ranges=[(0, 5), (5, 12), (12, 21)])


class HipSchemaTests(unittest.TestCase):
    def test_constants_follow_the_dataset_yaml(self):
        result = _run_with_schema(
            "from landmark.data import schema as s;"
            "print(s.REGION_NAMES, s.NUM_LANDMARKS, s.MAX_REGION_KEYPOINTS, s.REGION_OFFSETS);"
            "print(s.class_keypoint_mask(__import__('torch').tensor([1])).sum().item())",
            HIP_YAML,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("('femur', 'lesser_trochanter', 'greater_trochanter', 'acetabulum') 85 59 (0, 59, 66, 75)", result.stdout)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "7")

    def test_dataset_is_accepted_only_under_its_own_schema(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "HipX02"
            _write_hip_dataset(root)
            metadata = yaml.safe_load(HIP_YAML.read_text(encoding="utf-8"))
            metadata["path"] = str(root)
            source = Path(directory) / "hip.yaml"
            source.write_text(yaml.safe_dump(metadata), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, f"{schema.SCHEMA_ENV}"):
                prepare_dataset(source, cache_root=Path(directory) / "cache")

            result = _run_with_schema(
                "from landmark.data.prepare import prepare_dataset;"
                f"p = prepare_dataset({str(source)!r}, cache_root={str(Path(directory) / 'cache')!r});"
                "print(len(p.train_images), len(p.val_images))",
                HIP_YAML,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip().splitlines()[-1], "3 1")

    def test_hip_model_yamls_build_and_knee_model_yamls_are_rejected(self):
        program = f"""
from landmark.utils.api import KneePose
root = {str(MODEL_ROOT)!r}
rtmo = KneePose(root + "/rtmo-pose-hip.yaml").model
print(list(rtmo.kpt_shape), rtmo.nc)
head = KneePose(root + "/yolo26-pose-hip.yaml").model.model[-1]
print(list(head.kpt_shape), head.heatmap_channels)
errors = []
for name in ("rtmo-pose.yaml", "yolo26-pose-v9.yaml"):
    try:
        KneePose(root + "/" + name)
    except ValueError as exc:
        errors.append(str(exc))
print(len(errors), all("hip" in message for message in errors))
"""
        result = _run_with_schema(program, HIP_YAML)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.strip().splitlines()
        self.assertEqual(lines[-3:], ["[59, 3] 4", "[59, 3] 85", "2 True"])

    def test_rtmo_outputs_the_hip_landmark_count(self):
        result = _run_with_schema(
            "import torch;"
            "from landmark.models.rtmo import RTMOKneePose;"
            "m = RTMOKneePose().eval();"
            "print(tuple(m(torch.zeros(1, 3, 64, 64)).shape))",
            HIP_YAML,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "(1, 85, 3)")


class TrainBootstrapTests(unittest.TestCase):
    def test_dataset_yaml_with_schema_is_exported_for_model_imports(self):
        with patch.dict(os.environ, clear=False):
            os.environ.pop(schema.SCHEMA_ENV, None)
            train_module.activate_dataset_schema(["--dataset", str(HIP_YAML)])
            self.assertEqual(os.environ[schema.SCHEMA_ENV], str(HIP_YAML.resolve()))

    def test_dataset_folder_uses_its_yaml(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, clear=False):
            os.environ.pop(schema.SCHEMA_ENV, None)
            folder = Path(directory) / "HipX02"
            folder.mkdir()
            (folder / "data_points.yaml").write_text(HIP_YAML.read_text(encoding="utf-8"), encoding="utf-8")
            train_module.activate_dataset_schema(["--data", str(folder)])
            self.assertEqual(os.environ[schema.SCHEMA_ENV], str((folder / "data_points.yaml").resolve()))

    def test_dataset_without_schema_or_missing_dataset_leaves_default(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, clear=False):
            os.environ.pop(schema.SCHEMA_ENV, None)
            plain = Path(directory) / "plain.yaml"
            plain.write_text(yaml.safe_dump({"dataset_name": "plain"}), encoding="utf-8")
            train_module.activate_dataset_schema(["--dataset", str(plain)])
            train_module.activate_dataset_schema(["--dataset", str(Path(directory) / "missing.yaml")])
            train_module.activate_dataset_schema([])
            self.assertNotIn(schema.SCHEMA_ENV, os.environ)


if __name__ == "__main__":
    unittest.main()
