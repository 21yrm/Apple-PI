"""Preflight validation for GT and three-rollout predictions."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

from apple_pi.constants import NUM_ROLLOUTS
from apple_pi.types import InferenceProtocol, Subtrack

from .case import CaseData
from .dataset import load_dataset_index
from .predictions import prediction_path


def _video_readable(path: Path) -> bool:
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    cap.release()
    return ok


def _check_maps(
    path: Path,
    expected_ndim: int,
    errors: list[str],
    label: str,
) -> tuple[int, ...] | None:
    if not path.is_file():
        errors.append(f"Missing {label}: {path}")
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            if "maps.npy" not in archive.namelist():
                errors.append(f"{label} must contain NPZ key 'maps': {path}")
                return None
            with archive.open("maps.npy") as member:
                version = np.lib.format.read_magic(member)
                shape, _, _ = np.lib.format._read_array_header(member, version)
    except Exception as exc:
        errors.append(f"Cannot inspect {label} {path}: {exc}")
        return None
    if len(shape) != expected_ndim:
        errors.append(
            f"{label} expected {expected_ndim} dimensions, got {shape}: {path}"
        )
    return tuple(int(dimension) for dimension in shape)


def _validate_case_assets(case: CaseData, errors: list[str]) -> None:
    prefix = case.case_id
    required_references = {
        "annotations-only reference": case.white_bg_first_frame_path,
        "objects-only reference": case.white_bg_obj_first_frame_path,
        "future-state reference": case.subtrack_image_path,
    }
    for label, path in required_references.items():
        if path is None:
            errors.append(f"{prefix}: missing {label}")

    initial = case.case_dir / "initial_state"
    instantaneous = case.case_dir / "instantaneous_velocity"
    required_initial = [
        initial / "instance_segmentation_0000.npy",
    ]
    if not case.is_realworld:
        required_initial.extend(
            [
                initial / "mask_0000.npy",
                initial / "instance_segmentation_mapping_0000.json",
            ]
        )
    for path in required_initial:
        if not path.is_file():
            errors.append(f"{prefix}: missing SAM3 GT asset {path}")

    if not (instantaneous / "mask.npy").is_file():
        errors.append(
            f"{prefix}: missing future-state mask {instantaneous / 'mask.npy'}"
        )
    if not case.is_realworld and not (instantaneous / "mapping.json").is_file():
        errors.append(
            f"{prefix}: missing future-state mapping {instantaneous / 'mapping.json'}"
        )

    instance_shape = _check_maps(
        case.case_dir / "instance_segmentation" / "maps.npz",
        3,
        errors,
        f"{prefix} instance segmentation",
    )
    foreground_shape = _check_maps(
        case.case_dir / "mask" / "maps.npz",
        3,
        errors,
        f"{prefix} foreground masks",
    )
    if (
        instance_shape is not None
        and foreground_shape is not None
        and instance_shape != foreground_shape
    ):
        errors.append(
            f"{prefix}: instance/mask shape mismatch: "
            f"{instance_shape} vs {foreground_shape}"
        )

    if not case.is_realworld:
        depth_shape = _check_maps(
            case.case_dir / "depth" / "maps.npz",
            3,
            errors,
            f"{prefix} depth",
        )
        velocity_shape = _check_maps(
            case.case_dir / "velocity" / "maps.npz",
            4,
            errors,
            f"{prefix} velocity",
        )
        if velocity_shape is not None and velocity_shape[-1] != 3:
            errors.append(
                f"{prefix}: velocity maps must end in xyz dimension 3, "
                f"got {velocity_shape}"
            )
        if (
            depth_shape is not None
            and instance_shape is not None
            and depth_shape != instance_shape
        ):
            errors.append(
                f"{prefix}: depth/instance shape mismatch: "
                f"{depth_shape} vs {instance_shape}"
            )
        camera_dir = case.case_dir / "camera_parameters"
        if not camera_dir.is_dir() or not any(camera_dir.glob("*.json")):
            errors.append(f"{prefix}: missing camera parameter JSON files")


def validate_ground_truth(root: str | Path) -> list[str]:
    errors: list[str] = []
    try:
        dataset = load_dataset_index(root)
    except Exception as exc:
        return [str(exc)]
    for entry in dataset.cases:
        try:
            case = CaseData.load(entry.case_id, dataset.root / entry.path)
            if case.gt_video_path and not _video_readable(case.gt_video_path):
                errors.append(f"{entry.case_id}: unreadable GT video")
            if len(case.formula_info.get("choices", [])) != 4:
                errors.append(f"{entry.case_id}: formula choices must contain 4 items")
            if case.target_time < 0:
                errors.append(f"{entry.case_id}: target_time must be non-negative")
            if any(
                time <= 0 or time > case.physics_duration
                for time in case.deduction_timestamps
            ):
                errors.append(
                    f"{entry.case_id}: deduction_timestamps must be in "
                    f"(0, physics_duration]"
                )
            _validate_case_assets(case, errors)
        except Exception as exc:
            errors.append(f"{entry.case_id}: {exc}")
    return errors


def validate_predictions(
    gt_root: str | Path,
    prediction_root: str | Path,
    *,
    subtracks: tuple[Subtrack, ...] = tuple(Subtrack),
) -> list[str]:
    errors: list[str] = []
    prediction_root = Path(prediction_root)
    submission_path = prediction_root / "submission.json"
    if not submission_path.is_file():
        return [f"Missing {submission_path}"]
    try:
        submission = json.loads(submission_path.read_text(encoding="utf-8"))
        protocol = InferenceProtocol(submission["protocol"])
    except Exception as exc:
        return [f"Invalid submission.json: {exc}"]
    if int(submission.get("num_rollouts", -1)) != NUM_ROLLOUTS:
        errors.append(f"submission.json must declare num_rollouts={NUM_ROLLOUTS}")

    dataset = load_dataset_index(gt_root)
    for entry in dataset.cases:
        case = CaseData.load(entry.case_id, dataset.root / entry.path)
        for subtrack in subtracks:
            for rollout in range(NUM_ROLLOUTS):
                path = prediction_path(
                    prediction_root,
                    entry.case_id,
                    protocol,
                    subtrack,
                    rollout,
                )
                if protocol == InferenceProtocol.VIDEO:
                    if not path.is_file():
                        errors.append(f"Missing prediction: {path}")
                    elif not _video_readable(path):
                        errors.append(f"Unreadable video: {path}")
                elif subtrack != Subtrack.DEDUCTION:
                    if not path.is_file():
                        errors.append(f"Missing prediction: {path}")
                    elif cv2.imread(str(path), cv2.IMREAD_COLOR) is None:
                        errors.append(f"Unreadable image: {path}")
                else:
                    if not path.is_dir():
                        errors.append(f"Missing keyframe directory: {path}")
                        continue
                    for index, time_point in enumerate(
                        case.deduction_timestamps
                    ):
                        expected = path / (f"frame_{index:03d}_t{time_point:.3f}s.png")
                        if not expected.is_file():
                            errors.append(f"Missing keyframe: {expected}")
    return errors
