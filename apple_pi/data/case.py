"""Paper-compatible case loader for the public GT layout."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


@dataclass(frozen=True)
class CaseData:
    case_id: str
    case_dir: Path
    annotation_text: str
    first_frame_path: Path
    clean_first_frame_path: Path
    gt_frames_dir: Path
    white_bg_first_frame_path: Path | None
    white_bg_obj_first_frame_path: Path | None
    subtrack_image_path: Path | None
    gt_video_path: Path | None
    target_time: float
    physics_duration: float
    gt_fps: float
    gt_total_frames: int
    physics_type: str
    is_realworld: bool
    formula_info: dict[str, Any]
    deduction_timestamps: tuple[float, ...]

    @classmethod
    def load(cls, case_id: str, case_dir: str | Path) -> "CaseData":
        case_dir = Path(case_dir).expanduser().resolve()
        metadata_path = case_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing case metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        def required(relative: str) -> Path:
            path = case_dir / relative
            if not path.exists():
                raise FileNotFoundError(f"Missing required GT asset: {path}")
            return path

        def optional(relative: str | None) -> Path | None:
            if not relative:
                return None
            path = case_dir / relative
            return path if path.exists() else None

        gt_video = optional(metadata.get("gt_video"))
        gt_fps = float(metadata.get("gt_fps", 24))
        gt_total_frames = int(metadata.get("gt_total_frames", 0))
        if gt_video is not None and (gt_total_frames <= 0 or gt_fps <= 0):
            cap = cv2.VideoCapture(str(gt_video))
            detected_fps = float(cap.get(cv2.CAP_PROP_FPS))
            detected_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if detected_fps > 0:
                gt_fps = detected_fps
            if detected_total > 0:
                gt_total_frames = detected_total

        formula_info = dict(metadata["formula_info"])
        return cls(
            case_id=str(case_id),
            case_dir=case_dir,
            annotation_text=str(metadata["annotation"]),
            first_frame_path=required(metadata["input_image"]),
            clean_first_frame_path=required(metadata["clean_first_frame"]),
            gt_frames_dir=required(metadata["gt_frames_dir"]),
            white_bg_first_frame_path=optional(
                metadata.get("annotations_only_reference")
            ),
            white_bg_obj_first_frame_path=optional(
                metadata.get("objects_only_reference")
            ),
            subtrack_image_path=optional(metadata.get("future_state_reference")),
            gt_video_path=gt_video,
            target_time=float(metadata["target_time"]),
            physics_duration=float(metadata["physics_duration"]),
            gt_fps=gt_fps,
            gt_total_frames=gt_total_frames,
            physics_type=str(metadata["physics_type"]),
            is_realworld=bool(metadata["is_realworld"]),
            formula_info=formula_info,
            deduction_timestamps=tuple(
                float(value) for value in metadata["deduction_timestamps"]
            ),
        )

    def prompt_record(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "input_image": str(self.first_frame_path),
            "formula_choices": self.formula_info["choices"],
            "target_time": self.target_time,
            "physics_duration": self.physics_duration,
            "deduction_timestamps": self.deduction_timestamps,
        }
