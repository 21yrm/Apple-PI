"""Ground-truth RGB frame loading for programmatic metrics."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _decode_video(path: Path, skip_initial: bool) -> tuple[list[np.ndarray], float | None]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return [], None

    detected_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if detected_fps <= 0:
        detected_fps = None

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if skip_initial and frames:
        frames = frames[1:]
    return frames, detected_fps


def load_gt_frames(
    case_dir: str | Path,
    *,
    gt_video_path: str | Path | None = None,
    skip_initial: bool = True,
) -> tuple[list[np.ndarray], float | None]:
    """Load the GT RGB source declared by case metadata.

    A non-null ``gt_video_path`` is a complete GT sequence whose frame 0 is
    the condition frame. It takes precedence over the standalone
    ``rgb/0000.png`` condition-frame copy. When no GT video is declared, the
    simulator PNG sequence under ``rgb/`` is used.
    """
    case_dir = Path(case_dir)
    if gt_video_path is not None:
        return _decode_video(Path(gt_video_path), skip_initial)

    gt_frames_dir = case_dir / "rgb"
    gt_files = sorted(gt_frames_dir.glob("*.png"))
    if gt_files:
        if skip_initial:
            gt_files = [path for path in gt_files if path.name != "0000.png"]
        frames = []
        for path in gt_files:
            frame = cv2.imread(str(path))
            if frame is not None:
                frames.append(frame)
        return frames, None

    for candidate in (gt_frames_dir / "video.mp4", case_dir / "video.mp4"):
        if candidate.exists():
            return _decode_video(candidate, skip_initial)
    return [], None
