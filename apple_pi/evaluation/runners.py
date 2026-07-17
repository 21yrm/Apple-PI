"""Paper-compatible evaluation runners for video and image protocols."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from apple_pi.data.case import CaseData
from apple_pi.types import Subtrack
from apple_pi.utils import (
    bytes_to_data_url,
    extract_last_frame,
    frame_to_jpeg_bytes,
    image_to_data_url,
    parse_llm_json,
)

from .paper_evaluator import SubtrackScores, VideoEvaluator

logger = logging.getLogger(__name__)


def _formula_info(case: CaseData) -> dict[str, str]:
    cached = case.formula_info
    return {
        "correct_letter": str(cached["correct_letter"]),
        "correct_formula": str(cached["correct_formula"]),
        "annotation": str(cached.get("annotation", case.annotation_text)),
    }


def _evaluate_non_deduction_frame(
    frame: np.ndarray,
    case: CaseData,
    subtrack: Subtrack,
    evaluator: VideoEvaluator,
) -> SubtrackScores:
    first_frame_data_url = image_to_data_url(case.first_frame_path)
    annotations_ref = None
    if (
        subtrack in {Subtrack.PERCEPTION_TEXT, Subtrack.PERCEPTION_GRAPHIC}
        and case.white_bg_first_frame_path
    ):
        annotations_ref = image_to_data_url(case.white_bg_first_frame_path)

    objects_ref = None
    if subtrack == Subtrack.PERCEPTION_GRAPHIC and case.white_bg_obj_first_frame_path:
        objects_ref = image_to_data_url(case.white_bg_obj_first_frame_path)

    future_ref = None
    if subtrack == Subtrack.FORMULATION_GRAPHIC and case.subtrack_image_path:
        future_ref = image_to_data_url(case.subtrack_image_path)

    scores = evaluator.evaluate(
        subtrack=subtrack,
        first_frame_data_url=first_frame_data_url,
        gen_frames=[frame],
        gen_video_bytes=None,
        gt_frames=None,
        subtrack_ref_data_url=future_ref,
        formula_info=(
            _formula_info(case) if subtrack == Subtrack.FORMULATION_TEXT else None
        ),
        first_frame_white_bg_data_url=annotations_ref,
        first_frame_white_bg_obj_data_url=objects_ref,
    )
    return scores


def evaluate_image_prediction(
    image_path: str | Path,
    case: CaseData,
    subtrack: Subtrack,
    evaluator: VideoEvaluator,
) -> SubtrackScores:
    """Evaluate an image-model output for one non-Deduction subtrack."""
    if subtrack == Subtrack.DEDUCTION:
        raise ValueError("Use evaluate_deduction_keyframes for Deduction")
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    scores = _evaluate_non_deduction_frame(frame, case, subtrack, evaluator)

    # Matches the paper image-evaluation implementation: SAM3 IoU is added for
    # perception_graphic here; the remaining fields come from Gemini.
    if subtrack == Subtrack.PERCEPTION_GRAPHIC:
        try:
            from .segmentation import compute_segmentation_iou

            generated_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            scores.segmentation_iou = compute_segmentation_iou(
                gen_image=generated_rgb,
                case_dir=str(case.case_dir),
                is_realworld=case.is_realworld,
            )
        except Exception as exc:
            logger.warning("SAM3 segmentation IoU failed: %s", exc)
            scores.segmentation_iou = 0.0
    return scores


def _programmatic_video_metrics(
    video_path: str | Path,
    case: CaseData,
) -> Optional[dict[str, float]]:
    from .programmatic_metrics import (
        Evaluator as ProgrammaticEvaluator,
        MaskedPSNRMetric,
        PSNRMetric,
        SpatialIoUMetric,
        SpatiotemporalIoUMetric,
        VelocityMetric,
        WeightedSpatialIoUMetric,
    )

    evaluator = ProgrammaticEvaluator(
        [PSNRMetric(), MaskedPSNRMetric()],
        [SpatialIoUMetric(), SpatiotemporalIoUMetric(), WeightedSpatialIoUMetric()],
        velocity_metric=VelocityMetric() if not case.is_realworld else None,
    )
    try:
        return evaluator.evaluate_video(
            video_path=str(video_path),
            case_dir=str(case.case_dir),
            gt_fps=case.gt_fps,
            gt_video_path=case.gt_video_path,
            physics_duration=case.physics_duration,
            is_realworld=case.is_realworld,
        )
    except Exception as exc:
        logger.exception("Programmatic metric computation failed: %s", exc)
        return None


def evaluate_video_prediction(
    video_path: str | Path,
    case: CaseData,
    subtrack: Subtrack,
    evaluator: VideoEvaluator,
) -> SubtrackScores:
    """Evaluate one video rollout using the paper implementation."""
    video_path = Path(video_path)
    if subtrack != Subtrack.DEDUCTION:
        frame = extract_last_frame(video_path)
        scores = _evaluate_non_deduction_frame(frame, case, subtrack, evaluator)
    else:
        scores = evaluator.evaluate(
            subtrack=subtrack,
            first_frame_data_url=image_to_data_url(case.first_frame_path),
            gen_frames=[],
            gen_video_bytes=video_path.read_bytes(),
            gt_frames=None,
        )

    if subtrack == Subtrack.PERCEPTION_GRAPHIC:
        try:
            from .segmentation import compute_segmentation_iou

            frame = extract_last_frame(video_path)
            scores.segmentation_iou = compute_segmentation_iou(
                gen_image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                case_dir=str(case.case_dir),
                is_realworld=case.is_realworld,
            )
        except Exception as exc:
            logger.warning("SAM3 segmentation IoU failed: %s", exc)
            scores.segmentation_iou = 0.0

    if subtrack == Subtrack.FORMULATION_GRAPHIC:
        try:
            from .segmentation import compute_segmentation_iou

            frame = extract_last_frame(video_path)
            scores.formulation_graphic_segmentation_iou = compute_segmentation_iou(
                gen_image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                case_dir=str(case.case_dir),
                is_realworld=case.is_realworld,
                mask_subdir="instantaneous_velocity",
            )
        except Exception as exc:
            logger.warning(
                "SAM3 segmentation IoU for formulation_graphic failed: %s",
                exc,
            )
            scores.formulation_graphic_segmentation_iou = 0.0

    if subtrack == Subtrack.DEDUCTION:
        programmatic = _programmatic_video_metrics(video_path, case)
        if programmatic:
            scores.gen_psnr = programmatic.get("psnr", -1.0)
            scores.gen_masked_psnr = programmatic.get("masked_psnr", -1.0)
            scores.gen_spatial_iou = programmatic.get("spatial_iou", -1.0)
            scores.gen_spatiotemporal_iou = programmatic.get("spatiotemporal_iou", -1.0)
            scores.gen_weighted_spatial_iou = programmatic.get(
                "weighted_spatial_iou", -1.0
            )
            scores.gen_velocity_error = programmatic.get("velocity_error", -1.0)
    return scores


EVAL_PROMPT_DEDUCTION_KEYFRAME = """\
You are evaluating a single keyframe from a physics simulation generation model.

You are provided with:
1. INITIAL ANNOTATED PHYSICS SCENE (the first frame with coordinate axes, text labels, etc.)
2. GROUND TRUTH FRAME at t = {time_point:.2f}s (the correct physical state at this moment)
3. GENERATED FRAME at t = {time_point:.2f}s (the model's output for this moment)

Compare the GENERATED FRAME against the GROUND TRUTH FRAME. Score each from 0.0 to 1.0:

- gen_annotations_removed: Are all human annotations from the INITIAL SCENE \
(text labels, coordinate axes, initial velocity arrows) removed in the generated frame? \
(0 = annotations clearly visible, 1 = perfectly clean)
- object_consistency: Do the physical objects match the GT in identity, count, and shape? \
(0 = objects missing/hallucinated/severely deformed, 1 = exact match)
- visual_quality: Overall visual quality — sharpness, clarity, absence of artifacts. \
(0 = severe artifacts/blur, 1 = high quality crisp rendering)
- object_position_match: Are the objects at the same positions as in the GT frame? \
(0 = completely wrong positions, 1 = exact position match)
- physics_accuracy: Does the overall physical state (positions, deformations, interactions) \
match the GT? (0 = completely wrong, 1 = physically accurate)

Respond with ONLY JSON:
{{
  "gen_annotations_removed": <float>,
  "object_consistency": <float>,
  "visual_quality": <float>,
  "object_position_match": <float>,
  "physics_accuracy": <float>,
  "feedback": "<1-2 sentences on the biggest issues>"
}}"""

DEDUCTION_KEYFRAME_FIELDS = [
    "gen_annotations_removed",
    "object_consistency",
    "visual_quality",
    "object_position_match",
    "physics_accuracy",
]
_KEYFRAME_PATTERN = re.compile(r"frame_(\d+)_t([\d.]+)s\.(?:png|jpg|jpeg)$")


def _find_keyframes(keyframe_dir: str | Path) -> list[dict]:
    entries = []
    for path in sorted(Path(keyframe_dir).iterdir()):
        match = _KEYFRAME_PATTERN.match(path.name)
        if match:
            entries.append(
                {
                    "path": path,
                    "time": float(match.group(2)),
                    "index": int(match.group(1)),
                }
            )
    return sorted(entries, key=lambda item: item["time"])


def _get_gt_frame(case: CaseData, time_sec: float) -> Optional[np.ndarray]:
    frame_index = round(time_sec * case.gt_fps)
    if case.gt_video_path:
        cap = cv2.VideoCapture(str(case.gt_video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_index >= total:
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None
    path = case.gt_frames_dir / f"{frame_index:04d}.png"
    return cv2.imread(str(path), cv2.IMREAD_COLOR) if path.exists() else None


def _evaluate_keyframe_gemini(
    evaluator: VideoEvaluator,
    first_frame_data_url: str,
    generated_path: Path,
    gt_frame: np.ndarray,
    time_point: float,
) -> dict:
    generated = cv2.imread(str(generated_path), cv2.IMREAD_COLOR)
    if generated is None:
        raise FileNotFoundError(f"Cannot read generated frame: {generated_path}")
    content = [
        {
            "type": "text",
            "text": EVAL_PROMPT_DEDUCTION_KEYFRAME.format(time_point=time_point),
        },
        {"type": "text", "text": "INITIAL ANNOTATED PHYSICS SCENE:"},
        {"type": "image_url", "image_url": {"url": first_frame_data_url}},
        {"type": "text", "text": f"GROUND TRUTH FRAME at t = {time_point:.2f}s:"},
        {
            "type": "image_url",
            "image_url": {"url": bytes_to_data_url(frame_to_jpeg_bytes(gt_frame))},
        },
        {"type": "text", "text": f"GENERATED FRAME at t = {time_point:.2f}s:"},
        {
            "type": "image_url",
            "image_url": {"url": bytes_to_data_url(frame_to_jpeg_bytes(generated))},
        },
    ]
    raw = evaluator.client.chat_completion([{"role": "user", "content": content}])
    data = parse_llm_json(raw)
    if data is None:
        logger.error("Failed to parse Gemini response: %s", raw[:300])
        return {field: 0.0 for field in DEDUCTION_KEYFRAME_FIELDS}
    result = {}
    for field in DEDUCTION_KEYFRAME_FIELDS + ["feedback"]:
        value = data.get(field, 0.0)
        if field == "feedback":
            result[field] = str(value) if value else ""
        else:
            try:
                number = float(value)
                result[field] = max(0.0, min(1.0, number)) if number >= 0 else -1.0
            except (TypeError, ValueError):
                result[field] = 0.0
    return result


def evaluate_deduction_keyframes(
    keyframe_dir: str | Path,
    case: CaseData,
    evaluator: VideoEvaluator,
) -> dict:
    """Evaluate image-model Deduction exactly as the paper implementation."""
    keyframes = _find_keyframes(keyframe_dir)
    if not keyframes:
        raise FileNotFoundError(f"No Deduction keyframes found in {keyframe_dir}")

    first_frame_data_url = image_to_data_url(case.first_frame_path)
    per_frame = []
    all_scores = {field: [] for field in DEDUCTION_KEYFRAME_FIELDS}
    for keyframe in keyframes:
        result = {
            "time": keyframe["time"],
            "index": keyframe["index"],
            "path": str(keyframe["path"]),
            "gt_frame_idx": round(keyframe["time"] * case.gt_fps),
        }
        gt_frame = _get_gt_frame(case, keyframe["time"])
        if gt_frame is not None:
            try:
                gemini_scores = _evaluate_keyframe_gemini(
                    evaluator,
                    first_frame_data_url,
                    keyframe["path"],
                    gt_frame,
                    keyframe["time"],
                )
                result["gemini_scores"] = gemini_scores
                for field in DEDUCTION_KEYFRAME_FIELDS:
                    value = gemini_scores.get(field, 0.0)
                    if value >= 0:
                        all_scores[field].append(value)
            except Exception as exc:
                logger.warning("Gemini keyframe evaluation failed: %s", exc)
                result["gemini_error"] = str(exc)
        per_frame.append(result)

    gemini_avg = {
        field: float(np.mean(values)) if values else 0.0
        for field, values in all_scores.items()
    }

    from .programmatic_metrics import (
        Evaluator as ProgrammaticEvaluator,
        MaskedPSNRMetric,
        PSNRMetric,
        SpatialIoUMetric,
        SpatiotemporalIoUMetric,
        WeightedSpatialIoUMetric,
    )

    programmatic_evaluator = ProgrammaticEvaluator(
        [PSNRMetric(), MaskedPSNRMetric()],
        [SpatialIoUMetric(), SpatiotemporalIoUMetric(), WeightedSpatialIoUMetric()],
        velocity_metric=None,
    )
    try:
        programmatic = programmatic_evaluator.evaluate_keyframes(
            keyframe_dir=str(keyframe_dir),
            case_dir=str(case.case_dir),
            gt_fps=case.gt_fps,
            gt_video_path=case.gt_video_path,
            physics_duration=case.physics_duration,
            is_realworld=case.is_realworld,
        )
    except Exception as exc:
        logger.warning("Programmatic keyframe metrics failed: %s", exc)
        programmatic = None

    scores = SubtrackScores()
    scores.gen_annotations_removed = gemini_avg.get("gen_annotations_removed", 0.0)
    scores.object_consistency = gemini_avg.get("object_consistency", 0.0)
    scores.visual_quality = gemini_avg.get("visual_quality", 0.0)
    scores.motion_smoothness = gemini_avg.get("object_position_match", 0.0)
    scores.physics_accuracy = gemini_avg.get("physics_accuracy", 0.0)
    if programmatic:
        scores.gen_psnr = programmatic.get("psnr", -1.0)
        scores.gen_masked_psnr = programmatic.get("masked_psnr", -1.0)
        scores.gen_spatial_iou = programmatic.get("spatial_iou", -1.0)
        scores.gen_spatiotemporal_iou = programmatic.get("spatiotemporal_iou", -1.0)
        scores.gen_weighted_spatial_iou = programmatic.get("weighted_spatial_iou", -1.0)
        scores.gen_velocity_error = programmatic.get("velocity_error", -1.0)
    feedbacks = [
        item.get("gemini_scores", {}).get("feedback", "") for item in per_frame
    ]
    scores.feedback = " | ".join(value for value in feedbacks if value)
    return {
        "per_frame": per_frame,
        "gemini_avg": gemini_avg,
        "programmatic": programmatic,
        "subtrack_scores": scores.to_dict()["deduction"],
        "deduction_score": round(scores.deduction_score(), 3),
    }
