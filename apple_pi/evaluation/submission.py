"""Evaluate and aggregate the required three rollouts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from apple_pi.constants import (
    BENCHMARK_NAME,
    BENCHMARK_VERSION,
    NUM_ROLLOUTS,
    PROMPT_VERSION,
)
from apple_pi.data.case import CaseData
from apple_pi.data.dataset import load_dataset_index
from apple_pi.data.predictions import prediction_path
from apple_pi.types import InferenceProtocol, Subtrack

from .paper_evaluator import EvaluationConfig, VideoEvaluator
from .runners import (
    evaluate_deduction_keyframes,
    evaluate_image_prediction,
    evaluate_video_prediction,
)

logger = logging.getLogger(__name__)


def aggregate_rollouts(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate three paper rollout scores with an arithmetic mean."""
    successful = [record for record in records if "score" in record]
    values = [float(record["score"]) for record in successful]
    complete = len(values) == NUM_ROLLOUTS
    return {
        "num_expected": NUM_ROLLOUTS,
        "num_successful": len(values),
        "mean": round(mean(values), 4) if complete else None,
        "partial_mean": round(mean(values), 4) if values and not complete else None,
        "std": round(pstdev(values), 4)
        if len(values) > 1
        else (0.0 if values else None),
        "min": round(min(values), 4) if values else None,
        "max": round(max(values), 4) if values else None,
    }


def _evaluate_one(
    protocol: InferenceProtocol,
    path: Path,
    case: CaseData,
    subtrack: Subtrack,
    evaluator: VideoEvaluator,
) -> dict[str, Any]:
    if protocol == InferenceProtocol.VIDEO:
        scores = evaluate_video_prediction(path, case, subtrack, evaluator)
        return {
            "score": round(scores.subtrack_score(subtrack), 3),
            "details": scores.to_dict()[subtrack.value],
            "feedback": scores.feedback,
        }
    if subtrack == Subtrack.DEDUCTION:
        result = evaluate_deduction_keyframes(path, case, evaluator)
        return {
            "score": result["deduction_score"],
            "details": result["subtrack_scores"],
            "gemini_avg": result["gemini_avg"],
            "programmatic": result["programmatic"],
            "per_frame": result["per_frame"],
        }
    scores = evaluate_image_prediction(path, case, subtrack, evaluator)
    return {
        "score": round(scores.subtrack_score(subtrack), 3),
        "details": scores.to_dict()[subtrack.value],
        "feedback": scores.feedback,
    }


def evaluate_submission(
    gt_root: str | Path,
    prediction_root: str | Path,
    judge,
    *,
    output_path: str | Path | None = None,
    subtracks: tuple[Subtrack, ...] = tuple(Subtrack),
) -> dict[str, Any]:
    """Run the official evaluation over all cases and three rollouts."""
    gt_root = Path(gt_root).expanduser().resolve()
    prediction_root = Path(prediction_root).expanduser().resolve()
    submission = json.loads(
        (prediction_root / "submission.json").read_text(encoding="utf-8")
    )
    protocol = InferenceProtocol(submission["protocol"])
    dataset = load_dataset_index(gt_root)
    evaluator = VideoEvaluator(judge, EvaluationConfig())

    result: dict[str, Any] = {
        "benchmark": BENCHMARK_NAME,
        "benchmark_version": BENCHMARK_VERSION,
        "dataset_version": dataset.version,
        "prompt_version": PROMPT_VERSION,
        "num_rollouts": NUM_ROLLOUTS,
        "model": submission["model"],
        "protocol": protocol.value,
        "judge": getattr(judge, "model", "custom"),
        "cases": {},
    }

    for entry in dataset.cases:
        case = CaseData.load(entry.case_id, dataset.root / entry.path)
        case_result: dict[str, Any] = {}
        for subtrack in subtracks:
            rollout_results = []
            for rollout in range(NUM_ROLLOUTS):
                path = prediction_path(
                    prediction_root,
                    entry.case_id,
                    protocol,
                    subtrack,
                    rollout,
                )
                logger.info(
                    "Evaluating %s | %s | rollout %d",
                    entry.case_id,
                    subtrack.value,
                    rollout,
                )
                try:
                    rollout_result = _evaluate_one(
                        protocol,
                        path,
                        case,
                        subtrack,
                        evaluator,
                    )
                    rollout_result.update(
                        {"rollout": rollout, "input": str(path), "status": "ok"}
                    )
                except Exception as exc:
                    logger.exception(
                        "Evaluation failed for %s | %s | rollout %d",
                        entry.case_id,
                        subtrack.value,
                        rollout,
                    )
                    rollout_result = {
                        "rollout": rollout,
                        "input": str(path),
                        "status": "failed",
                        "error": str(exc),
                    }
                rollout_results.append(rollout_result)
                if output_path:
                    _write_result(output_path, result)
            case_result[subtrack.value] = {
                "rollouts": rollout_results,
                "aggregate": aggregate_rollouts(rollout_results),
            }
        result["cases"][entry.case_id] = case_result
        if output_path:
            _write_result(output_path, result)

    result["summary"] = _aggregate_dataset(result["cases"], subtracks)
    if output_path:
        _write_result(output_path, result)
    return result


def _aggregate_dataset(
    cases: dict[str, Any],
    subtracks: tuple[Subtrack, ...],
) -> dict[str, Any]:
    summary = {}
    for subtrack in subtracks:
        values = []
        for case in cases.values():
            value = case[subtrack.value]["aggregate"]["mean"]
            if value is not None:
                values.append(float(value))
        summary[subtrack.value] = {
            "num_cases": len(values),
            "mean": round(mean(values), 4) if values else None,
        }
    return summary


def _write_result(path: str | Path, result: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
