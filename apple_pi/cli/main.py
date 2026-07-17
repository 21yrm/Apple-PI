"""Command-line interface for the Apple-PI public release."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from apple_pi.constants import (
    DEFAULT_DATASET_REPO,
    DEFAULT_GEMINI_MODEL,
    MOGE_MODEL_REPO,
    MOGE_MODEL_REVISION,
    SAM3_MODEL_REPO,
    SAM3_MODEL_REVISION,
)
from apple_pi.data import (
    CaseData,
    load_dataset_index,
    validate_ground_truth,
    validate_predictions,
)
from apple_pi.evaluation.gemini import GeminiJudge
from apple_pi.cli.doctor import run_doctor
from apple_pi.evaluation.submission import evaluate_submission
from apple_pi.prompts import export_prompt_records
from apple_pi.types import InferenceProtocol, Subtrack


def _fail_on_errors(errors: list[str]) -> None:
    if not errors:
        print("Validation passed.")
        return
    print(f"Validation failed with {len(errors)} error(s):")
    for error in errors:
        print(f"  - {error}")
    raise SystemExit(1)


def doctor_command(args: argparse.Namespace) -> None:
    raise SystemExit(run_doctor())


def download_data(args: argparse.Namespace) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface-hub to download Apple-PI-GT") from exc
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=args.output,
    )


def download_models(_args: argparse.Namespace) -> None:
    """Preflight the exact gated/public checkpoints used by evaluation."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Install huggingface-hub to download model checkpoints"
        ) from exc

    print(f"Downloading {SAM3_MODEL_REPO}@{SAM3_MODEL_REVISION} ...")
    sam3_path = snapshot_download(
        repo_id=SAM3_MODEL_REPO,
        revision=SAM3_MODEL_REVISION,
        allow_patterns=["config.json", "model.safetensors", "processor_config.json"],
    )
    print(f"SAM3 ready: {sam3_path}")

    print(f"Downloading {MOGE_MODEL_REPO}@{MOGE_MODEL_REVISION} ...")
    moge_path = hf_hub_download(
        repo_id=MOGE_MODEL_REPO,
        filename="model.pt",
        revision=MOGE_MODEL_REVISION,
    )
    print(f"MoGe-2 ready: {moge_path}")


def export_prompts(args: argparse.Namespace) -> None:
    dataset = load_dataset_index(args.gt_dir)
    cases = [
        CaseData.load(entry.case_id, dataset.root / entry.path).prompt_record()
        for entry in dataset.cases
    ]
    records = export_prompt_records(cases, InferenceProtocol(args.protocol))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Exported {len(records)} prompt records to {output}")


def evaluate(args: argparse.Namespace) -> None:
    subtracks = tuple(Subtrack) if args.subtrack == "all" else (Subtrack(args.subtrack),)
    _fail_on_errors(validate_ground_truth(args.gt_dir))
    _fail_on_errors(validate_predictions(args.gt_dir, args.pred_dir, subtracks=subtracks))
    judge = GeminiJudge(model=args.gemini_model, cache_dir=args.cache_dir)
    result = evaluate_submission(
        args.gt_dir,
        args.pred_dir,
        judge,
        output_path=args.output,
        subtracks=subtracks,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apple-pi",
        description="Apple-PI inference prompts and paper-compatible evaluation",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor")
    doctor.set_defaults(func=doctor_command)

    download = subparsers.add_parser("download-data")
    download.add_argument("--repo", default=DEFAULT_DATASET_REPO)
    download.add_argument("--output", default="data/apple_pi")
    download.set_defaults(func=download_data)

    models = subparsers.add_parser(
        "download-models",
        help="download the pinned SAM3 and MoGe-2 checkpoints",
    )
    models.set_defaults(func=download_models)

    validate_gt = subparsers.add_parser("validate-gt")
    validate_gt.add_argument("--gt-dir", required=True)
    validate_gt.set_defaults(
        func=lambda args: _fail_on_errors(validate_ground_truth(args.gt_dir))
    )

    validate_pred = subparsers.add_parser("validate-predictions")
    validate_pred.add_argument("--gt-dir", required=True)
    validate_pred.add_argument("--pred-dir", required=True)
    validate_pred.set_defaults(
        func=lambda args: _fail_on_errors(
            validate_predictions(args.gt_dir, args.pred_dir)
        )
    )

    prompts = subparsers.add_parser("export-prompts")
    prompts.add_argument("--gt-dir", required=True)
    prompts.add_argument(
        "--protocol",
        choices=[protocol.value for protocol in InferenceProtocol],
        required=True,
    )
    prompts.add_argument("--output", required=True)
    prompts.set_defaults(func=export_prompts)

    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--gt-dir", required=True)
    evaluation.add_argument("--pred-dir", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument(
        "--gemini-model",
        default=os.getenv("APPLE_PI_GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
    )
    evaluation.add_argument("--cache-dir", default=".cache/apple_pi/gemini")
    evaluation.add_argument(
        "--subtrack",
        default="all",
        choices=["all", *(subtrack.value for subtrack in Subtrack)],
    )
    evaluation.set_defaults(func=evaluate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
