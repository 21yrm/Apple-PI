"""Environment diagnostics for the full paper evaluation stack."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import shutil
import sys

from apple_pi.constants import (
    MOGE_MODEL_REPO,
    MOGE_MODEL_REVISION,
    SAM3_MODEL_REPO,
    SAM3_MODEL_REVISION,
)


def run_doctor() -> int:
    checks = [
        (
            "Python 3.10",
            sys.version_info[:2] == (3, 10),
            sys.version.split()[0],
        ),
        ("git executable", bool(shutil.which("git")), shutil.which("git") or "missing"),
        (
            "ffmpeg executable",
            bool(shutil.which("ffmpeg")),
            shutil.which("ffmpeg") or "missing",
        ),
    ]

    try:
        import torch

        checks.append(("PyTorch", True, torch.__version__))
        cuda_available = torch.cuda.is_available()
        cuda_detail = (
            torch.cuda.get_device_name(0) if cuda_available else str(cuda_available)
        )
        checks.append(("CUDA available", cuda_available, cuda_detail))
    except Exception as exc:
        checks.append(("PyTorch", False, str(exc)))

    try:
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor  # noqa: F401

        checks.append(
            (
                "SAM3 Transformers classes",
                True,
                f"transformers {version('transformers')}",
            )
        )
    except Exception as exc:
        checks.append(("SAM3 Transformers classes", False, str(exc)))

    try:
        from moge.model.v2 import MoGeModel  # noqa: F401

        checks.append(("MoGe-2", True, f"moge {version('moge')}"))
    except Exception as exc:
        checks.append(("MoGe-2", False, str(exc)))

    checks.append(
        (
            "GEMINI_API_KEY",
            bool(os.getenv("GEMINI_API_KEY")),
            "set" if os.getenv("GEMINI_API_KEY") else "missing",
        )
    )

    try:
        from huggingface_hub import get_token, try_to_load_from_cache

        token = get_token()
        checks.append(
            ("Hugging Face token", bool(token), "set" if token else "missing")
        )
        sam3_cached = try_to_load_from_cache(
            SAM3_MODEL_REPO,
            "model.safetensors",
            revision=SAM3_MODEL_REVISION,
        )
        checks.append(
            (
                "Pinned SAM3 checkpoint",
                isinstance(sam3_cached, str) and Path(sam3_cached).is_file(),
                "cached" if isinstance(sam3_cached, str) else "run download-models",
            )
        )
        moge_cached = try_to_load_from_cache(
            MOGE_MODEL_REPO,
            "model.pt",
            revision=MOGE_MODEL_REVISION,
        )
        checks.append(
            (
                "Pinned MoGe-2 checkpoint",
                isinstance(moge_cached, str) and Path(moge_cached).is_file(),
                "cached" if isinstance(moge_cached, str) else "run download-models",
            )
        )
    except Exception as exc:
        checks.append(("Hugging Face token", False, str(exc)))

    try:
        checks.append(("google-genai", True, version("google-genai")))
    except PackageNotFoundError as exc:
        checks.append(("google-genai", False, str(exc)))

    print("Apple-PI environment check")
    print("=" * 32)
    for name, ok, detail in checks:
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 1
