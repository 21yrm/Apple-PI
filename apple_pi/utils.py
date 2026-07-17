"""Media and JSON helpers retained from the paper evaluation code."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_llm_json(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def image_to_data_url(path: str | Path) -> str:
    path = Path(path)
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def bytes_to_data_url(data: bytes, mime_type: str = "image/jpeg") -> str:
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def frame_to_jpeg_bytes(frame: np.ndarray, quality: int = 85) -> bytes:
    ok, buffer = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not ok:
        raise ValueError("Failed to encode frame as JPEG")
    return buffer.tobytes()


def extract_last_frame(path: str | Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Could not read frames from {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not decode final frame from {path}")
    return frame
