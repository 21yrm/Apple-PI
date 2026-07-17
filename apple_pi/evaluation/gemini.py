"""Official Gemini 3.0 Flash judge adapter.

The adapter preserves the paper evaluator's ordered multimodal message
construction while replacing the private proxy endpoint with Google's public
Gemini API.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from dotenv import load_dotenv
import tempfile
import time
from pathlib import Path
from typing import Any

from apple_pi.constants import DEFAULT_GEMINI_MODEL

logger = logging.getLogger(__name__)
load_dotenv()


class GeminiJudge:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_GEMINI_MODEL,
        cache_dir: str | Path = ".cache/apple_pi/gemini",
        max_retries: int = 5,
    ):
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "Gemini evaluation requires google-genai. "
                "Install the release environment from environment.yml."
            ) from exc
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini evaluation")
        self.client = genai.Client(api_key=resolved_key)
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries

    @staticmethod
    def _decode_data_url(url: str) -> tuple[str, bytes]:
        header, encoded = url.split(",", 1)
        mime_type = header[5:].split(";", 1)[0]
        return mime_type, base64.b64decode(encoded)

    def _cache_path(self, messages: list[dict[str, Any]]) -> Path:
        payload = json.dumps(
            {"model": self.model, "messages": messages},
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        return self.cache_dir / f"{hashlib.sha256(payload).hexdigest()}.json"

    def _upload_video(self, video_bytes: bytes, mime_type: str):
        suffix = ".mp4" if mime_type == "video/mp4" else ".video"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(video_bytes)
            path = handle.name
        try:
            uploaded = self.client.files.upload(
                file=path,
                config={"mime_type": mime_type},
            )
            while getattr(getattr(uploaded, "state", None), "name", None) in {
                None,
                "PROCESSING",
            }:
                time.sleep(2)
                uploaded = self.client.files.get(name=uploaded.name)
            if getattr(getattr(uploaded, "state", None), "name", None) == "FAILED":
                raise RuntimeError("Gemini video processing failed")
            return uploaded
        finally:
            Path(path).unlink(missing_ok=True)

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[Any]:
        from google.genai import types

        parts: list[Any] = []
        for message in messages:
            for item in message["content"]:
                if item["type"] == "text":
                    parts.append(item["text"])
                    continue
                mime_type, data = self._decode_data_url(item["image_url"]["url"])
                if mime_type.startswith("video/") and len(data) >= 20 * 1024 * 1024:
                    parts.append(self._upload_video(data, mime_type))
                else:
                    parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))
        return parts

    def chat_completion(self, messages: list[dict[str, Any]]) -> str:
        cache_path = self._cache_path(messages)
        if cache_path.is_file():
            return json.loads(cache_path.read_text(encoding="utf-8"))["text"]

        from google.genai import types

        parts = self._convert_messages(messages)
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=parts,
                    config=config,
                )
                text = response.text
                if not text:
                    raise RuntimeError("Gemini returned an empty response")
                cache_path.write_text(
                    json.dumps(
                        {"model": self.model, "text": text},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                return text
            except Exception as exc:
                last_error = exc
                wait = min(30, 5 * (attempt + 1))
                logger.warning(
                    "Gemini request failed (%s/%s): %s; retrying in %ss",
                    attempt + 1,
                    self.max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"Gemini evaluation failed after {self.max_retries} attempts"
        ) from last_error
