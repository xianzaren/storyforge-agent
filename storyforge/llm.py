from __future__ import annotations

import json
import os
import urllib.request
import base64
import mimetypes
from pathlib import Path
from typing import Any


class CompatibleLLMClient:
    """Small OpenAI-compatible client using only the Python standard library."""

    def __init__(self) -> None:
        self.api_key = os.getenv("STORYFORGE_API_KEY", "")
        self.base_url = os.getenv("STORYFORGE_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("STORYFORGE_MODEL", "")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.model)

    def generate_json(self, system: str, user: str, timeout: int = 90) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("LLM client is not configured")
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.4,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)

    @property
    def vision_enabled(self) -> bool:
        return bool(self.api_key and os.getenv("STORYFORGE_VISION_MODEL"))

    def generate_vision_json(
        self,
        image_path: Path,
        system: str,
        user: str,
        timeout: int = 90,
    ) -> dict[str, Any]:
        model = os.getenv("STORYFORGE_VISION_MODEL", "")
        if not self.api_key or not model:
            raise RuntimeError("Vision model is not configured")
        mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    ],
                },
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return json.loads(data["choices"][0]["message"]["content"])
