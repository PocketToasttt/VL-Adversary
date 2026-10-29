"""OpenAI-compatible client used by VLM and LLM attack stages."""

from typing import Any, Dict, List, Optional

import requests
import json
from pathlib import Path
import subprocess
import sys
import time


_HTTP_WORKER_COMMAND = [sys.executable, str(Path(__file__).with_name("http_completion_worker.py"))]


def _bounded_completion_request(request, timeout):
    """Enforce a wall-clock deadline including DNS, with live progress logs."""
    started = time.monotonic()
    process = subprocess.Popen(_HTTP_WORKER_COMMAND, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pending_input = json.dumps(request)
    try:
        while True:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise requests.Timeout(f"Model request exceeded {timeout}s total deadline (including DNS)")
            try:
                stdout, _ = process.communicate(input=pending_input, timeout=min(15, remaining))
                break
            except subprocess.TimeoutExpired:
                pending_input = None
                print(f"Model request waiting: model={request['payload']['model']} "
                      f"elapsed={time.monotonic() - started:.0f}s/{timeout}s", flush=True)
        if process.returncode != 0:
            raise RuntimeError(f"Model HTTP worker exited with code {process.returncode}")
        result = json.loads(stdout)
        if not result.get("ok"):
            if result.get("retryable"):
                raise requests.ConnectionError(result["error"])
            raise RuntimeError(result["error"])
        return result["content"]
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()

from .config import LLMServiceConfig


class LLMClient:
    def __init__(self, config: LLMServiceConfig):
        self.config = config

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def create_completion(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.3,
        timeout: float = 120,
    ) -> str:
        if not self.config.enabled or not self.config.api_key:
            raise RuntimeError("LLM service is disabled or missing an API key")

        request = {
            "url": f"{self.config.api_base}/chat/completions",
            "headers": {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            "payload": {
                "model": model or self.config.model,
                "messages": messages,
                "temperature": temperature,
            },
            "timeout": timeout,
        }
        for attempt in range(2):
            started = time.monotonic()
            print(f"Model HTTP start: model={request['payload']['model']} attempt={attempt + 1}/2", flush=True)
            try:
                result = _bounded_completion_request(request, timeout)
                print(f"Model HTTP success: elapsed={time.monotonic() - started:.1f}s", flush=True)
                return result
            except (requests.Timeout, requests.ConnectionError) as error:
                if attempt:
                    raise
                print(f"Model HTTP transient failure: {error}; retry once in 2s", flush=True)
                time.sleep(2)

    def create_code(self, prompt: str, *, timeout: float = 120) -> str:
        generated_code = self.create_completion(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            timeout=timeout,
        )
        return self.strip_code_fence(generated_code)

    @staticmethod
    def strip_code_fence(code: str) -> str:
        stripped = code.strip()
        if stripped.startswith("```"):
            first_newline = stripped.find("\n")
            if first_newline >= 0:
                stripped = stripped[first_newline + 1:].strip()
            else:
                stripped = stripped[len("```"):].strip()
        if stripped.endswith("```"):
            stripped = stripped[:-len("```")].strip()
        return stripped

