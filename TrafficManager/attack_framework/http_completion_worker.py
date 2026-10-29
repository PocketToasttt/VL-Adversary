"""One HTTP request per process; credentials are passed only through stdin."""
import json
import sys

import requests


def perform_request(request):
    try:
        response = requests.post(request["url"], headers=request["headers"],
                                 json=request["payload"], timeout=request["timeout"])
        response.raise_for_status()
        return {"ok": True, "content": response.json()["choices"][0]["message"]["content"]}
    except requests.RequestException as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        return {"ok": False,
                "retryable": isinstance(error, (requests.Timeout, requests.ConnectionError))
                or status in (429, 502, 503, 504),
                "error": type(error).__name__ + (f" HTTP {status}" if status else "")}
    except (ValueError, KeyError, IndexError, TypeError) as error:
        return {"ok": False, "retryable": False,
                "error": "Invalid API response: " + type(error).__name__}


if __name__ == "__main__":
    print(json.dumps(perform_request(json.load(sys.stdin)), ensure_ascii=True), flush=True)
