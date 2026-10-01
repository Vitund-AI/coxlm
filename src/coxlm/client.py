"""The remote model: ``coxlm.connect(url)`` talks to a coxlm server's POST /v1/decide.

Standard library only (urllib), so a laptop without a GPU or torch can use it.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from .decide import Answer
from .schema import Schema
from .wire import answer_from_json, decide_request


class CoxlmError(RuntimeError):
    """The server rejected a request or could not be reached."""


class RemoteModel:
    """Same contract as a locally loaded model: ``decide(states, schema) -> list[dict[name, Answer]]``."""

    def __init__(self, url: str, timeout: float = 120.0, headers: dict[str, str] | None = None) -> None:
        url = url.rstrip("/")
        if "://" not in url:
            url = "http://" + url
        self.url = url
        self.timeout = timeout
        self.headers = dict(headers or {})

    def __repr__(self) -> str:
        return f"RemoteModel({self.url!r})"

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json", **self.headers})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            try:
                err = json.loads(detail).get("error")
                detail = err.get("message", detail) if isinstance(err, dict) else (err or detail)
            except (ValueError, AttributeError):
                pass
            raise CoxlmError(f"{method} {path}: HTTP {e.code}: {detail}") from None
        except urllib.error.URLError as e:
            raise CoxlmError(f"cannot reach coxlm server at {self.url}: {e.reason}") from None
        return json.loads(payload)

    def health(self) -> dict:
        return self._request("GET", "/health")

    def decide(self, states: list, schema: Schema) -> list[dict[str, Answer]]:
        """States (str, dict or list each) -> typed answers per field, one dict per state."""
        if isinstance(states, (str, bytes)):
            raise TypeError("decide() takes a list of states; wrap a single state as [state]")
        if not isinstance(schema, Schema):
            raise TypeError("schema must be a coxlm Schema, e.g. from coxlm.questions(...)")
        out = self._request("POST", "/v1/decide", decide_request(list(states), schema))
        if "error" in out:
            err = out["error"]
            raise CoxlmError(err.get("message", str(err)) if isinstance(err, dict) else str(err))
        rows = out.get("answers") or []
        if len(rows) != len(states):
            raise CoxlmError(f"server returned {len(rows)} answers for {len(states)} states")
        return [{name: answer_from_json(a) for name, a in row.items()} for row in rows]


def connect(url: str, timeout: float = 120.0, headers: dict[str, str] | None = None) -> RemoteModel:
    """A model served by ``coxlm-serve`` at ``url`` (e.g. "http://host:8000")."""
    return RemoteModel(url, timeout=timeout, headers=headers)
