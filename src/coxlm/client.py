"""The remote model: ``coxlm.connect(url)`` talks to a coxlm server's POST /v1/decide.

Standard library only (urllib), so a laptop without a GPU or torch can use it.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Iterable, Iterator, overload

from .decide import Answers, Q, answers_class, as_schema, iter_batches
from .schema import Schema
from .wire import answer_from_json, decide_request


class CoxlmError(RuntimeError):
    """The server rejected a request or could not be reached."""


class RemoteModel:
    """Same contract as a locally loaded model: ``decide(state, questions)`` and ``decide_batch(states, questions)``."""

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

    @overload
    def decide(self, state: Any, questions: type[Q]) -> Q: ...
    @overload
    def decide(self, state: Any, questions: Schema) -> Answers: ...

    def decide(self, state, questions):
        """One state (str, dict or list) -> its answers; a Questions subclass comes back as an instance of itself."""
        return self._decide([state], questions)[0]

    @overload
    def decide_batch(self, states: Iterable[Any], questions: type[Q]) -> list[Q]: ...
    @overload
    def decide_batch(self, states: Iterable[Any], questions: Schema) -> list[Answers]: ...

    def decide_batch(self, states, questions):
        """Several states in one request -> one answers object per state, in order."""
        if isinstance(states, (str, bytes, dict)):
            raise TypeError("decide_batch() takes a list of states; for one state use decide(state, questions)")
        return self._decide(list(states), questions)  # any iterable; consumed whole

    @overload
    def decide_iter(self, states: Iterable[Any], questions: type[Q], batch_size: int = 32) -> Iterator[Q]: ...
    @overload
    def decide_iter(self, states: Iterable[Any], questions: Schema, batch_size: int = 32) -> Iterator[Answers]: ...

    def decide_iter(self, states, questions, batch_size=32):
        """Any iterable of states (a generator, a file, a cursor) -> answers one state at a time, in order, sent in
        batches of ``batch_size`` behind the scenes. Nothing is read ahead beyond the current batch."""
        if isinstance(states, (str, bytes, dict)):
            raise TypeError("decide_iter() takes an iterable of states; for one state use decide(state, questions)")
        for batch in iter_batches(states, batch_size):
            yield from self._decide(batch, questions)

    def _decide(self, states: list, questions) -> list:
        schema = as_schema(questions)
        cls = answers_class(questions)
        out = self._request("POST", "/v1/decide", decide_request(states, schema))
        if "error" in out:
            err = out["error"]
            raise CoxlmError(err.get("message", str(err)) if isinstance(err, dict) else str(err))
        rows = out.get("answers") or []
        if len(rows) != len(states):
            raise CoxlmError(f"server returned {len(rows)} answers for {len(states)} states")
        return [cls({name: answer_from_json(a) for name, a in row.items()}) for row in rows]


def connect(url: str, timeout: float = 120.0, headers: dict[str, str] | None = None) -> RemoteModel:
    """A model served by ``coxlm-serve`` at ``url`` (e.g. "http://host:8000")."""
    return RemoteModel(url, timeout=timeout, headers=headers)
