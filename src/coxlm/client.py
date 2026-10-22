"""The remote model: ``coxlm.connect(url)`` talks to a coxlm server's POST /v1/decide.

Standard library only (http.client), so a laptop without a GPU or torch can use it. Each thread keeps one HTTP/1.1
connection open to the server, so only the first request pays the TCP (and TLS) handshake.
"""
from __future__ import annotations

import http.client
import json
import threading
import urllib.parse
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
        self.closed = False
        u = urllib.parse.urlsplit(url)
        self._https = u.scheme == "https"
        self._host, self._port, self._base = u.hostname, u.port, u.path.rstrip("/")
        self._local = threading.local()  # one persistent connection per thread (http.client is not thread-safe)
        self._conns: list = []
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"RemoteModel({self.url!r})"

    def close(self) -> None:
        """Stop using the server: close the open connections; later calls raise CoxlmError."""
        self.closed = True
        with self._lock:
            for c in self._conns:
                c.close()
            self._conns.clear()

    def __enter__(self) -> "RemoteModel":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _conn(self, fresh: bool = False):
        c = getattr(self._local, "conn", None)
        if c is None or fresh:
            if c is not None:
                c.close()
            cls = http.client.HTTPSConnection if self._https else http.client.HTTPConnection
            c = cls(self._host, self._port, timeout=self.timeout)
            self._local.conn = c
            with self._lock:
                self._conns.append(c)
        return c

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        if self.closed:
            raise CoxlmError(f"this model was closed; connect to {self.url} again to use it")
        data = None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json", **self.headers}
        for attempt in (0, 1):  # a kept-alive connection the server has since closed is retried once on a new one
            conn = self._conn(fresh=attempt == 1)
            try:
                conn.request(method, self._base + path, body=data, headers=headers)
                resp = conn.getresponse()
                payload = resp.read()
                break
            except (OSError, http.client.HTTPException) as e:
                # on the reused connection any socket error may only mean the server closed it while idle: retry once
                # on a new connection; an error on the new one is real
                conn.close()
                if attempt == 1:
                    self._local.conn = None
                    raise CoxlmError(f"cannot reach coxlm server at {self.url}: {e}") from None
        if resp.status >= 400:
            detail = payload.decode(errors="replace")
            try:
                err = json.loads(detail).get("error")
                detail = err.get("message", detail) if isinstance(err, dict) else (err or detail)
            except (ValueError, AttributeError):
                pass
            raise CoxlmError(f"{method} {path}: HTTP {resp.status}: {detail}")
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
