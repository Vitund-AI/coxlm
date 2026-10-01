"""Shared fixtures: a deterministic fake model and a live server around it (CPU only, no torch)."""
from __future__ import annotations

import hashlib
import math
import threading

import pytest

from coxlm.decide import answers_for, render_state


class FakeModel:
    """decide(states, schema) with deterministic, state-dependent distributions; records its calls."""

    max_length = 2048
    tokenizer = None

    def __init__(self):
        self.calls = []

    def decide(self, states, schema):
        self.calls.append((list(states), schema))
        out = []
        for st in states:
            text = render_state(st)
            pred = {}
            for f in schema:
                logits = [int(hashlib.sha1(f"{text}|{f.name}|{o}".encode()).hexdigest()[:6], 16) / 0xFFFFFF * 4
                          for o in f.all_options]
                z = sum(math.exp(x) for x in logits)
                pred[f.name] = [math.exp(x) / z for x in logits]
            out.append(answers_for(schema, pred))
        return out


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def server(fake_model):
    from coxlm import serve

    httpd = serve.serve(fake_model, port=0, host="127.0.0.1", encoder="Example/Fake-1B-Base", checkpoint="fake.pt")
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", fake_model
    httpd.shutdown()
    httpd.server_close()
