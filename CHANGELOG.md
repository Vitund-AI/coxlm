# Changelog

## 0.2.1 (2026-10-06)

- Models are context managers: `with coxlm.load(path) as model:` releases the weights and the GPU memory at the end
  of the block, and `with coxlm.connect(url) as model:` closes the client. `close()` does the same directly; a closed
  model raises an error if used again.

## 0.2.0 (2026-10-06)

API changes (breaking):

- `decide(state, questions)` takes **one** state and returns its answers. A list passed to `decide` is one state
  (rendered as numbered lines), as it always was inside a batch. For several states use `decide_batch(states,
  questions)`, which returns a list in order from one request; it accepts any iterable.
- Answers are an `Answers` mapping instead of a plain dict: read them by attribute (`ans.team`) or by name
  (`ans["team"]`), and iterate them like a dict. They are read-only.

New:

- Class-based questions: subclass `Questions` with one `choice` / `score` / `yesno` attribute per question. The answers
  come back as an instance of the class, so editors complete the question names and type checkers (mypy, pyright)
  report misspelt ones. Subclasses inherit their parents' questions.
- `decide_iter(states, questions, batch_size=32)` streams any iterable of states through the model in batches and
  yields answers in order, reading no further ahead than the current batch.
- `Answer.pick(min_confidence, default=None)`: the top option when the model is at least that sure, else `default`,
  for `match` statements and other branching. `Answer.ranked()`: options with probabilities, most likely first.
  Answers also work with structural patterns (`case Answer(choice="billing", confidence=c) if c > 0.9`).
- `questions()` warns when a question name is one of the answers' mapping methods (`keys`, `items`, `values`, `get`);
  such a question is read as `ans["items"]`. A `Questions` subclass refuses those names.

Also in this release:

- The backbone runs in bf16 by default, whatever precision the checkpoint was saved in (`--dtype` / `load(dtype=)` override).
- The model module holds only what released checkpoints use: the packed question layout with the slot or pointer readout (with option isolation). Research-only settings are gone, and a checkpoint that records one is refused with the setting named, instead of being read wrongly. Answers from supported checkpoints are unchanged (verified identical).
- Pointer-readout checkpoints on hybrid backbones (Qwen3.5) raise a clear error: they need cache forks, not yet in coxlm.

## 0.1.0 (2026-09-30)

First release: the inference package extracted from the cox research code.

- `coxlm.connect(url)`: a standard-library client for a coxlm server (no torch needed).
- `coxlm.load(path, encoder=...)`: local inference on adapter (LoRA) or full-weight checkpoints, read with the settings recorded in the checkpoint (`pip install "coxlm[local]"`).
- Schema API: `questions`, `choice`, `score`, `yesno`, `multilabel`, `multiscore`; typed `Answer` with calibrated confidence and the full distribution.
- `coxlm-serve`: HTTP server with the native `POST /v1/decide`, a System One-compatible `POST /v1/systemone`, a demo page, `/health` and `/v1/models`.
