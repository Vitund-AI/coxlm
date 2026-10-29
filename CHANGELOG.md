# Changelog

## 0.7.0 (not released yet)

- `load()` reads INT8 builds (cox_config.json `"format_version": 2`, `"weights": {"format": "int8wo"}`, e.g.
  `Vitund/cox-1-3b-int8`): int8 weights with one bf16 scale per output channel for the backbone's linear layers and
  per row for the embedding table. On CUDA they stay int8 in GPU memory (cox-1-3b: about 2.9 GB loaded instead of
  5.9 GB) as torchao `Int8Tensor` weights when torchao is installed (`coxlm[int8]`), or coxlm's own int8 layer with
  the same products otherwise. On the CPU, Apple MPS and MLX they are converted to bf16 at load. `COXLM_INT8`
  overrides where they stay int8. Adapters and fine-tuning refuse INT8 builds (use the bf16 build).
- Older coxlm versions refuse INT8 builds with "export format 2 is newer than this coxlm; upgrade coxlm".

## 0.6.3

- System One endpoint: option and level descriptions that contain line breaks (verse, addresses) or are JSON objects
  are kept as one option each. They were split into extra options, so a two-option question could come back with
  seven answers. Answers are keyed by the caller's own option names.
- System One responses report `usage.input_tokens` as an integer (the state's tokens), never `null`.
- `coxlm-serve --reject-overlong`: answer HTTP 422 ("longer than the maximum model length") for a state over
  `--max-length` instead of truncating it.
- Tests run on GitHub Actions (core without torch on Python 3.10-3.13, CPU PyTorch, MLX CPU build).

## 0.6.2

- `load()` reads packed exports (`"weights": {"format": "codebook4"}` in cox_config.json): 4-bit codebook weights,
  about half the download, unpacked to bfloat16 as the model loads, so answers and speed match the unpacked export.
  PyTorch and MLX backends.

## Unreleased

- `coxlm.finetune(model, examples, questions, out_dir)`: adapts a model export to your labelled examples with a
  small LoRA adapter (hard labels or probability targets; `(text, answers)` pairs or JSONL rows). It sets a
  stratified held-out share aside, fits one temperature on it, and prints and saves (`report.json`) a before/after
  report: accuracy, skill over the most common label, ECE, and the share of decisions automatable at 95% accuracy.
  It also checks a small built-in set of general questions and warns if they got worse.
- `coxlm.load(export, adapter=...)` applies such an adapter (a folder or a Hub repo id; PEFT-named LoRA tensors),
  with its temperature; it refuses an adapter trained on another base. `merge=True` folds it into the weights.
  PyTorch backend only for now.
- Qwen3.5 models on the CPU or Apple MPS no longer fail when flash-linear-attention is installed: its kernels take
  CUDA tensors only, so those calls now use the PyTorch reference.
- `examples/doom`: fine-tune the model on Doom decisions and watch it play ViZDoom.

## 0.4.2 (unreleased)

- `load()` picks the Apple GPU (MPS) on Apple silicon when there is no CUDA GPU, and uses float32 on an Apple-silicon
  CPU, where PyTorch has no fast bfloat16 maths (2.3x faster per answer there). Same answers as on CUDA.
- README: what to expect on Macs (Apple silicon via MPS, Intel Macs through a server).

## 0.4.1 (unreleased)

- No more false warning that yes/no questions without descriptions are unreadable: cox-1 exports declare that
  the model reads a question's wording.
- A one-time notice when the model runs on the CPU, with the reason (no CUDA build of PyTorch, or no GPU).
- The `fast` extra also installs flash-linear-attention (it only installed `kernels`, so the linear-attention
  layers fell back to slow PyTorch code unless flash-linear-attention was installed separately).
- README: requirements and speed (GPU, CPU, Windows); `examples/demo.py` with load and per-answer timing.

## 0.4.0 (unreleased)

- Loads cox-1 model exports (the format of `Vitund/cox-1-3b`): a Hugging Face text-model folder plus
  `cox_head.safetensors` and `cox_config.json`, from a folder or a Hub repo id.
- Faster: a short yes/no request takes about 13 ms instead of 27 ms on an RTX 4090 (short requests now run in one
  pass), a question with 77 options about 135 ms instead of 400 ms (the question is read once and each option forks
  from it; option rows are grouped by length), and a 6k-token text about 350 ms instead of 630 ms (see the next
  item). Answers are unchanged within bf16 rounding.
- Offline use (`HF_HUB_OFFLINE=1`) now keeps the fast linear-attention kernels; before, both silently fell back to
  the slow PyTorch path.
- `COXLM_COMPILE=1` (opt-in) compiles each backbone block: about 19% faster on long texts.
- `coxlm-serve` starts with model exports, and no longer frees and re-allocates GPU memory after every request.
- Yes/no questions take `noul` and `true`/`false` as well as `yesno` and `yes`/`no`. A System One `noul`
  question's `true` and `false` criteria now reach the model as the yes/no descriptions, as `{"no", "yes"}`
  descriptions do on the native API and in `yesno(...)`; before, they were dropped.

## 0.3.0 (2026-10-09)

- Pointer-readout models on hybrid backbones (Qwen3.5) now run. The states are encoded once with the cache on, and
  the cache is forked per branch: each option line continues from the state and its question head, the answer from
  the head alone, and a yes/no question scored by the slot readout from the state with its whole block, as the
  research models were trained. Branch rows run in chunks of at most `COXLM_FORK_TOKENS` tokens (default 16384), so
  memory stays bounded for questions with many options. Checked against the research implementation on eight suite
  tasks (77- and 100-option choices, yes/no, scores, several fields per text): identical answers, probabilities
  within 0.0003 in fp32 (within 0.03 in bf16, kernel differences).
- Load release folders and Hugging Face repos: `coxlm.load` and `coxlm-serve` accept a `model.pt`, a folder with
  `model.safetensors` + `config.json`, or an `owner/name` repo id (downloaded and cached; `revision=` pins a version).
  Full-weight checkpoints build the backbone from its configuration only, without downloading the base weights.

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
