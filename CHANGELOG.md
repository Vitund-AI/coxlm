# Changelog

## Unreleased

- The backbone runs in bf16 by default, whatever precision the checkpoint was saved in (`--dtype` / `load(dtype=)` override).
- The model module holds only what released checkpoints use: the packed question layout with the slot or pointer readout (with option isolation). Research-only settings are gone, and a checkpoint that records one is refused with the setting named, instead of being read wrongly. Answers from supported checkpoints are unchanged (verified identical).
- Pointer-readout checkpoints on hybrid backbones (Qwen3.5) raise a clear error: they need cache forks, not yet in coxlm.

## 0.1.0 (2026-09-30)

First release: the inference package extracted from the cox research code.

- `coxlm.connect(url)`: a standard-library client for a coxlm server (no torch needed).
- `coxlm.load(path, encoder=...)`: local inference on adapter (LoRA) or full-weight checkpoints, read with the settings recorded in the checkpoint (`pip install "coxlm[local]"`).
- Schema API: `questions`, `choice`, `score`, `yesno`, `multilabel`, `multiscore`; typed `Answer` with calibrated confidence and the full distribution.
- `coxlm-serve`: HTTP server with the native `POST /v1/decide`, a System One-compatible `POST /v1/systemone`, a demo page, `/health` and `/v1/models`.
