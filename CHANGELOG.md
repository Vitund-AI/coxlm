# Changelog

## 0.1.0 (2026-09-30)

First release: the inference package extracted from the cox research code.

- `coxlm.connect(url)`: a standard-library client for a coxlm server (no torch needed).
- `coxlm.load(path, encoder=...)`: local inference on adapter (LoRA) or full-weight checkpoints, read with the settings recorded in the checkpoint (`pip install "coxlm[local]"`).
- Schema API: `questions`, `choice`, `score`, `yesno`, `multilabel`, `multiscore`; typed `Answer` with calibrated confidence and the full distribution.
- `coxlm-serve`: HTTP server with the native `POST /v1/decide`, a System One-compatible `POST /v1/systemone`, a demo page, `/health` and `/v1/models`.
