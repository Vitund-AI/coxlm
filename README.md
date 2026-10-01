# coxlm

Typed questions in, calibrated probabilities out, in one forward pass.

You give coxlm some text (a "state") and a set of typed questions about it: pick
one of these options, rate it on this scale, is this true. A fine-tuned encoder
reads the state once and answers every question in the same forward pass. It does
not generate tokens, so each answer is a probability distribution over the
options you defined. The model cannot return an option you did not offer. Every
answer comes with a confidence that is calibrated: answers given with confidence
0.9 were right about 90% of the time in evaluation.

This is a research build. **Model weights are not distributed yet.** Use
`coxlm.connect()` to talk to a running coxlm server. If you have a checkpoint,
`coxlm.load()` runs it on your own GPU.

## Install

From git:

```bash
pip install "git+<repository url>"                 # the client: no torch, no GPU needed
pip install "coxlm[local] @ git+<repository url>"  # also run checkpoints locally (torch, transformers, peft)
pip install "coxlm[serve] @ git+<repository url>"  # also run the coxlm-serve server
```

Python 3.10 or newer. The base install uses only the standard library. Importing
`coxlm` does not import torch.

## Quickstart

```python
import coxlm
from coxlm import questions, choice, score, yesno
model = coxlm.connect("http://boron:8000")      # remote: talks to /v1/decide on a coxlm server
# or: model = coxlm.load("path/to/model.pt", encoder="Qwen/Qwen3.5-4B-Base")   # local GPU
schema = questions(
    team=choice(["billing", "support", "sales"], instructions="Which team handles this?"),
    urgency=score(["low", "medium", "high"], instructions="How urgent is it?"),
    refund=yesno("Is the customer asking for a refund?"))
[ans] = model.decide(["My card was charged twice!!"], schema)
if ans["refund"].p_yes > 0.8: ...
elif ans["team"].confidence < 0.6: print(ans["team"].probabilities)
```

`connect()` and `load()` return models with the same contract:
`decide(states, schema)` returns one dict per state that maps each question
name to an `Answer`. A state can be a string, a dict (rendered as `key: value`
lines) or a list (rendered as numbered lines). Every question in the schema is
answered in the same pass and independently of the others.

## The three question types

| type | build it with | what you read |
|---|---|---|
| choice | `choice(options, instructions="...")` | `.choice` (the top option), `.confidence` (its probability), `.probabilities` |
| score | `score(levels, instructions="...", values=None)` | `.score` (the expected level, which can fall between two levels), `.confidence`, `.probabilities`, `.legend` |
| yes/no | `yesno("question", criteria=None)` | `.p_yes` (P(yes)), `.confidence`, `.probabilities` |

- `options` / `levels` is a list of names or a dict `{name: description}`.
  Descriptions are read by the model, so a short definition of each option helps.
- Score levels are ordered low to high. By default `.score` runs from 1 to k.
  Pass `values=[...]` to read it in your own units instead, e.g. `values=[-2, -1, 0, 1, 2]`.
- `yesno` takes optional `criteria={"no": "...", "yes": "..."}` that say what each answer means.
- `Answer.probabilities` is always the full distribution. For choice and yes/no
  questions, `.confidence` is the probability of the top option. For a score,
  it is the probability mass on the level or levels that `.score` sits on or between.

`multilabel(features)` (one independent yes/no per feature) and
`multiscore(aspects, levels)` (one score per aspect on a shared scale) build
schemas of many questions. All of them are answered in one pass.

## Running a server

```bash
pip install "coxlm[serve] @ git+<repository url>"
coxlm-serve --model path/to/model.pt --encoder Qwen/Qwen3.5-4B-Base --port 8000
```

The checkpoint records the backbone it was trained on, so you can leave out
`--encoder`. Adapter (LoRA) checkpoints and full-weight checkpoints are told
apart automatically. `python -m coxlm.serve ...` does the same as `coxlm-serve`.
The first run downloads the backbone from the Hugging Face Hub. A 4B backbone
needs about 10 GB of GPU memory in bf16.

| endpoint | what it does |
|---|---|
| `GET /` | demo page: enter states and questions and see the answers |
| `GET /health` | liveness check, plus the served model and checkpoint |
| `GET /v1/models` | the served model id |
| `POST /v1/decide` | the native API that `coxlm.connect()` uses |
| `POST /v1/systemone` | System One-compatible endpoint (`noul` / `choice` / `score` questions) |
| `POST /infer` | the demo page's endpoint, which also handles `order`, `features` and `multiscore` questions |

`POST /v1/decide` request and response:

```json
{"states": ["My card was charged twice!!"],
 "questions": {
   "team":    {"type": "choice", "instructions": "Which team handles this?", "options": ["billing", "support", "sales"]},
   "urgency": {"type": "score",  "instructions": "How urgent is it?", "options": ["low", "medium", "high"]},
   "refund":  {"type": "yesno",  "instructions": "Is the customer asking for a refund?"}}}
```

```json
{"model": "coxlm-qwen3.5-4b",
 "answers": [{"team":    {"kind": "choice", "choice": "billing", "confidence": 0.93, "probabilities": {"billing": 0.93, "...": 0.0}},
              "urgency": {"kind": "score", "score": 2.4, "confidence": 0.81, "probabilities": {"...": 0.0}},
              "refund":  {"kind": "yesno", "p_yes": 0.88, "confidence": 0.88, "probabilities": {"no": 0.12, "yes": 0.88}}}],
 "infer_ms": 41.0}
```

The numbers above only show the shape of a response; they are not real model output.
`options` can also be an object of `{option: description}`. A score question
takes an optional `values` list. A yes/no question takes optional
`criteria: {"no": ..., "yes": ...}`.

The server has no authentication. Run it on a trusted network.

## Environment variables

- `COX_QUANT=fp8|fp4w|fp4` simulates FP8 or NVFP4 quantized inference after
  loading. It measures the accuracy cost only and does not make inference faster.
- `COX_HUB_CONV1D=0` turns off the Hugging Face hub causal-conv1d kernel. That
  kernel is used for Qwen3.5 backbones when the `kernels` package is installed
  (`pip install "coxlm[fast]"`).

## License

Apache 2.0. See `LICENSE` and `NOTICE`. Backbones and any checkpoints carry
their own licences.
