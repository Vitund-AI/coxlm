# coxlm

Typed questions in, calibrated probabilities out, in one forward pass.

You give coxlm some text (a "state") and a set of typed questions about it: pick
one of these options, rate it on this scale, is this true. A fine-tuned encoder
reads the state once and answers every question in the same forward pass. It does
not generate tokens, so each answer is a probability distribution over the
options you defined. The model cannot return an option you did not offer. Every
answer comes with a confidence that is calibrated: answers given with confidence
0.9 were right about 90% of the time in evaluation.

The open-weight model is **cox-1-3b** ([Vitund/cox-1-3b](https://huggingface.co/Vitund/cox-1-3b), Apache-2.0).
`coxlm.load("Vitund/cox-1-3b")` downloads and runs it on your own GPU, and
`coxlm.connect()` talks to a running coxlm server.

## Install

From git:

```bash
pip install "git+https://github.com/Vitund-AI/coxlm.git"                 # the client: no torch, no GPU needed
pip install "coxlm[local] @ git+https://github.com/Vitund-AI/coxlm.git"  # also run checkpoints locally (torch, transformers, peft)
pip install "coxlm[local,fast] @ git+https://github.com/Vitund-AI/coxlm.git"  # Linux / WSL2: plus the fast kernels
pip install "coxlm[serve] @ git+https://github.com/Vitund-AI/coxlm.git"  # also run the coxlm-serve server
pip install "coxlm[mlx] @ git+https://github.com/Vitund-AI/coxlm.git"    # Apple silicon: the native MLX backend (no torch)
```

Python 3.10 or newer. The base install uses only the standard library. Importing
`coxlm` does not import torch.

## Requirements and speed

A CUDA GPU is recommended: cox-1-3b needs about 6 GB of GPU memory and answers a short request with a few
questions in a few tens of milliseconds. Without a GPU it still works on the CPU, at a few seconds per answer
(about 3 s for a short request with three questions). If coxlm reports that it is running on the CPU although you
have an NVIDIA GPU, your PyTorch build has no CUDA support: install a CUDA build from
[pytorch.org](https://pytorch.org/get-started/locally/).

On Linux (and WSL2), install the `fast` extra (`coxlm[local,fast]`) for the fast linear-attention kernels. On native Windows the kernels usually don't load (Triton, which
they need, has no official Windows build), so coxlm falls back to slower PyTorch code with the same answers; for
full speed on Windows, use WSL2. The very first answer after installing takes longer (about 25 s on an RTX 4080
SUPER) while the fast kernels compile for your GPU; they are cached, so later runs start straight away.

On a Mac with Apple silicon (M1 or later), install the `mlx` extra (`coxlm[mlx]`): `coxlm.load()` then runs model
exports such as `Vitund/cox-1-3b` on MLX, Apple's own array framework, whose Metal kernels cover the model's
linear-attention layers. It needs no PyTorch and gives the same answers as the PyTorch backend (within bf16
rounding), several times faster: on an Apple-silicon MacBook a short request with a few questions takes about 0.4 s
and a long text about 2 s, against about 4 s and 10 s with PyTorch on the Apple GPU. The MLX backend is new: please
report problems. `coxlm.load(..., quantize=4)` quantizes the backbone's linear layers to 4 bits (about 3 GB instead
of 6), but in our first tests it was no faster than bf16 and changed some answers, so treat it as experimental and
use the default bf16 for real work. Without the `mlx` extra (`coxlm[local]` only), `coxlm.load()` uses PyTorch on
the Apple GPU (MPS), which gives the same answers more slowly; `backend="torch"` or `backend="mlx"` picks one
explicitly.
Intel Macs can't run the model locally (neither PyTorch nor MLX supports them), but they can use `coxlm.connect()`
with a server running elsewhere.

`examples/demo.py` is a first run with the load time and the per-answer time shown separately.

## Quickstart

```python
import coxlm
from coxlm import Questions, choice, score, yesno

# Connect to a running coxlm server (no GPU or PyTorch needed on this machine)
model = coxlm.connect("http://localhost:8000")

# Or run the model on your own GPU (pip install "coxlm[local]"): a Hugging Face repo id, a release folder
# or a model.pt checkpoint
# model = coxlm.load("Vitund/cox-1-3b")


# The questions, as a class: one attribute per question
class Ticket(Questions):
    team = choice(["billing", "support", "sales"], instructions="Which team handles this?")
    urgency = score(["low", "medium", "high"], instructions="How urgent is it?")
    refund = yesno("Is the customer asking for a refund?")


# One text in, one Ticket out
ans = model.decide("My card was charged twice!!", Ticket)

# Each answer carries its probabilities, so the code can act on how sure the model is
if ans.refund.p_yes > 0.8:
    print("Refund request")
elif ans.team.confidence < 0.6:
    print("Unsure which team:", ans.team.probabilities)
else:
    print("Route to", ans.team.choice)
print("Urgency (1 = low, 3 = high):", round(ans.urgency.score, 1))
```

`connect()` and `load()` return models with the same methods:

| method | takes | returns |
|---|---|---|
| `decide(state, questions)` | one state | its answers |
| `decide_batch(states, questions)` | a list (or any iterable) of states | a list of answers, in order, from one request |
| `decide_iter(states, questions, batch_size=32)` | any iterable: a generator, a file, a cursor | answers one at a time, in order, sent in batches |

A state can be a string, a dict (rendered as `key: value` lines) or a list (rendered as numbered lines), so a list
passed to `decide` is one state, not several. Every question is answered in the same pass and independently of the
others.

`questions` is a `Questions` subclass, as above, or a schema built with `questions(...)`:

```python
from coxlm import questions

schema = questions(
    team=choice(["billing", "support", "sales"], instructions="Which team handles this?"),
    refund=yesno("Is the customer asking for a refund?"),
)
ans = model.decide("My card was charged twice!!", schema)
```

Either way, answers are read by attribute (`ans.team`) or by name (`ans["team"]`), and iterate like a dict
(`for name, answer in ans.items()`). With a class, the answers are an instance of it: editors complete `ans.` with
your question names, and type checkers report a misspelt one. Subclasses inherit their parents' questions.

## Using answers in your code

Being unsure is a case of its own. `pick(min_confidence)` returns the top option (`"yes"` or `"no"` for a yes/no
question) when the model is at least that sure, and `None` otherwise, so a `match` can branch on it:

```python
match ans.team.pick(0.7):
    case "billing":
        send_to_billing(ticket)
    case "support" | "sales" as team:
        send_to(team, ticket)
    case None:
        ask_a_person(ticket)        # the model is not sure which team
```

Answers also work with Python's structural patterns:

```python
from coxlm import Answer

match ans.team:
    case Answer(choice="billing", confidence=c) if c > 0.9:
        refund_automatically(ticket)
    case Answer(choice="billing"):
        queue_for_billing(ticket)

match ans:
    case {"refund": Answer(p_yes=p)} if p > 0.8:
        start_refund(ticket)
```

`ranked()` lists the options most likely first, for "the two most likely teams" or a fallback order:

```python
for team, p in ans.team.ranked()[:2]:
    print(f"{team}: {p:.0%}")
```

Large inputs stream through `decide_iter` without being held in memory:

```python
with open("tickets.txt") as f:
    for ans in model.decide_iter((line.strip() for line in f), Ticket, batch_size=64):
        if ans.refund.pick(0.9) == "yes":
            flag_for_refund(ans)
```

### Releasing a model

Both kinds of model work as context managers. A local model's weights and GPU memory are released when the block
ends, so a script can run one model after another on the same GPU:

```python
with coxlm.load("path/to/model.pt") as model:
    results = model.decide_batch(texts, Ticket)
# the GPU memory is free again here
```

`model.close()` does the same without a `with` block. A closed model raises an error if it is used again.

## The three question types

| type | build it with | what you read |
|---|---|---|
| choice | `choice(options, instructions="...")` | `.choice` (the top option), `.confidence` (its probability), `.probabilities` |
| score | `score(levels, instructions="...", values=None)` | `.score` (the expected level, which can fall between two levels), `.confidence`, `.probabilities`, `.legend` |
| yes/no | `yesno("question", criteria=None)` | `.p_yes` (P(yes)), `.confidence`, `.probabilities` |

Every answer also has `.pick(min_confidence)` and `.ranked()` (see [Using answers in your code](#using-answers-in-your-code)).

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

## Putting items in order

`order_items` puts a set of items (events, steps, tasks) in order. It builds on the question types above and needs
no special model support.

```python
from coxlm import order_items

events = [
    "checkout latency alarms fire",
    "the configuration change is deployed",
    "the change is rolled back",
    "checkout recovers",
]
result = order_items(model, events, mode="pairwise", context=incident_review_text,
                     instructions="In what order did these events happen?")
print(result["ordered_items"])
```

Items are listed in a random order in the prompt, so the order you pass them in carries no information.

| mode | questions asked | what you get |
|---|---|---|
| `"score"` (default) | one per item: which position does it occupy? (n questions) | `order`, `ordered_items`, `positions` (each item's expected position and distribution over positions) |
| `"pairwise"` | one per pair: does i come before j? (n(n-1)/2 questions) | everything below |

Pairwise mode costs more questions but reads more out of the answers. It thresholds the pairwise probabilities into
the precedences the model is confident about (`epsilon`, default 0.15) and returns:

- `order` / `ordered_items`: the most likely order consistent with those precedences;
- `graph`: the confident precedences, as `[before, after, probability]` edges (transitively reduced);
- `levels`: groups of items that can happen at the same time;
- `ranges`: each item's feasible positions, `[earliest, latest]`;
- `flexible_pairs`: pairs the model leaves free, and `unresolved_pairs`: pairs it is unsure about *and* contradicts
  itself on;
- `consistency`: the share of item triples whose pairwise answers do not form a cycle (1.0 is fully consistent);
- `linear_extensions` and `top_orders`: how many orders fit the precedences, and the most probable complete orders
  with their probabilities (up to 9 items).

When a set of steps has no single correct order, the graph and `levels` are the useful output, not `order`. Worked
examples with real output: [vitund.ai/open-source/coxlm/examples](https://vitund.ai/open-source/coxlm/examples).

## Fine-tune on your own data

When the model gets your task mostly right but not right enough, `coxlm.finetune` adapts it to your labelled
examples. It trains a small LoRA adapter (and, by default, the head's readout weights) on top of a model export and
leaves the base model unchanged, so one base can serve several adapters. It needs the `local` extra and, in practice,
a CUDA GPU.

```python
import coxlm

examples = [
    ("My card was charged twice!!", {"team": "billing", "refund": "yes"}),
    ("The app crashes when I open settings.", {"team": "support", "refund": "no"}),
    # ... a few hundred or more
]
path, report = coxlm.finetune("Vitund/cox-1-3b", examples, Ticket, "ticket-adapter")

model = coxlm.load("Vitund/cox-1-3b", adapter="ticket-adapter")
model.decide("I was billed for a plan I cancelled.", Ticket)
```

An answer can be the correct option, or a probability per option (`{"billing": 0.7, "support": 0.3}`) when the
label itself is uncertain. Examples can also come from a JSONL file with one `{"state": ..., "truth": {...}}` row
per line (`"targets"` for probabilities, and an optional per-row `"schema"`). Labels for some questions may be
missing; those questions are left out of that example's loss.

Before training, a share of the examples (`holdout`, 15% by default, stratified on the labels) is set aside. After
training, coxlm fits one temperature on that held-out part, a single number that makes the probabilities match how
often the model is actually right there, and stores it in the adapter, where `load` applies it. It then prints a
report, also saved as `report.json`, comparing the base model and the adapted one on the held-out part: accuracy;
skill over always answering the most common label (0 means no better than that, 1 means always right); calibration
error (ECE); and the share of decisions you could hand to the model at 95% accuracy, taking its most confident
answers first. It also asks both models a small built-in set of general questions and warns if the adapted model
got worse at them, which is a sign it changed more than your task.

What to expect: the report tells you whether the adapter helped, on examples it was not trained on. Aim for a few
hundred clear examples per question or more; with a few dozen, the held-out numbers are too rough to trust, and
coxlm says so. A fine-tune teaches the model your labels; it cannot fix labels that are inconsistent, and it does not make the
model better at tasks you did not train on. Keep the questions' wording the same at inference as in training.

The main settings are `steps` (by default about two passes over your examples), `lr` (1e-4), `batch_size` (8),
`lora_r` (8) and `train_head` (the head's readout weights; the temperature is left to the held-out fit).
`micro_batch` (2) trades memory for speed without changing the result: the Doom example needs about 10 GB at 2 and
about 15 GB at 4, and trained in 42 minutes on an RTX 4080 SUPER under WSL2 (see [examples/doom](examples/doom)).
Progress lines show seconds per step and the time left. When the answers are probabilities, the report measures
calibration error against those probabilities. `load(..., merge=True)` folds the adapter into the base weights for slightly faster
answers. An adapter records the exact base it was trained on, and `load` refuses it on any other base. Adapters run
on the PyTorch backend; MLX support comes later. [examples/doom](examples/doom) fine-tunes the model to play Doom.

## Running a server

```bash
pip install "coxlm[serve] @ git+https://github.com/Vitund-AI/coxlm.git"
coxlm-serve --model Vitund/cox-1-3b --port 8000
```

The checkpoint records the backbone it was trained on, so you can leave out
`--encoder`. `--model` also takes a release folder (`model.safetensors` +
`config.json`) or a Hugging Face repo id (`owner/name`, with `--revision` to pin a
version). Adapter (LoRA) checkpoints and full-weight checkpoints are told apart
automatically, and so are the two readouts: slot-head and pointer-head models
both run, including pointer-head models on hybrid backbones such as Qwen3.5. `python -m coxlm.serve ...` does the same as `coxlm-serve`.
The first run downloads the backbone from the Hugging Face Hub. The backbone
runs in bf16, the precision models are trained and evaluated in, whatever
precision the checkpoint was saved in (`--dtype` overrides it). cox-1-3b needs
about 6 GB of GPU memory for 1k-token texts and about 7 GB at 8k.

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

- `COXLM_BACKEND=torch|mlx` picks the backend `coxlm.load()` uses when none is passed (default: MLX on Apple
  silicon when the `mlx` extra is installed and the model is an export, PyTorch everywhere else).

- `COX_QUANT=fp8|fp4w|fp4` simulates FP8 or NVFP4 quantized inference after
  loading. It measures the accuracy cost only and does not make inference faster.
- `COX_HUB_CONV1D=0` turns off the Hugging Face hub causal-conv1d kernel. That
  kernel is used for Qwen3.5 backbones when the `kernels` package is installed
  (`pip install "coxlm[fast]"`). It works offline (`HF_HUB_OFFLINE=1`) once it has
  been downloaded by one online run.
- `COXLM_COMPILE=1` compiles each backbone block with `torch.compile`: about 19%
  faster on long texts, no change on short ones, and roughly 100 s of compiling
  over the first requests. Useful for long-running servers.
- `COXLM_KEEP_CACHE_GB` (default 4): the server gives cached GPU memory back to
  the system when more than this much is unused, for example after a very long text.

## License

Apache 2.0. See `LICENSE` and `NOTICE`. Backbones and any checkpoints carry
their own licences.
