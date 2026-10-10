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
pip install "git+https://github.com/Vitund-AI/coxlm.git"                 # the client: no torch, no GPU needed
pip install "coxlm[local] @ git+https://github.com/Vitund-AI/coxlm.git"  # also run checkpoints locally (torch, transformers, peft)
pip install "coxlm[serve] @ git+https://github.com/Vitund-AI/coxlm.git"  # also run the coxlm-serve server
```

Python 3.10 or newer. The base install uses only the standard library. Importing
`coxlm` does not import torch.

## Quickstart

```python
import coxlm
from coxlm import Questions, choice, score, yesno

# Connect to a running coxlm server (no GPU or PyTorch needed on this machine)
model = coxlm.connect("http://localhost:8000")

# Or run a checkpoint on your own GPU (pip install "coxlm[local]"): a model.pt file, a release folder
# (model.safetensors + config.json), or a Hugging Face repo id such as "owner/name"
# model = coxlm.load("path/to/model.pt")


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

## Running a server

```bash
pip install "coxlm[serve] @ git+https://github.com/Vitund-AI/coxlm.git"
coxlm-serve --model path/to/model.pt --encoder Qwen/Qwen3.5-4B-Base --port 8000
```

The checkpoint records the backbone it was trained on, so you can leave out
`--encoder`. `--model` also takes a release folder (`model.safetensors` +
`config.json`) or a Hugging Face repo id (`owner/name`, with `--revision` to pin a
version). Adapter (LoRA) checkpoints and full-weight checkpoints are told apart
automatically, and so are the two readouts: slot-head and pointer-head models
both run, including pointer-head models on hybrid backbones such as Qwen3.5. `python -m coxlm.serve ...` does the same as `coxlm-serve`.
The first run downloads the backbone from the Hugging Face Hub. The backbone
runs in bf16, the precision models are trained and evaluated in, whatever
precision the checkpoint was saved in (`--dtype` overrides it). A 4B backbone
needs about 10 GB of GPU memory.

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
