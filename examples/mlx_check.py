"""Check the MLX backend on an Apple-silicon Mac: the same requests through PyTorch (MPS) and MLX, the largest
probability difference per case, and the per-answer time of each backend (bf16, and MLX with 4-bit weights).

    pip install "coxlm[local,mlx]"      # both backends
    python examples/mlx_check.py        # or: python examples/mlx_check.py /path/to/export

The backends run one after the other (each holds the model, about 6 GB). The cases cover every way a request is
run: a short request (one pass), a long text (cache forks), a question with many options (nested forks), and
order_items. Differences up to about 0.03 are bf16 rounding; the top answers should match.
"""
import gc
import statistics
import sys
import time

import coxlm
from coxlm import Questions, choice, order_items, questions, score, yesno
from coxlm.decide import as_schema

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Vitund/cox-1-3b"


class Ticket(Questions):
    team = choice(["billing", "support", "sales"], instructions="Which team handles this?")
    urgency = score(["low", "medium", "high"], instructions="How urgent is it?")
    refund = yesno("Is the customer asking for a refund?")


TEXT = "My card was charged twice for the same order and I want my money back today."
LONG = " ".join([
    "The supplier shall deliver the goods within thirty days of each purchase order and shall bear the risk of loss "
    "until delivery. Either party may terminate this agreement on ninety days' written notice. The customer shall pay "
    "each invoice within forty-five days; late payments accrue interest at one percent per month."] * 40)
TOPICS = ["termination", "payment terms", "delivery", "risk of loss", "confidentiality", "indemnification",
          "governing law", "warranties", "limitation of liability", "assignment", "notices", "force majeure",
          "insurance", "audit rights", "intellectual property", "non-compete", "severability", "entire agreement",
          "amendments", "dispute resolution"]
CASES = [
    ("short (one pass)", TEXT, Ticket),
    ("long text (forks)", LONG, Ticket),
    ("20 options (nested)", LONG[:600], questions(topic=choice(TOPICS, instructions="What is this clause mainly about?"))),
]
STEPS = ["preheat the oven", "mix the batter", "pour the batter into the tin", "bake for 30 minutes", "let it cool"]


def run(model):
    probs = {}
    for name, text, qs in CASES:  # {field: [probabilities]}
        probs[name] = model.predict([text], as_schema(qs))[0]
    o = order_items(model, STEPS, mode="pairwise", instructions="Order these baking steps.")
    probs["order_items"] = {f"{i}<{j}": [p, 1 - p] for i, row in enumerate(o["pairs"])
                            for j, p in enumerate(row) if p is not None and i < j}
    model.decide(TEXT, Ticket)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        model.decide(TEXT, Ticket)
        times.append((time.perf_counter() - t0) * 1000)
    t_long = time.perf_counter()
    model.decide(LONG, Ticket)
    t_long = (time.perf_counter() - t_long) * 1000
    return probs, statistics.median(times), t_long, o["order"]


def load_and_run(label, **kw):
    t0 = time.perf_counter()
    model = coxlm.load(MODEL, **kw)
    print(f"{label}: loaded in {time.perf_counter() - t0:.1f} s", flush=True)
    out = run(model)
    print(f"{label}: per answer (short request, median of 10) {out[1]:.0f} ms; long text {out[2]:.0f} ms; "
          f"order {out[3]}", flush=True)
    model.close()
    del model
    gc.collect()
    return out


def compare(a, b):
    worst = 0.0
    for case in a:
        d, same = 0.0, True
        for field in a[case]:
            pa, pb = a[case][field], b[case][field]
            d = max(d, max(abs(x - y) for x, y in zip(pa, pb)))
            same &= pa.index(max(pa)) == pb.index(max(pb))
        worst = max(worst, d)
        print(f"  {case:22s} max |dp| {d:.4f}  same top answers: {same}")
    return worst


torch_out = load_and_run("PyTorch", backend="torch")
mlx_out = load_and_run("MLX bf16", backend="mlx")
mlx4_out = load_and_run("MLX 4-bit", backend="mlx", quantize=4)
print("\nMLX bf16 vs PyTorch:")
compare(torch_out[0], mlx_out[0])
print("MLX 4-bit vs PyTorch:")
compare(torch_out[0], mlx4_out[0])
