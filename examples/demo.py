"""A first run of Cox: three typed questions about one text, with the load time and the per-answer time shown
separately (the first answer includes one-off warm-up, so the steady per-answer time is the median of ten).

    pip install "coxlm[local] @ git+https://github.com/Vitund-AI/coxlm.git"
    python examples/demo.py
"""
import statistics
import time

import torch

import coxlm
from coxlm import Questions, choice, score, yesno


class Ticket(Questions):
    team = choice(["billing", "support", "sales"], instructions="Which team handles this?")
    urgency = score(["low", "medium", "high"], instructions="How urgent is it?")
    refund = yesno("Is the customer asking for a refund?")


TEXT = "My card was charged twice for the same order and I want my money back today."
LEVELS = ["low", "medium", "high"]

print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

print("loading Vitund/cox-1-3b (the first run downloads about 5.8 GB)...", flush=True)
t0 = time.perf_counter()
model = coxlm.load("Vitund/cox-1-3b")
print(f"load (download or cache, weights to the device): {time.perf_counter() - t0:.1f} s")
print("backend:", "MLX (Apple silicon)" if type(model).__module__.endswith("mlx_backend") else "PyTorch")

print("first answer...", flush=True)
t0 = time.perf_counter()
ans = model.decide(TEXT, Ticket)  # the first call includes one-off warm-up
print(f"first answer (with warm-up): {(time.perf_counter() - t0) * 1000:.0f} ms")

times = []
print("timing 10 more answers:")
for i in range(10):  # the steady state: what each request costs
    t0 = time.perf_counter()
    ans = model.decide(TEXT, Ticket)
    times.append((time.perf_counter() - t0) * 1000)
    print(f"  {i + 1:2d}/10  {times[-1]:6.0f} ms  team={ans.team.choice} ({ans.team.confidence:.2f})  "
          f"urgency={ans.urgency.score:.2f}  refund p_yes={ans.refund.p_yes:.2f}", flush=True)
print(f"per answer, median of 10: {statistics.median(times):.0f} ms (min {min(times):.0f}, max {max(times):.0f})")

level = LEVELS[min(len(LEVELS) - 1, max(0, round(ans.urgency.score) - 1))]
print()
print("team:   ", ans.team.choice, f"(confidence {ans.team.confidence:.2f})")
print("urgency:", level, f"(score {ans.urgency.score:.2f} on low=1 ... high=3)")
print("refund: ", f"p_yes = {ans.refund.p_yes:.2f}")
