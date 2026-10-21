"""Fine-tune Cox on Doom decisions, then print the held-out report.

Each row of sample.jsonl is one moment of ViZDoom's defend_the_center scenario, written as JSON text (health, ammo,
which way the player faces, the button pressed last, and the monsters nearest first, with how fast each is closing),
with two questions: which button to press, and whether a monster can hurt the player within a second. The answers
are soft targets: a probability per button from a look-ahead planner that tried each button in the game.

    pip install "coxlm[local]"
    python examples/doom/finetune_doom.py                        # the 2,000-row sample beside this script
    python examples/doom/finetune_doom.py --hub-dataset           # the full set from the Hugging Face Hub
    python examples/doom/play.py --adapter cox-doom-adapter      # then watch it play

Needs a CUDA GPU in practice (lower --micro-batch if it runs out of memory); on a CPU it runs, but very slowly.
"""
from __future__ import annotations

import argparse
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def hub_rows(repo_id: str, revision: str | None) -> list[str]:
    """Every .jsonl file of a Hugging Face dataset repo, downloaded (and cached) on first use."""
    from huggingface_hub import snapshot_download

    folder = snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision, allow_patterns=["*.jsonl"])
    files = sorted(glob.glob(os.path.join(folder, "**", "*.jsonl"), recursive=True))
    if not files:
        raise SystemExit(f"{repo_id}: no .jsonl files in the dataset")
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="Vitund/cox-1-3b", help="the base model export (Hub id or folder)")
    ap.add_argument("--data", default=os.path.join(HERE, "sample.jsonl"), help="a JSONL file of decisions")
    ap.add_argument("--hub-dataset", nargs="?", const="Vitund/cox-doom-decisions", default=None,
                    help="train on a Hugging Face dataset of decisions instead (default id: %(const)s)")
    ap.add_argument("--hub-revision", default=None)
    ap.add_argument("--out", default="cox-doom-adapter")
    ap.add_argument("--steps", type=int, default=None, help="default: about two passes over the training split")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--micro-batch", type=int, default=2, help="examples per forward pass (lower it if memory runs out)")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import coxlm

    from doom_questions import Doom

    data = args.data
    if args.hub_dataset:
        files = hub_rows(args.hub_dataset, args.hub_revision)
        data = [line for f in files for line in open(f) if line.strip()]
        import json
        data = [json.loads(line) for line in data]

    path, report = coxlm.finetune(args.model, data, Doom, args.out, steps=args.steps, lr=args.lr,
                                  batch_size=args.batch_size, micro_batch=args.micro_batch, lora_r=args.lora_r,
                                  holdout=args.holdout, seed=args.seed, device=args.device)
    print(f"\nwatch it play:  python {os.path.join(os.path.relpath(HERE), 'play.py')} --adapter {path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, HERE)
    main()
