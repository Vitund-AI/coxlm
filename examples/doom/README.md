# Doom: fine-tune Cox on game decisions, then watch it play

This example adapts `Vitund/cox-1-3b` to one narrow task, playing ViZDoom's `defend_the_center` scenario, with
`coxlm.finetune`, and then lets the adapted model play. In the scenario the player stands in the middle of a round
room, can only turn left, turn right or shoot, and monsters close in from every side. Each kill scores a point,
dying costs one, and an episode lasts at most one minute of game time.

At every decision (every 4 game tics) the game frame is written out as JSON text (health, ammo, which way the player
faces, the button pressed last, and each monster's distance, bearing and how fast it is closing), and Cox answers two
questions about it: which button to press, and whether a monster can hurt the player within the next second. The
player presses the button Cox picks.

Out of the box, Cox has never seen Doom and plays badly. With the commands below (the 2,000-row sample here and the
default settings), the fine-tune took cox-1-3b from about 0.2 kills per episode to **23.6** on average over 10
episodes at seed 1234 (every episode between 22 and 25, out of a possible 25). The look-ahead planner whose decisions
it learned from scores 25.0.

What we measured on an RTX 4080 SUPER (16 GB) under WSL2: about 10 GB of GPU memory, about 6 s per training step and
42 minutes for the default 425 steps; then about 100 ms per decision while playing, including the game itself.

| fine-tune | held-out accuracy | button accuracy | score per episode |
|---|---|---|---|
| none (the base model) | 79.3% | 65.3% | about 0.2 |
| 30 steps | 86.7% | 80.0% | about 7 to 8 |
| 425 steps (the default) | 93.3% | 90.7% | 23.6 |

A game is a chain of 300 to 400 decisions per episode, and one bad turn at the wrong moment ends it, so a few points
of held-out accuracy make a large difference to the score. The same holds wherever Cox's answers drive a sequence of
actions: judge a fine-tune by the end result, not only by the held-out numbers.

## Files

- `sample.jsonl`: 2,000 decisions (4 MB), a random sample of the 21,543 in our Doom data, stratified on the answers.
  Each row has the frame as text (`state`), the best button and the danger answer (`truth`), the probability per
  answer that training aims for (`targets`), and the questions (`schema`). The targets come from a look-ahead planner
  that tried each button in the game and scored what followed, so ties between equally good buttons share the
  probability.
- `doom_questions.py`: the two questions, as a `coxlm.Questions` class.
- `finetune_doom.py`: fine-tunes the model and prints the held-out report (see the main README).
- `play.py`: plays episodes with the base or the adapted model and prints each episode's score.

## Commands

```bash
git clone https://github.com/Vitund-AI/coxlm.git && cd coxlm
pip install -e ".[local,fast]"                                          # [local] on native Windows (see below)
python examples/doom/finetune_doom.py --out cox-doom-adapter            # about 2 passes over the 1,700 training rows

pip install vizdoom                                                      # only needed for play.py
python examples/doom/play.py                                            # the base model, 20 episodes at seed 1234
python examples/doom/play.py --adapter cox-doom-adapter                 # the adapted model, same episodes
python examples/doom/play.py --adapter cox-doom-adapter --window --episodes 1   # watch one (needs a display)
```

`finetune_doom.py --hub-dataset` trains on the full set from the Hugging Face Hub instead (by default the dataset
`Vitund/cox-doom-decisions`, which is not published yet; pass another id after the flag).

Fine-tuning needs a CUDA GPU in practice: each training example is read as five sequences of about 750 tokens through a
3B-parameter model. With the default `--micro-batch 2` it needs about 10 GB, so a 12 GB card is enough; `--micro-batch 4`
needed about 15 GB on our 16 GB card, and `--micro-batch 1` roughly halves what the examples add. On Windows, use WSL2:
the fast linear-attention kernels need Triton, which has no official native Windows build, and without them training
is much slower. The micro-batch changes speed and memory only, not the result (each step is still 8 examples).

The progress lines show seconds per step and the time left. The loss settles well above zero here, since the
planner's targets are probabilities (often split between two turns): about 0.385 per question is the lowest it can
go on this data. With soft targets like these, the report measures calibration error against the targets'
probabilities rather than against their top answer.

The held-out numbers in the report are measured on rows set aside from the same episodes, and neighbouring frames of
one episode look alike, so they are on the optimistic side. The score from `play.py` is the real test.

## Licences

- This example's code and `sample.jsonl`: Apache 2.0, like the rest of coxlm.
- [ViZDoom](https://github.com/Farama-Foundation/ViZDoom) (`pip install vizdoom`): MIT licence, as its package
  declares. Its game engine is derived from ZDoom; see ViZDoom's repository for the licence terms of those parts.
- The `defend_the_center` scenario (`defend_the_center.cfg` and `defend_the_center.wad`) ships inside the vizdoom
  package as part of ViZDoom. The WAD holds only the map and its scripts; the game's graphics, sounds and monsters
  come from `freedoom2.wad`, Freedoom: Phase 2, which also ships with vizdoom and is free content under the modified
  (3-clause) BSD licence ([freedoom.github.io](https://freedoom.github.io/)). No original Doom game data is needed or
  used.
- The wording of the button question and of the bearing note follows the defend_the_center harness in
  [firelex/jeff](https://github.com/firelex/jeff) (MIT licence, Copyright (c) 2026 Mathias Strasser).
