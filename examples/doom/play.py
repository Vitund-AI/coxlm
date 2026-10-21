"""Watch Cox play ViZDoom's defend_the_center: at every decision (every 4 game tics) the current frame is written
as the same JSON text the training rows hold, Cox answers the two questions in doom_questions.py, and the player
presses the button Cox picks. Prints each episode's score: +1 per kill, -1 if the player dies (the scenario's
reward); an episode ends when the player dies or after 2,100 tics (one minute).

    pip install "coxlm[local]" vizdoom
    python examples/doom/play.py --adapter cox-doom-adapter              # the fine-tuned model
    python examples/doom/play.py                                         # the base model, for comparison
    python examples/doom/play.py --adapter cox-doom-adapter --window     # show the game window (needs a display)

Episodes are fixed by --seed: episode i is the (i+1)-th episode of a game seeded with it, so runs are comparable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TICS = 4  # game tics per decision
CFG_BUTTONS = ("TURN_LEFT", "TURN_RIGHT", "ATTACK")  # the scenario's available_buttons, in its .cfg order
REACH = 64  # distance at which a melee monster is roughly on the player (for the ETA)
MONSTERS = frozenset(("Demon", "Cacodemon", "ZombieMan", "ShotgunGuy", "HellKnight", "Imp", "Revenant",
                      "MarineChainsawVzd"))
NOTE = ("bearing 0 = dead ahead. POSITIVE bearing means the monster is to your LEFT (turn left to face it); "
        "NEGATIVE means to your RIGHT. closing = how much nearer the monster got since the previous decision "
        "(4 game tics ago; negative = moving away, null = not seen then). eta_tics = game tics until it "
        "reaches you at that speed (35 tics = 1 second; null = not approaching).")


def make_game(seed: int, window: bool = False):
    """defend_the_center with object positions and the player's position and angle exposed."""
    import vizdoom as vzd

    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "defend_the_center.cfg"))
    game.set_doom_config_path(os.path.join(tempfile.gettempdir(), "coxlm_doom_vizdoom.ini"))  # not in the cwd
    game.set_window_visible(window)
    game.set_objects_info_enabled(True)
    game.set_available_game_variables([vzd.GameVariable.HEALTH, vzd.GameVariable.AMMO2, vzd.GameVariable.POSITION_X,
                                       vzd.GameVariable.POSITION_Y, vzd.GameVariable.ANGLE])
    game.set_seed(seed)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    return game


def monsters_of(state) -> list[dict]:
    """Monsters in a frame, nearest first: object id, kind, distance and bearing (degrees, positive = to the left)."""
    if state is None:
        return []
    _, _, px, py, ang = state.game_variables[:5]
    out = []
    for o in state.objects:
        if o.name in MONSTERS:
            dx, dy = o.position_x - px, o.position_y - py
            out.append({"id": o.id, "kind": o.name, "distance": math.hypot(dx, dy),
                        "bearing_deg": (math.degrees(math.atan2(dy, dx)) - ang + 540) % 360 - 180})
    out.sort(key=lambda m: round(m["distance"]))
    return out


def describe(state, prev_state=None, prev_button: str | None = None, max_monsters: int = 8) -> dict:
    """The frame as the training rows describe it: health, ammo, facing, the previous button, and per monster
    (nearest first) its distance, bearing, how much nearer it got since the previous decision and when it arrives."""
    hp, ammo, _, _, ang = state.game_variables[:5]
    before = {m["id"]: m for m in monsters_of(prev_state)}
    monsters = []
    for m in monsters_of(state)[:max_monsters]:
        item = {"kind": m["kind"], "distance": round(m["distance"]), "bearing_deg": round(m["bearing_deg"])}
        p = before.get(m["id"])
        if p is None:
            item["closing"], item["eta_tics"] = None, None
        else:
            closing = p["distance"] - m["distance"]
            item["closing"] = round(closing)
            speed = closing / TICS
            item["eta_tics"] = max(0, round((m["distance"] - REACH) / speed)) if speed > 0.5 else None
        monsters.append(item)
    return {"health": int(hp), "ammo": int(ammo), "facing_deg": round(ang), "previous_button": prev_button,
            "monsters_nearest_first": monsters, "note": NOTE}


def play_episode(model, seed: int, episode: int, window: bool, verbose: bool) -> dict:
    from doom_questions import BUTTONS, Doom

    game = make_game(seed, window)
    for _ in range(episode + 1):
        game.new_episode()
    prev_state, prev_button, decisions, seconds = None, None, 0, 0.0
    pressed = {b: 0 for b in BUTTONS}
    while not game.is_episode_finished():
        state = game.get_state()
        if state is None:
            break
        text = json.dumps(describe(state, prev_state, prev_button), ensure_ascii=False, indent=2)
        t0 = time.perf_counter()
        ans = model.decide(text, Doom)
        seconds += time.perf_counter() - t0
        button = ans.button.choice
        if verbose:
            probs = " ".join(f"{b}={ans.button.probabilities[b]:.2f}" for b in BUTTONS)
            print(f"  tic {state.tic:5d}  {button:<10} {probs}  danger={ans.danger.p_yes:.2f}")
        pressed[button] += 1
        decisions += 1
        game.make_action([int(b == button) for b in CFG_BUTTONS], TICS)
        prev_state, prev_button = state, button
    score = game.get_total_reward()
    game.close()
    return {"episode": episode, "score": score, "decisions": decisions, "buttons": pressed,
            "ms_per_decision": 1000 * seconds / max(decisions, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="Vitund/cox-1-3b", help="the base model export (Hub id or folder)")
    ap.add_argument("--adapter", default=None, help="a fine-tune adapter (finetune_doom.py's output); none = base model")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--window", action="store_true", help="show the game window")
    ap.add_argument("--verbose", action="store_true", help="print every decision")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="write the per-episode results here (JSON)")
    args = ap.parse_args()
    try:
        import vizdoom  # noqa: F401
    except ImportError:
        raise SystemExit("this example needs ViZDoom: pip install vizdoom")

    import coxlm

    model = coxlm.load(args.model, adapter=args.adapter, device=args.device)
    results = []
    for e in range(args.episodes):
        r = play_episode(model, args.seed, e, args.window, args.verbose)
        results.append(r)
        print(f"episode {e:2d}: score {r['score']:+.0f}  ({r['decisions']} decisions, {r['ms_per_decision']:.0f} ms each, "
              f"buttons {r['buttons']})", flush=True)
    scores = [r["score"] for r in results]
    print(f"\nmean score over {len(scores)} episodes (seed {args.seed}): {statistics.fmean(scores):.1f}"
          + (f" (sd {statistics.stdev(scores):.1f})" if len(scores) > 1 else ""))
    if args.out:
        json.dump({"model": args.model, "adapter": args.adapter, "seed": args.seed, "episodes": results,
                   "mean_score": statistics.fmean(scores)}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
