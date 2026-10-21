"""The two questions the Doom example asks at every decision, shared by finetune_doom.py and play.py. They are the
same questions (and the same wording) as the "schema" in every row of sample.jsonl."""
from coxlm import Questions, choice, yesno

BUTTONS = ("ATTACK", "TURN_LEFT", "TURN_RIGHT")


class Doom(Questions):
    button = choice(
        {"ATTACK": "Fire your weapon down your line of sight. This is the only way to score, but a shot that is not "
                   "lined up hits nothing and wastes a bullet you cannot spare.",
         "TURN_LEFT": "Rotate left. This swings your line of sight toward monsters that are to your left, and away "
                      "from ones on your right.",
         "TURN_RIGHT": "Rotate right. This swings your line of sight toward monsters that are to your right, and away "
                       "from ones on your left."},
        instructions="You are fighting for your life in Doom, surrounded by monsters. Pick the single best button to "
                     "press right now.")
    danger = yesno("Is a monster close enough to hurt you within the next second?")
