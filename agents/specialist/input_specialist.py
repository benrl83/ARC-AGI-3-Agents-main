# agents/specialist/input_specialist.py
import random
from agents.structs import GameAction

class InputSpecialist:
    """Generates actions to explore the game's interface systematically."""
    def __init__(self):
        self._keyboard_actions = [
            GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
            GameAction.ACTION4, GameAction.ACTION5
        ]
        self._tried_keyboard_actions = set()
        self._click_attempts = 0

    def generate_exploratory_action(self) -> GameAction:
        """
        Generates the next logical action for interface discovery.
        First tries all keyboard inputs, then starts clicking randomly.
        """
        untried_keys = [a for a in self._keyboard_actions if a not in self._tried_keyboard_actions]
        if untried_keys:
            action = untried_keys[0]
            self._tried_keyboard_actions.add(action)
            return action

        self._click_attempts += 1
        action = GameAction.ACTION6
        action.set_data({
            'x': random.randint(0, 63),
            'y': random.randint(0, 63)
        })
        return action