# agents/specialist/input_specialist.py
import random
from agents.structs import GameAction

class InputSpecialist:
    def __init__(self):
        self._keyboard_actions = [
            GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
            GameAction.ACTION4, GameAction.ACTION5
        ]
        self._click_attempts = 0

    # --- UPGRADE: Now accepts the knowledge base ---
    def generate_exploratory_action(self, knowledge: dict) -> GameAction:
        """
        Generates an action for interface discovery, avoiding already-known mechanics.
        """
        known_mechanics = knowledge.get('mechanics_model', {}).get('action_effects', {})

        # Find keyboard actions that we still know nothing about
        untried_keys = [
            action for action in self._keyboard_actions 
            if action.name not in known_mechanics or known_mechanics[action.name]['tries'] == 0
        ]

        if untried_keys:
            return untried_keys[0]

        # If all keys have a known effect, default to exploring clicks
        self._click_attempts += 1
        action = GameAction.ACTION6
        action.set_data({
            'x': random.randint(0, 63),
            'y': random.randint(0, 63)
        })
        return action