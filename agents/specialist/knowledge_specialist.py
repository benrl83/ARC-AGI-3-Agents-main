# agents/specialist/knowledge_specialist.py
import logging

logger = logging.getLogger(__name__)

class KnowledgeSpecialist:
    """
    Manages the agent's structured knowledge base, separating low-level
    mechanics from high-level strategic insights.
    """
    def __init__(self):
        self.knowledge = {
            # LEVEL 1: Factual, data-driven model of game physics
            "mechanics_model": {
                "action_effects": {} # e.g., "ACTION1": {"tries": 10, "successes": 9}
            },
            # LEVEL 2: Abstract, inferred model of the game's purpose
            "strategic_model": {
                "hypotheses": [], # e.g., ["The goal is to clear all blue squares."]
                "current_goal": "Discover a successful action."
            }
        }

    def get_knowledge_summary(self) -> dict:
        """Returns the current structured knowledge."""
        return self.knowledge

    def update_mechanics_from_event(self, event: dict):
        """
        Updates the low-level Mechanics Model. Called every turn.
        This is data-driven and does not use an LLM.
        """
        action_name = event['action']['name']
        action_effects = self.knowledge['mechanics_model']['action_effects']

        if action_name not in action_effects:
            action_effects[action_name] = {'tries': 0, 'successes': 0, 'effect_fingerprints': []}
        
        action_effects[action_name]['tries'] += 1
        if event['success']:
            action_effects[action_name]['successes'] += 1
            # A simple "fingerprint" of the effect
            effect_summary = f"{event['effect_delta']['pixels_changed']} pixels changed."
            if effect_summary not in action_effects[action_name]['effect_fingerprints']:
                 action_effects[action_name]['effect_fingerprints'].append(effect_summary)


    def update_strategic_model(self, new_hypotheses: list, new_goal: str):
        """Updates the high-level Strategic Model based on LLM analysis."""
        logger.info("Updating Strategic Model with new insights.")
        self.knowledge['strategic_model']['current_goal'] = new_goal
        self.knowledge['strategic_model']['hypotheses'] = new_hypotheses