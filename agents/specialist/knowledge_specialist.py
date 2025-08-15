# agents/specialist/knowledge_specialist.py
import json
import logging

logger = logging.getLogger(__name__)

class KnowledgeSpecialist:
    """
    Manages the agent's distilled knowledge base, summarizing raw memory into beliefs.
    """
    def __init__(self):
        self.knowledge = {
            "known_actions": {},
            # --- THE FIX IS HERE: Initialize as a list ---
            "object_hypotheses": [], 
            "game_goal_hypothesis": {"goal": "Unknown", "confidence": 0.1}
        }

    def get_knowledge_summary(self) -> dict:
        """Returns the current structured knowledge."""
        return self.knowledge

    def update_knowledge(self, new_hypotheses: list, new_goal: str):
        """Updates the knowledge base with new insights from the LLM Detective."""
        logger.info("Updating Knowledge Base with new insights.")
        self.knowledge['game_goal_hypothesis']['goal'] = new_goal
        
        # Now this will correctly append the new list of hypotheses
        self.knowledge['object_hypotheses'].append(new_hypotheses)