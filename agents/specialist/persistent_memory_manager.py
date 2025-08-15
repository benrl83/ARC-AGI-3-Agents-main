# agents/specialist/persistent_memory_manager.py
import json
import os
import logging

logger = logging.getLogger(__name__)

class PersistentMemoryManager:
    """Handles saving and loading of an agent's 'brain' for specific games."""
    
    def __init__(self, memory_dir="agent_brains"):
        self.memory_dir = memory_dir
        if not os.path.exists(self.memory_dir):
            os.makedirs(self.memory_dir)

    def _get_filepath(self, game_id: str) -> str:
        """Generates a consistent filename for a game's brain file."""
        return os.path.join(self.memory_dir, f"brain_{game_id}.json")

    # --- THE FIX IS HERE: Add 'knowledge' as an argument ---
    def save_state(self, game_id: str, memory_events: list, hypotheses: list, knowledge: dict):
        """Saves the agent's current memory and beliefs to a file."""
        filepath = self._get_filepath(game_id)
        logger.info(f"Saving agent brain for game '{game_id}' to {filepath}")
        
        # We now save the distilled knowledge as the primary object
        state_dump = {
            "knowledge": knowledge,
            "raw_memory_events": memory_events, # Save raw events for debugging
        }
        
        try:
            with open(filepath, "w") as f:
                json.dump(state_dump, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save brain file for {game_id}: {e}")

    def load_state(self, game_id: str) -> dict:
        """Loads the agent's memory and beliefs if a brain file exists."""
        filepath = self._get_filepath(game_id)
        if os.path.exists(filepath):
            try:
                logger.critical(f"--- PREVIOUS BRAIN FOUND for '{game_id}'. Loading past experiences. ---")
                with open(filepath, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load or parse brain file for {game_id}: {e}")
        
        logger.info(f"No previous brain found for '{game_id}'. Starting with a blank page.")
        # Return a structure that matches what we save
        return {"knowledge": None, "raw_memory_events": []}