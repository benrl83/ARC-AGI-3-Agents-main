# agents/specialist/reasoning_log_specialist.py
import json

class ReasoningLogSpecialist:
    """Formats the agent's thoughts for display on the website replay."""
    
    def format_thought_process(self, goal: str, hypotheses: list) -> str:
        """Creates a clean, human-readable string of the agent's reasoning."""
        log_data = {
            "Strategic Goal": goal,
            "Supporting Hypotheses": hypotheses
        }
        # Use pretty-printed JSON for a nice, structured look in the log
        return json.dumps(log_data, indent=2)