# agents/specialist/memory_specialist.py
from agents.structs import GameAction

class MemorySpecialist:
    """Records a detailed, structured history of all cause-and-effect events."""
    def __init__(self):
        self.events = []
        self.turn_number = 0

    def record_event(self, action: GameAction, delta: dict):
        self.turn_number += 1
        
        action_had_effect = delta.get("pixels_changed", 0) > 0 or delta.get("score_change", 0) != 0
        
        action_data = None
        if hasattr(action, 'data'):
            action_data = action.data
        
        event = {
            "turn": self.turn_number,
            "action": {"name": action.name, "data": action_data},
            "effect_delta": delta,
            "success": action_had_effect
        }
        self.events.append(event)
    
    def get_full_history(self):
        return self.events

    def get_recent_history(self, num_events: int = 20):
        return self.events[-num_events:]