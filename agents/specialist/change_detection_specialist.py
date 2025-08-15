# agents/specialist/change_detection_specialist.py
from agents.structs import FrameData

class ChangeDetectionSpecialist:
    """Compares frames before and after an action to find the exact effect."""
    def detect_delta(self, frame_before: FrameData, frame_after: FrameData) -> dict:
        """
        Calculates a detailed, data-driven 'delta' between two frames.
        """
        pixels_changed = 0
        changes = []
        
        grid_before = frame_before.frame[0]
        grid_after = frame_after.frame[0]

        for y in range(len(grid_before)):
            for x in range(len(grid_before[y])):
                if grid_before[y][x] != grid_after[y][x]:
                    pixels_changed += 1
                    changes.append({
                        "pos": (x, y),
                        "before": grid_before[y][x],
                        "after": grid_after[y][x]
                    })
        
        return {
            "pixels_changed": pixels_changed,
            "score_change": frame_after.score - frame_before.score,
            "game_state_change": frame_after.state.name if frame_before.state != frame_after.state else None,
            "specific_changes": changes[:20]
        }