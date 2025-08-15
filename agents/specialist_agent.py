# agents/specialist_agent.py
import logging
from enum import Enum, auto
from typing import Optional, Any # <-- Import Optional and Any
import json

from agents.agent import Agent
from agents.structs import FrameData, GameAction, GameState
from agents.specialist.input_specialist import InputSpecialist
from agents.specialist.change_detection_specialist import ChangeDetectionSpecialist
from agents.specialist.memory_specialist import MemorySpecialist
from agents.specialist.llm_specialists import LLMSpecialists
from agents.specialist.persistent_memory_manager import PersistentMemoryManager

logger = logging.getLogger(__name__)

class AgentPhase(Enum):
    STARTING = auto()
    EXPLORATION = auto()
    HYPOTHESIS = auto()
    EXECUTION = auto()

class SpecialistAgent(Agent):
    MAX_ACTIONS = 150
    EXPLORATION_TURNS = 15

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_specialist = InputSpecialist()
        self.change_detector = ChangeDetectionSpecialist()
        self.memory = MemorySpecialist()
        self.llm = LLMSpecialists()
        self.memory_manager = PersistentMemoryManager()
        loaded_state = self.memory_manager.load_state(self.game_id)
        self.hypotheses = loaded_state.get("hypotheses", [])
        self.memory.events = loaded_state.get("memory_events", [])
        self.memory.turn_number = len(self.memory.events)
        if self.memory.turn_number > 0:
            logger.info(f"Loaded {self.memory.turn_number} past events. Skipping initial exploration.")
            self.phase = AgentPhase.HYPOTHESIS
        else:
            self.phase = AgentPhase.STARTING
        self.current_goal = "Explore the interface."
        self.current_plan: list[GameAction] = []
        self.last_frame: FrameData | None = None
        self.last_action: GameAction | None = None
        self.frustration_counter = 0
        self.last_failed_goal: str | None = None

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state == GameState.WIN or self.action_counter >= self.MAX_ACTIONS

    # --- NEW FUNCTION: The "Thinking Cap" memory summarizer ---
    def _summarize_memory_for_llm(self) -> dict:
        """Creates a small, token-efficient summary of the agent's memory."""
        all_events = self.memory.get_full_history()
        action_stats = {}
        
        for event in all_events:
            action_name = event['action']['name']
            if action_name not in action_stats:
                action_stats[action_name] = {'tries': 0, 'successes': 0}
            action_stats[action_name]['tries'] += 1
            if event['success']:
                action_stats[action_name]['successes'] += 1
        
        # Get the last 5 successful events to give context
        last_successful_events = [e for e in all_events if e['success']][-5:]

        return {
            "total_turns_lived": len(all_events),
            "current_hypotheses": self.hypotheses,
            "action_effectiveness": action_stats,
            "recent_successful_events": last_successful_events
        }

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if not latest_frame.frame:
            logger.error("Received an empty frame, possibly after a game error. Re-syncing with RESET.")
            self.phase = AgentPhase.STARTING
            
        if self.phase == AgentPhase.STARTING:
            logger.info("--- STARTING phase: Sending initial RESET ---")
            self.phase = AgentPhase.EXPLORATION
            self.last_frame = latest_frame
            self.last_action = GameAction.RESET
            return GameAction.RESET

        if self.last_frame and self.last_action and self.last_frame.frame:
            delta = self.change_detector.detect_delta(self.last_frame, latest_frame)
            self.memory.record_event(self.last_action, delta)
            if not self.memory.events[-1]['success'] and self.phase == AgentPhase.EXECUTION:
                logger.warning("Action had no effect. Invalidating plan.")
                self.frustration_counter += 1
                self.last_failed_goal = self.current_goal
                self.current_plan = []
            else:
                self.frustration_counter = 0
                self.last_failed_goal = None
        
        if not self.current_plan:
            if self.frustration_counter >= 3:
                logger.critical("--- AGENT FRUSTRATED: Forcing re-exploration ---")
                self.phase = AgentPhase.EXPLORATION
                self.memory.turn_number = self.EXPLORATION_TURNS - 5
                self.frustration_counter = 0
            
            if self.phase == AgentPhase.EXPLORATION and self.memory.turn_number >= self.EXPLORATION_TURNS:
                self.phase = AgentPhase.HYPOTHESIS

            if self.phase == AgentPhase.HYPOTHESIS:
                logger.info("--- Entering HYPOTHESIS phase ---")
                # --- UPGRADE: Use the summary instead of full history ---
                memory_summary = self._summarize_memory_for_llm()
                analysis = self.llm.detective_analyze_and_hypothesize(memory_summary, self.last_failed_goal)
                self.hypotheses = analysis.get("hypotheses", [])
                self.current_goal = analysis.get("goal", "Continue exploring.")
                self.phase = AgentPhase.EXECUTION

            if self.phase == AgentPhase.EXECUTION:
                logger.info("--- Entering EXECUTION phase: creating new plan ---")
                # --- UPGRADE: Use the summary instead of full history ---
                memory_summary = self._summarize_memory_for_llm()
                plan_actions = self.llm.grandmaster_create_plan(self.current_goal, memory_summary)
                self.current_plan = [GameAction.from_name(name) for name in plan_actions if "ACTION" in name]
                if not self.current_plan:
                    logger.info("Plan invalid. Reverting to HYPOTHESIS.")
                    self.phase = AgentPhase.HYPOTHESIS
                    self.current_plan = [self.input_specialist.generate_exploratory_action()]

        if self.current_plan:
            action_to_take = self.current_plan.pop(0)
            if not self.current_plan:
                logger.info("Plan complete. Re-evaluating next turn.")
                self.phase = AgentPhase.HYPOTHESIS
        else:
            logger.info(f"--- In EXPLORATION phase (Turn {self.memory.turn_number + 1}/{self.EXPLORATION_TURNS}) ---")
            action_to_take = self.input_specialist.generate_exploratory_action()
        
        self.last_frame = latest_frame
        self.last_action = action_to_take
        self.guid = latest_frame.guid
        if hasattr(action_to_take, 'data'):
            self.data = action_to_take.data
        else:
            self.data = None
        return action_to_take

    # --- UPGRADE: Fix the cleanup function signature ---
    def cleanup(self, scorecard: Optional[Any] = None) -> None:
        """Called when a game is over. Saves the agent's memory via the manager."""
        logger.info(f"--- Game Over for {self.game_id}. Saving brain. ---")
        self.memory_manager.save_state(
            game_id=self.game_id,
            memory_events=self.memory.get_full_history(),
            hypotheses=self.hypotheses
        )
        # The base class cleanup is empty, so no need to call super().