# agents/specialist_agent.py
import logging
from enum import Enum, auto
from typing import Optional, Any
import json

from agents.agent import Agent
from agents.structs import FrameData, GameAction, GameState
from agents.specialist.input_specialist import InputSpecialist
from agents.specialist.change_detection_specialist import ChangeDetectionSpecialist
from agents.specialist.memory_specialist import MemorySpecialist
from agents.specialist.llm_specialists import LLMSpecialists
from agents.specialist.persistent_memory_manager import PersistentMemoryManager
from agents.specialist.knowledge_specialist import KnowledgeSpecialist

logger = logging.getLogger(__name__)

class AgentPhase(Enum):
    STARTING = auto()
    EXPLORATION = auto()
    INVESTIGATION = auto()
    EXECUTION = auto()

class SpecialistAgent(Agent):
    MAX_ACTIONS = 150
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_specialist = InputSpecialist()
        self.change_detector = ChangeDetectionSpecialist()
        self.memory = MemorySpecialist()
        self.knowledge = KnowledgeSpecialist()
        self.llm = LLMSpecialists()
        self.memory_manager = PersistentMemoryManager()
        
        loaded_state = self.memory_manager.load_state(self.game_id)
        if loaded_state.get("knowledge"):
            self.knowledge.knowledge = loaded_state["knowledge"]
        
        if len(self.knowledge.knowledge['known_actions']) > 0 or len(self.knowledge.knowledge['object_hypotheses']) > 0 :
            logger.info("Loaded prior knowledge. Starting with goal-oriented execution.")
            self.phase = AgentPhase.EXECUTION
        else:
            self.phase = AgentPhase.STARTING

        self.current_goal = "Discover a successful action."
        self.current_plan: list[GameAction] = []
        self.last_frame: FrameData | None = None
        self.last_action: GameAction | None = None

        # --- NEW "Boredom" Mechanic ---
        self.recent_goals = []

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state == GameState.WIN or self.action_counter >= self.MAX_ACTIONS

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if not latest_frame.frame:
            logger.error("Received empty frame. Re-syncing with RESET.")
            self.phase = AgentPhase.STARTING
            
        if self.phase == AgentPhase.STARTING:
            logger.info("--- STARTING: Sending initial RESET ---")
            self.phase = AgentPhase.EXPLORATION
            self.last_frame = latest_frame
            self.last_action = GameAction.RESET
            return GameAction.RESET

        if self.last_frame and self.last_action and self.last_frame.frame:
            delta = self.change_detector.detect_delta(self.last_frame, latest_frame)
            self.memory.record_event(self.last_action, delta)
            last_event = self.memory.events[-1]
            
            if self.phase == AgentPhase.EXPLORATION and last_event['success']:
                logger.critical(f"--- BREAKTHROUGH! Action '{last_event['action']['name']}' had an effect. Switching to INVESTIGATION. ---")
                self.phase = AgentPhase.INVESTIGATION
                self.current_plan = []

        if not self.current_plan:
            # --- UPGRADE: Boredom check ---
            # If the last 3 goals were the same, get bored and explore.
            if len(self.recent_goals) > 3 and len(set(self.recent_goals[-3:])) == 1:
                logger.critical("--- AGENT IS BORED: Stuck in a thought loop. Forcing broad exploration. ---")
                self.phase = AgentPhase.EXPLORATION
                self.recent_goals = [] # Reset goals to allow new thinking later
            
            if self.phase == AgentPhase.INVESTIGATION or self.phase == AgentPhase.EXECUTION:
                logger.info(f"--- Entering {self.phase.name} phase: Consulting Detective ---")
                analysis = self.llm.detective_propose_experiment(self.knowledge.get_knowledge_summary(), self.memory.get_recent_history(5))
                self.knowledge.update_knowledge(analysis.get("hypotheses", []), analysis.get("goal", ""))
                self.current_goal = analysis.get("goal", "Continue exploring.")
                
                # Track the new goal for the boredom check
                self.recent_goals.append(self.current_goal)
                
                logger.info("--- Consulting Grandmaster to create a plan ---")
                plan_actions = self.llm.grandmaster_create_plan(self.current_goal, self.knowledge.get_knowledge_summary())
                self.current_plan = [GameAction.from_name(name) for name in plan_actions if "ACTION" in name]
                
                if not self.current_plan:
                    logger.warning("Grandmaster failed to create a plan. Reverting to exploration.")
                    self.phase = AgentPhase.EXPLORATION
            
        if self.current_plan:
            action_to_take = self.current_plan.pop(0)
            if not self.current_plan:
                logger.info("Plan complete. Will re-evaluate next turn.")
                self.phase = AgentPhase.EXECUTION if self.knowledge.knowledge['game_goal_hypothesis']['goal'] != "Unknown" else AgentPhase.INVESTIGATION
        else:
            logger.info(f"--- In EXPLORATION phase (Turn {self.memory.turn_number + 1}) ---")
            action_to_take = self.input_specialist.generate_exploratory_action()
        
        self.last_frame = latest_frame
        self.last_action = action_to_take
        self.guid = latest_frame.guid
        self.data = action_to_take.data if hasattr(action_to_take, 'data') else None
        return action_to_take

    def cleanup(self, scorecard: Optional[Any] = None) -> None:
        logger.info(f"--- Game Over for {self.game_id}. Saving brain. ---")
        self.memory_manager.save_state(
            game_id=self.game_id,
            memory_events=self.memory.get_full_history(),
            hypotheses=[], # This is now part of the knowledge object
            knowledge=self.knowledge.get_knowledge_summary()
        )