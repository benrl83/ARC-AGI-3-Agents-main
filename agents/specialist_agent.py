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
from agents.specialist.reasoning_log_specialist import ReasoningLogSpecialist

logger = logging.getLogger(__name__)

class AgentPhase(Enum):
    VISUAL_ANALYSIS = auto()
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
        self.reasoning_logger = ReasoningLogSpecialist()
        
        loaded_state = self.memory_manager.load_state(self.game_id)
        if loaded_state and loaded_state.get("knowledge"):
            self.knowledge.knowledge = loaded_state["knowledge"]
        
        if self.knowledge.knowledge['strategic_model']['current_goal'] != "Discover a successful action.":
            logger.info("Loaded prior knowledge. Starting with goal-oriented execution.")
            self.phase = AgentPhase.EXECUTION
        else:
            self.phase = AgentPhase.VISUAL_ANALYSIS

        self.current_plan: list[GameAction] = []
        self.last_frame: FrameData | None = None
        self.last_action: GameAction | None = None
        self.recent_goals = []
        self.pending_reasoning_log: str | None = None

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state == GameState.WIN or self.action_counter >= self.MAX_ACTIONS

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if not latest_frame.frame:
            self.phase = AgentPhase.STARTING
            
        if self.phase == AgentPhase.VISUAL_ANALYSIS:
            logger.critical("--- Entering VISUAL ANALYSIS phase: Analyzing initial screen. ---")
            analysis = self.llm.detective_initial_analysis(latest_frame)
            self.knowledge.update_strategic_model(analysis.get("hypotheses", []), analysis.get("goal", ""))
            self.pending_reasoning_log = self.reasoning_logger.format_thought_process(
                self.knowledge.knowledge['strategic_model']['current_goal'],
                self.knowledge.knowledge['strategic_model']['hypotheses']
            )
            logger.info("--- Consulting Grandmaster for initial exploration plan ---")
            plan_actions = self.llm.grandmaster_create_plan(
                self.knowledge.knowledge['strategic_model']['current_goal'], 
                self.knowledge.get_knowledge_summary()
            )
            self.current_plan = [GameAction.from_name(name) for name in plan_actions if "ACTION" in name]
            self.phase = AgentPhase.STARTING

        if self.phase == AgentPhase.STARTING:
            self.phase = AgentPhase.EXPLORATION
            self.last_frame = latest_frame
            self.last_action = GameAction.RESET
            action_to_take = GameAction.RESET
            if self.pending_reasoning_log:
                action_to_take.reasoning = self.pending_reasoning_log
                self.pending_reasoning_log = None
            return action_to_take

        if self.last_frame and self.last_action and self.last_frame.frame:
            delta = self.change_detector.detect_delta(self.last_frame, latest_frame)
            self.memory.record_event(self.last_action, delta)
            last_event = self.memory.events[-1]
            self.knowledge.update_mechanics_from_event(last_event)
            if self.phase == AgentPhase.EXPLORATION and last_event['success']:
                logger.critical(f"--- BREAKTHROUGH! Switching to INVESTIGATION. ---")
                self.phase = AgentPhase.INVESTIGATION
                self.current_plan = []

        if not self.current_plan:
            if len(self.recent_goals) > 3 and len(set(self.recent_goals[-3:])) == 1:
                logger.critical("--- AGENT BORED: Stuck in a thought loop. Forcing broad exploration. ---")
                self.phase = AgentPhase.EXPLORATION
                self.recent_goals = []
            
            if self.phase == AgentPhase.INVESTIGATION or self.phase == AgentPhase.EXECUTION:
                logger.info(f"--- Entering {self.phase.name} phase: Consulting HQ Analyst ---")
                analysis = self.llm.detective_update_strategy(
                    self.knowledge.get_knowledge_summary(), 
                    self.memory.get_recent_history(5),
                    latest_frame
                )
                self.knowledge.update_strategic_model(analysis.get("hypotheses", []), analysis.get("goal", ""))
                self.recent_goals.append(self.knowledge.knowledge['strategic_model']['current_goal'])
                self.pending_reasoning_log = self.reasoning_logger.format_thought_process(
                    self.knowledge.knowledge['strategic_model']['current_goal'],
                    self.knowledge.knowledge['strategic_model']['hypotheses']
                )
                logger.info("--- Consulting Grandmaster to create a plan ---")
                plan_actions = self.llm.grandmaster_create_plan(self.knowledge.knowledge['strategic_model']['current_goal'], self.knowledge.get_knowledge_summary())
                self.current_plan = [GameAction.from_name(name) for name in plan_actions if "ACTION" in name]
                if not self.current_plan:
                    self.phase = AgentPhase.EXPLORATION
            
        if self.current_plan:
            action_to_take = self.current_plan.pop(0)
            if not self.current_plan:
                self.phase = AgentPhase.EXECUTION if self.knowledge.knowledge['strategic_model']['current_goal'] != "Unknown" else AgentPhase.INVESTIGATION
        else:
            logger.info(f"--- In EXPLORATION phase (Turn {self.memory.turn_number + 1}) ---")
            action_to_take = self.input_specialist.generate_exploratory_action(self.knowledge.get_knowledge_summary())
        
        self.last_frame = latest_frame
        self.last_action = action_to_take
        self.guid = latest_frame.guid
        self.data = action_to_take.data if hasattr(action_to_take, 'data') else None

        if self.pending_reasoning_log:
            action_to_take.reasoning = self.pending_reasoning_log
            self.pending_reasoning_log = None
            
        return action_to_take

    def cleanup(self, scorecard: Optional[Any] = None) -> None:
        """Called when a game is over. Saves the agent's brain via the manager."""
        logger.info(f"--- Game Over for {self.game_id}. Saving brain. ---")
        # --- THE FIX IS HERE: The call now matches the manager's function signature ---
        self.memory_manager.save_state(
            game_id=self.game_id,
            memory_events=self.memory.get_full_history(),
            knowledge=self.knowledge.get_knowledge_summary()
        )