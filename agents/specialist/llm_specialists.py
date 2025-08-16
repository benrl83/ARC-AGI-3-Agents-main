# agents/specialist/llm_specialists.py
import json
import logging
import os
from openai import OpenAI as OpenAIClient
from agents.structs import GameAction, FrameData

logger = logging.getLogger(__name__)

class LLMSpecialists:
    def __init__(self):
        self.client = OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY", ""))
        self.model = "gpt-4o-mini"
        self._system_message_detective = { "role": "system", "content": "You are a brilliant HQ Analyst interpreting field data..." }
        self._system_message_grandmaster = { "role": "system", "content": "You are a tactician..." }

    # --- NEW FUNCTION for "Look before you leap" ---
    def detective_initial_analysis(self, initial_frame: FrameData) -> dict:
        """Performs a special, one-time analysis of the starting screen."""
        logger.info("Performing initial visual analysis.")
        
        def pretty_print_grid(grid: list[list[int]]) -> str:
            return "\n".join(["".join([f"{cell:2}" for cell in row]) for row in grid])

        user_prompt = (
            "You are seeing this game for the first time. Here is the initial screen. Based on the visual layout, what are the most interesting coordinates to click? Identify distinct objects and suggest a short (3-5 step) 'click exploration plan' to test the most promising visual elements. Formulate your initial hypotheses about the game."
            f"\n\nINITIAL SCREEN:\nScore: {initial_frame.score}\nGrid:\n{pretty_print_grid(initial_frame.frame[0])}"
        )

        messages = [self._system_message_detective, {"role": "user", "content": user_prompt}]
        tools = [{ "type": "function", "function": { "name": "submit_initial_analysis", "description": "Submit initial hypotheses and a targeted exploration goal.", "parameters": { "type": "object", "properties": { "hypotheses": { "type": "array", "items": {"type": "string"}, "description": "A list of initial beliefs about the game's symbols and purpose." }, "goal": { "type": "string", "description": "A focused, short-term exploration goal based on the visuals." } }, "required": ["hypotheses", "goal"] } } }]
        
        try:
            response = self.client.chat.completions.create(model=self.model, messages=messages, tools=tools, tool_choice="auto")
            tool_call = response.choices[0].message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            return args
        except Exception as e:
            logger.error(f"LLM initial analysis failed: {e}")
            return {"goal": "Explore interface randomly.", "hypotheses": ["Initial visual analysis failed."]}

    def detective_update_strategy(self, knowledge: dict, recent_events: list[dict], current_frame: FrameData) -> dict:
        # ... (this function is unchanged) ...
        def pretty_print_grid(grid: list[list[int]]) -> str: return "\n".join(["".join([f"{cell:2}" for cell in row]) for row in grid])
        user_prompt = (f"CURRENT VISUAL STATE:\nScore: {current_frame.score}\nGrid:\n{pretty_print_grid(current_frame.frame[0])}\n\n" f"CURRENT KNOWLEDGE BASE:\n{json.dumps(knowledge, indent=2)}\n\n" f"MOST RECENT EVENTS:\n{json.dumps(recent_events, indent=2)}\n\n" "Analyze all information. Update the strategic model by defining the next high-level goal and providing your reasoning as a new set of hypotheses.")
        messages = [self._system_message_detective, {"role": "user", "content": user_prompt}]
        tools = [{ "type": "function", "function": { "name": "submit_strategic_update", "description": "Submit the updated high-level strategy.", "parameters": { "type": "object", "properties": { "hypotheses": { "type": "array", "items": {"type": "string"}, "description": "A list of updated beliefs about the game's meaning, goals, and tactics." }, "goal": { "type": "string", "description": "A single, high-level strategic goal to pursue next." } }, "required": ["hypotheses", "goal"] } } }]
        try:
            response = self.client.chat.completions.create(model=self.model, messages=messages, tools=tools, tool_choice="auto")
            tool_call = response.choices[0].message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            logger.info(f"HQ Analyst Goal: {args.get('goal')}")
            logger.info(f"HQ Analyst Hypotheses: {args.get('hypotheses')}")
            return args
        except Exception as e:
            logger.error(f"LLM Detective call failed: {e}")
            return {"goal": "Explore randomly due to error.", "hypotheses": []}

    def grandmaster_create_plan(self, goal: str, knowledge: dict) -> list[str]:
        # ... (this function is unchanged) ...
        valid_actions = [action.name for action in GameAction]
        prompt = f"Goal: {goal}\n\nKnown Game Mechanics:\n{json.dumps(knowledge.get('mechanics_model'), indent=2)}\n\nIMPORTANT: Create a short plan (1-5 steps) to achieve the goal. You can ONLY use actions from this list: {valid_actions}"
        messages = [self._system_message_grandmaster, {"role": "user", "content": prompt}]
        tools = [{ "type": "function", "function": { "name": "submit_plan", "description": "Submit a sequence of actions.", "parameters": { "type": "object", "properties": { "action_sequence": { "type": "array", "items": {"type": "string", "enum": valid_actions}, "description": "A list of action names from the allowed list." } }, "required": ["action_sequence"] } } }]
        try:
            response = self.client.chat.completions.create(model=self.model, messages=messages, tools=tools, tool_choice="auto")
            tool_call = response.choices[0].message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            plan = args.get("action_sequence", [])
            logger.info(f"Grandmaster Plan: {plan}")
            return plan
        except Exception as e:
            logger.error(f"LLM Grandmaster call failed: {e}")
            return []