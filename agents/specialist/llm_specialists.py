# agents/specialist/llm_specialists.py
import json
import logging
import os
from openai import OpenAI as OpenAIClient
from agents.structs import GameAction

logger = logging.getLogger(__name__)

class LLMSpecialists:
    """Houses the LLM-driven specialists: The Detective and The Grandmaster."""
    def __init__(self):
        self.client = OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY", ""))
        self.model = "gpt-4o-mini"
        self._system_message_detective = {
            "role": "system",
            "content": "You are a detective analyzing game mechanics. Based on a summary of the agent's experiences, you must identify patterns and propose a strategic goal. Do NOT repeat failed strategies. You must call a tool."
        }
        self._system_message_grandmaster = {
            "role": "system",
            "content": "You are a grandmaster tactician. Given a strategic goal and a summary of recent events, create a short, concrete sequence of actions to achieve it. You must call the 'submit_plan' tool."
        }

    # --- UPGRADE: Accepts a concise summary instead of the full history ---
    def detective_analyze_and_hypothesize(self, memory_summary: dict, last_failed_goal: str | None) -> dict:
        """Analyzes a summary of history to infer rules and propose a high-level goal."""
        
        user_prompt = f"Memory Summary:\n{json.dumps(memory_summary, indent=2)}"
        if last_failed_goal:
            user_prompt += f"\n\nIMPORTANT: The previous strategic goal '{last_failed_goal}' failed. Propose a DIFFERENT and more creative strategic goal. Prioritize actions that have been successful in the past."

        messages = [self._system_message_detective, {"role": "user", "content": user_prompt}]
        tools = [{ "type": "function", "function": { "name": "submit_goal_and_hypotheses", "description": "Submit inferred hypotheses and a strategic goal.", "parameters": { "type": "object", "properties": { "hypotheses": { "type": "array", "items": {"type": "string"}, "description": "A list of beliefs about game rules." }, "goal": { "type": "string", "description": "A single, high-level strategic goal to pursue." } }, "required": ["hypotheses", "goal"] } } }]
        
        try:
            response = self.client.chat.completions.create(model=self.model, messages=messages, tools=tools, tool_choice="auto")
            tool_call = response.choices[0].message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            logger.info(f"Detective Goal: {args.get('goal')}")
            logger.info(f"Detective Hypotheses: {args.get('hypotheses')}")
            return args
        except Exception as e:
            logger.error(f"LLM Detective call failed: {e}")
            return {"goal": "Explore randomly due to error.", "hypotheses": []}

    # --- UPGRADE: Accepts a concise summary instead of the full history ---
    def grandmaster_create_plan(self, goal: str, memory_summary: dict) -> list[str]:
        valid_actions = [action.name for action in GameAction]
        prompt = f"Goal: {goal}\n\nRecent Events Summary:\n{json.dumps(memory_summary, indent=2)}\n\nIMPORTANT: You can ONLY use actions from this list: {valid_actions}"
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