# agents/specialist/llm_specialists.py
import json
import logging
import os
from openai import OpenAI as OpenAIClient
from agents.structs import GameAction

logger = logging.getLogger(__name__)

class LLMSpecialists:
    def __init__(self):
        self.client = OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY", ""))
        self.model = "gpt-4o-mini"
        self._system_message_detective = {
            "role": "system",
            "content": "You are a scientific detective analyzing game mechanics from experimental data. Your goal is to build a knowledge base of how the world works and propose the next critical experiment. You must call a tool."
        }
        self._system_message_grandmaster = {
            "role": "system",
            "content": "You are a tactician. Given a high-level goal, create a short, concrete sequence of actions to achieve it based on the provided knowledge. You must call the 'submit_plan' tool."
        }

    def detective_propose_experiment(self, knowledge_summary: dict, recent_events: list[dict]) -> dict:
        """Analyzes knowledge and recent events to propose the next experiment or goal."""
        user_prompt = (
            f"This is our current understanding of the world:\n{json.dumps(knowledge_summary, indent=2)}\n\n"
            f"Here are the most recent events:\n{json.dumps(recent_events, indent=2)}\n\n"
            "Based on this, what is the single most important goal to pursue next? This could be to test a hypothesis, investigate a surprising result, or attempt to achieve what you believe is the game's objective. Formulate new hypotheses based on the new data."
        )
        messages = [self._system_message_detective, {"role": "user", "content": user_prompt}]
        tools = [{ "type": "function", "function": { "name": "submit_goal_and_hypotheses", "description": "Submit updated hypotheses and the next strategic goal.", "parameters": { "type": "object", "properties": { "hypotheses": { "type": "array", "items": {"type": "string"}, "description": "A list of updated beliefs about game rules." }, "goal": { "type": "string", "description": "A single, high-level strategic goal to pursue." } }, "required": ["hypotheses", "goal"] } } }]
        
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

    def grandmaster_create_plan(self, goal: str, knowledge_summary: dict) -> list[str]:
        valid_actions = [action.name for action in GameAction]
        prompt = f"Goal: {goal}\n\nOur current knowledge:\n{json.dumps(knowledge_summary, indent=2)}\n\nIMPORTANT: Create a short plan (1-5 steps) to achieve the goal. You can ONLY use actions from this list: {valid_actions}"
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