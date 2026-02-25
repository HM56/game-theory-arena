"""
Agent Implementation
LLM-powered agents for game theory scenarios
"""

import asyncio
import json
import re
from typing import List
from dataclasses import dataclass

from game_engine import BaseAgent, AgentAction
from llm_providers import (
    LLMOrchestrator, LLMModel,
    GPT_OSS_120B, GPT_OSS_20B, LLAMA_3_1_8B, LLAMA_3_3_70B, QWEN_3_32B, KIMI_K2
)


class LLMAgent(BaseAgent):
    """Agent powered by LLM via Groq"""

    def __init__(
        self,
        agent_id: str,
        model: LLMModel,
        llm_orchestrator: LLMOrchestrator,
        personality: str = "",
        temperature: float = 0.7
    ):
        super().__init__(
            agent_id=agent_id,
            model_name=model.display_name,
            system_prompt=self._build_system_prompt(personality)
        )
        self.model = model
        self.llm = llm_orchestrator
        self.temperature = temperature
        self.personality = personality

    def _build_system_prompt(self, personality: str) -> str:
        base = """You are a rational agent playing a game theory scenario.
Your goal is to maximize your score/payoff.

When deciding on an action:
1. Analyze the current game state
2. Consider what other players might do
3. Think about long-term vs short-term gains
4. Choose the action that best advances your interests

Always respond with your action clearly."""

        if personality:
            return f"{base}\n\nYour personality/strategy: {personality}"
        return base

    async def decide_action(
        self,
        game_state: str,
        available_actions: List[str],
        action_format: str = "free_text",
        system_prompt_override: str = None
    ) -> AgentAction:

        # Use game-provided system prompt if available, otherwise use default
        effective_system_prompt = system_prompt_override or self.system_prompt

        prompt = f"""Current Game State:
{game_state}

"""

        if action_format == "structured" and available_actions:
            prompt += f"""Available actions: {', '.join(available_actions)}

IMPORTANT: Show your reasoning first, then end with a clear decision in the format:
ACTION: [your choice]"""
        elif action_format == "multiple_choice" and available_actions:
            prompt += f"""Choose one of: {', '.join(available_actions)}

IMPORTANT: Show your reasoning first, then end with your clear choice in the format:
ACTION: [your choice]"""
        else:
            prompt += """IMPORTANT: Think through your strategy first, then clearly state your final decision at the end using this exact format:

ACTION: [your choice]

Make sure to put "ACTION:" followed by your choice on its own line at the very end."""

        try:
            response = await self.llm.get_completion(
                prompt=prompt,
                model=self.model.model_id,
                system_prompt=effective_system_prompt,
                temperature=self.temperature
            )

            # For CustomGame, don't extract here - let the game do it
            # Return raw response and let the game's extract_action handle it
            # Just pass through the raw response as the action for now
            return AgentAction(
                agent_id=self.agent_id,
                action=response[:200],  # Truncated for display
                reasoning=response[:500],  # First 500 chars as reasoning
                raw_response=response  # Full response for game to extract from
            )

        except Exception as e:
            # Fallback action
            fallback = available_actions[0] if available_actions else "pass"
            return AgentAction(
                agent_id=self.agent_id,
                action=fallback,
                reasoning=f"Error: {str(e)}",
                raw_response=""
            )

    def _extract_action(self, response: str, available_actions: List[str], action_format: str) -> str:
        """Extract clean action from LLM response - handles thinking + final decision"""

        response_lower = response.strip().lower()

        # For structured games, look for the actual action keywords
        valid_actions = ["cooperate", "defect", "coop", "def"]

        # First, look for explicit decision markers
        decision_markers = [
            "final decision",
            "i choose",
            "i will",
            "my action",
            "decision:",
            "action:",
            "therefore",
            "so i",
        ]

        # Split into lines and look for decision
        lines = response_lower.split('\n')

        # Look for lines with decision markers
        for i, line in enumerate(lines):
            for marker in decision_markers:
                if marker in line:
                    # Extract action from this line
                    for action in valid_actions:
                        if action in line:
                            # Map short forms to full
                            if action in ["coop", "cooperate"]:
                                return "cooperate"
                            return "defect"

        # Look for the last occurrence of action keywords (likely final decision)
        last_coop = response_lower.rfind("coop")
        last_defect = response_lower.rfind("defect")

        # Use whichever appears later (likely the final decision after thinking)
        if last_defect > last_coop:
            return "defect"
        elif last_coop > last_defect:
            return "cooperate"

        # Fallback: check first 200 chars for quick decision
        for action in valid_actions:
            if action in response_lower[:200]:
                if action in ["coop", "cooperate"]:
                    return "cooperate"
                return "defect"

        # Last resort - return a reasonable default
        return "cooperate"


class AgentFactory:
    """Factory for creating different types of agents"""

    def __init__(self, llm_orchestrator: LLMOrchestrator):
        self.llm = llm_orchestrator

    def create_agent(
        self,
        agent_id: str,
        model_name: str,
        personality: str = "",
        temperature: float = 0.7
    ) -> LLMAgent:

        # Map model name to LLMModel
        model_map = {
            "gpt-oss-120b": GPT_OSS_120B,
            "gpt-oss-20b": GPT_OSS_20B,
            "llama-3.1-8b": LLAMA_3_1_8B,
            "llama-3.3-70b": LLAMA_3_3_70B,
            "qwen-3-32b": QWEN_3_32B,
            "kimi-k2": KIMI_K2,
        }

        model = model_map.get(model_name, LLAMA_3_3_70B)

        return LLMAgent(
            agent_id=agent_id,
            model=model,
            llm_orchestrator=self.llm,
            personality=personality,
            temperature=temperature
        )

    def create_multiple_agents(
        self,
        config: List[dict]  # [{"model": "llama-3.3-70b", "count": 2, "personality": "..."}]
    ) -> List[LLMAgent]:
        agents = []
        agent_counter = 0

        for cfg in config:
            for i in range(cfg.get("count", 1)):
                agent_id = f"{cfg['model']}-{agent_counter+1}"
                agents.append(self.create_agent(
                    agent_id=agent_id,
                    model_name=cfg["model"],
                    personality=cfg.get("personality", ""),
                    temperature=cfg.get("temperature", 0.7)
                ))
                agent_counter += 1

        return agents
