"""
Agent Implementation
LLM-powered agents for game theory scenarios
"""

import asyncio
import json
import re
import logging
from typing import List
from dataclasses import dataclass

from game_engine import BaseAgent, AgentAction
from llm_providers import (
    LLMOrchestrator, LLMModel,
    GPT_OSS_120B, GPT_OSS_20B, LLAMA_3_1_8B, LLAMA_3_3_70B, QWEN_3_32B, KIMI_K2
)

logger = logging.getLogger(__name__)


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
        base = """You are a rational agent in a game theory competition.

PRIMARY OBJECTIVE: Maximize your total score by the end of the game.

CRITICAL OUTPUT FORMAT:
You MUST separate your thinking from your final action using XML-style tags:

<reasoning>
Your strategic analysis goes here.
- Analyze current game state
- Consider opponent behavior patterns
- Weigh short-term vs long-term outcomes
</reasoning>

<action>
Your final action choice goes here - JUST the action value
</action>

EXAMPLES:
If choosing to cooperate:
<reasoning>
Given the repeated nature and my opponent's history of cooperation, I'll cooperate to build trust.
</reasoning>
<action>
cooperate
</action>

If bidding 15:
<reasoning>
With 3 rounds left and being behind, I need an aggressive bid. 15 is optimal.
</reasoning>
<action>
15
</action>

IMPORTANT: The <action> tag must contain ONLY your action - no explanations, no "I choose", just the raw action value."""

        if personality:
            return f"""{base}

ADDITIONAL STRATEGIC CONTEXT:
{personality}

Remember: Your personality guides your strategy, but your action must still be output in the <action> tags."""
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

        # Build the prompt with clear structure
        prompt = f"""=== CURRENT GAME STATE ===
{game_state}"""

        # Add action guidance based on format
        if action_format == "structured" and available_actions:
            prompt += f"""

=== YOUR OPTIONS ===
{', '.join(available_actions)}

=== YOUR TASK ===
Analyze and choose. Use this format:
<reasoning>
Your strategic analysis
</reasoning>
<action>
your_choice
</action>"""
        elif action_format == "multiple_choice" and available_actions:
            prompt += f"""

=== YOUR OPTIONS ===
Choose one: {', '.join(available_actions)}

=== YOUR TASK ===
Analyze and choose. Use this format:
<reasoning>
Your strategic analysis
</reasoning>
<action>
your_choice
</action>"""
        else:
            prompt += """

=== YOUR TASK ===
Analyze the situation and make your decision. Use the format:
<reasoning>
Your strategic analysis
</reasoning>
<action>
your_action_value
</action>"""

        try:
            logger.info(f"[{self.agent_id}] Requesting action from {self.model.model_id}")

            response = await self.llm.get_completion(
                prompt=prompt,
                model=self.model.model_id,
                system_prompt=effective_system_prompt,
                temperature=self.temperature
            )

            # Check for empty response
            if not response or not response.strip():
                logger.warning(f"[{self.agent_id}] EMPTY RESPONSE received from {self.model.model_id}")
                logger.warning(f"[{self.agent_id}] System prompt length: {len(effective_system_prompt)}, User prompt length: {len(prompt)}")
                # Return a fallback action
                fallback = available_actions[0] if available_actions else "pass"
                return AgentAction(
                    agent_id=self.agent_id,
                    action=fallback,
                    reasoning="[MODEL RETURNED EMPTY RESPONSE - Using fallback]",
                    raw_response="[EMPTY]"
                )

            logger.info(f"[{self.agent_id}] Received {len(response)} chars from {self.model.model_id}")

            # Extract reasoning and action from the response
            reasoning = self._extract_reasoning(response)
            action = self._extract_action_from_tags(response)

            # Warn if action extraction failed
            if not action:
                logger.warning(f"[{self.agent_id}] Failed to extract action from response. Response preview: {response[:200]}")
                action = available_actions[0] if available_actions else "pass"

            return AgentAction(
                agent_id=self.agent_id,
                action=action or response[:100],  # Fallback to truncated response
                reasoning=reasoning or response[:500],
                raw_response=response
            )

        except Exception as e:
            logger.error(f"[{self.agent_id}] Exception in decide_action: {str(e)}", exc_info=True)
            # Fallback action
            fallback = available_actions[0] if available_actions else "pass"
            return AgentAction(
                agent_id=self.agent_id,
                action=fallback,
                reasoning=f"Error: {str(e)}",
                raw_response=""
            )

    def _extract_reasoning(self, response: str) -> str:
        """Extract content from <reasoning> tags"""
        import re
        pattern = r'<reasoning>(.*?)</reasoning>'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # If no tags, return empty - the game will handle it
        return ""

    def _extract_action_from_tags(self, response: str) -> str:
        """Extract content from <action> tags"""
        import re
        pattern = r'<action>(.*?)</action>'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # If no tags found, return empty - let the game's extraction handle it
        return ""

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
