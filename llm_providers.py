"""
LLM Provider Integration
Supports multiple LLM backends (Groq, with extensibility for more)
"""

import os
import asyncio
from groq import Groq
from typing import List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class LLMModel:
    """Available LLM model configuration"""
    model_id: str
    display_name: str
    provider: str


# Available models (correct Groq model IDs)
GPT_OSS_120B = LLMModel("openai/gpt-oss-120b", "GPT-OSS 120B", "groq")
GPT_OSS_20B = LLMModel("openai/gpt-oss-20b", "GPT-OSS 20B", "groq")
LLAMA_3_1_8B = LLMModel("llama-3.1-8b-instant", "Llama 3.1 8B", "groq")
LLAMA_3_3_70B = LLMModel("llama-3.3-70b-versatile", "Llama 3.3 70B", "groq")
QWEN_3_32B = LLMModel("qwen/qwen3-32b", "Qwen 3 32B", "groq")
KIMI_K2 = LLMModel("moonshotai/kimi-k2-instruct-0905", "Kimi K2", "groq")


class GroqProvider:
    """Groq API provider"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        self.client = Groq(api_key=self.api_key)

    async def chat_completion(
        self,
        messages: List[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: dict = None
    ) -> str:
        """
        Get chat completion from Groq
        messages: List of {"role": "user|assistant|system", "content": "..."}
        """
        try:
            kwargs = {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if response_format:
                kwargs["response_format"] = response_format

            response = self.client.chat.completions.create(**kwargs)

            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"Groq API error: {str(e)}")


class LLMOrchestrator:
    """Manages multiple LLM providers and models"""

    def __init__(self):
        self.groq = GroqProvider()

    async def get_completion(
        self,
        prompt: str,
        model: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        json_mode: bool = False
    ) -> str:
        """Get completion from specified model"""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response_format = {"type": "json_object"} if json_mode else None

        return await self.groq.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            response_format=response_format
        )

    async def analyze_rules(self, rules: str) -> dict:
        """Analyze game rules and suggest configuration"""
        prompt = f"""Analyze these game rules and suggest optimal configuration:

{rules}

Provide JSON with:
{{
    "game_type": "what kind of game this is (e.g., prisoner's dilemma, auction)",
    "min_players": minimum number of players,
    "max_players": maximum number of players,
    "recommended_players": optimal number,
    "state_visibility": "full" | "partial" | "own_only",
    "action_format": "free_text" | "structured" | "multiple_choice",
    "typical_rounds": approximate number of rounds or game duration,
    "complexity": "low" | "medium" | "high",
    "suggested_models": ["list", "of", "suitable", "models"],
    "key_variables": ["var1", "var2"] // parameters user might want to tweak
}}

Return ONLY valid JSON."""

        try:
            response = await self.get_completion(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                json_mode=True
            )
            import json
            return json.loads(response)
        except Exception as e:
            # Fallback to defaults
            return {
                "game_type": "custom",
                "min_players": 2,
                "max_players": 8,
                "recommended_players": 4,
                "state_visibility": "full",
                "action_format": "free_text",
                "typical_rounds": 10,
                "complexity": "medium",
                "suggested_models": ["llama-3.3-70b-versatile"],
                "key_variables": []
            }
