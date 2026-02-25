"""
Game Theory Arena - Core Game Engine
Abstract base classes for games and agents
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import json


class GameStatus(Enum):
    SETUP = "setup"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"


@dataclass
class GameEvent:
    """Event that happens during gameplay"""
    event_type: str  # "action", "result", "error", "game_end"
    timestamp: float
    agent_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self):
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "data": self.data,
            "message": self.message
        }


@dataclass
class AgentAction:
    """Action taken by an agent"""
    agent_id: str
    action: str
    reasoning: str = ""
    raw_response: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseGame(ABC):
    """Abstract base class for all games"""

    def __init__(self, rules: str = ""):
        self.rules = rules
        self.status = GameStatus.SETUP
        self.state: Dict[str, Any] = {}
        self.history: List[GameEvent] = []
        self.round_number = 0

    @abstractmethod
    def get_state_description(self, agent_id: str = None) -> str:
        """Get current state as text description for an agent"""
        pass

    @abstractmethod
    def validate_action(self, action: str) -> bool:
        """Check if action is valid"""
        pass

    @abstractmethod
    def process_actions(self, actions: List[AgentAction]) -> List[GameEvent]:
        """Process all actions and return resulting events"""
        pass

    @abstractmethod
    def is_game_over(self) -> bool:
        """Check if game has ended"""
        pass

    @abstractmethod
    def get_scores(self) -> Dict[str, float]:
        """Get current scores for all agents"""
        pass

    def get_full_history(self) -> str:
        """Get full game history as text"""
        lines = [f"Round {self.round_number}"]
        for event in self.history:
            if event.message:
                lines.append(f"  {event.message}")
        return "\n".join(lines)

    def to_dict(self):
        return {
            "status": self.status.value,
            "state": self.state,
            "round_number": self.round_number,
            "rules": self.rules,
            "scores": self.get_scores()
        }


class BaseAgent(ABC):
    """Abstract base class for all agents"""

    def __init__(self, agent_id: str, model_name: str, system_prompt: str = ""):
        self.agent_id = agent_id
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.score = 0.0
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    async def decide_action(self, game_state: str, available_actions: List[str]) -> AgentAction:
        """Decide and return an action based on game state"""
        pass

    def to_dict(self):
        return {
            "agent_id": self.agent_id,
            "model_name": self.model_name,
            "score": self.score,
            "metadata": self.metadata
        }
