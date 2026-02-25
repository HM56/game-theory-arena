"""
Sample Games Implementation
Classic game theory scenarios
"""

import time
import random
from typing import List, Dict, Any
from dataclasses import dataclass

from game_engine import BaseGame, GameEvent, AgentAction, GameStatus


class PrisonersDilemma(BaseGame):
    """
    Classic Prisoner's Dilemma
    Two players, each chooses to Cooperate or Defect
    """

    def __init__(self, rules: str = ""):
        super().__init__(rules or self._default_rules())
        self.payoff_matrix = {
            ("cooperate", "cooperate"): (3, 3),   # Both cooperate
            ("cooperate", "defect"): (0, 5),      # Sucker's payoff
            ("defect", "cooperate"): (5, 0),      # Temptation
            ("defect", "defect"): (1, 1),         # Both defect
        }
        self.max_rounds = 10
        self.scores = {}

    def _default_rules(self):
        return """Prisoner's Dilemma:
Two players simultaneously choose to COOPERATE or DEFECT.
Payoffs:
- Both cooperate: 3 points each
- One cooperates, one defects: defector gets 5, cooperator gets 0
- Both defect: 1 point each

Goal: Maximize your total score over multiple rounds."""

    def setup(self, agents: List):
        """Initialize game with agents"""
        if len(agents) != 2:
            raise ValueError("Prisoner's Dilemma requires exactly 2 players")

        self.agents = agents
        self.scores = {agent.agent_id: 0 for agent in agents}
        self.state = {"rounds_played": 0}

    def get_state_description(self, agent_id: str = None) -> str:
        current_round = self.round_number + 1

        desc = f"Round {current_round}/{self.max_rounds}\n"
        desc += f"Rounds remaining: {self.max_rounds - current_round}\n"
        desc += f"Your score: {self.scores.get(agent_id, 0)}\n"

        # Show opponent's score
        for other_id in self.scores:
            if other_id != agent_id:
                desc += f"Opponent ({other_id}) score: {self.scores[other_id]}\n"

        if self.round_number > 0:
            desc += "\nLast round actions:\n"
            for event in self.history[-2:]:
                if event.message:
                    desc += f"  {event.message}\n"

        return desc

    def validate_action(self, action: str) -> bool:
        return action.lower().strip() in ["cooperate", "defect", "coop", "def"]

    def process_actions(self, actions: List[AgentAction]) -> List[GameEvent]:
        events = []
        timestamp = time.time()

        # Normalize actions
        normalized = {}
        for action in actions:
            act = action.action.lower().strip()
            if act in ["coop", "cooperate"]:
                normalized[action.agent_id] = "cooperate"
            else:
                normalized[action.agent_id] = "defect"

        agent_ids = list(normalized.keys())
        action_a, action_b = normalized[agent_ids[0]], normalized[agent_ids[1]]

        # Calculate payoffs
        scores = self.payoff_matrix.get((action_a, action_b), (0, 0))
        self.scores[agent_ids[0]] += scores[0]
        self.scores[agent_ids[1]] += scores[1]

        # Create events
        events.append(GameEvent(
            event_type="action",
            timestamp=timestamp,
            agent_id=agent_ids[0],
            message=f"{agent_ids[0]} chose {action_a}",
            data={"action": action_a}
        ))

        events.append(GameEvent(
            event_type="action",
            timestamp=timestamp,
            agent_id=agent_ids[1],
            message=f"{agent_ids[1]} chose {action_b}",
            data={"action": action_b}
        ))

        events.append(GameEvent(
            event_type="result",
            timestamp=timestamp,
            message=f"Results: {agent_ids[0]} +{scores[0]}, {agent_ids[1]} +{scores[1]}",
            data={"scores": self.scores.copy()}
        ))

        self.history.extend(events)
        self.round_number += 1

        return events

    def is_game_over(self) -> bool:
        return self.round_number >= self.max_rounds

    def get_scores(self) -> Dict[str, float]:
        return self.scores.copy()


class PublicGoodsGame(BaseGame):
    """
    Public Goods Game
    N players decide how much to contribute to a public pot
    Total is multiplied and redistributed equally
    """

    def __init__(self, rules: str = ""):
        super().__init__(rules or self._default_rules())
        self.multiplier = 1.8
        self.endowment = 10
        self.max_rounds = 10
        self.scores = {}

    def _default_rules(self):
        return """Public Goods Game:
Each player starts with {endowment} points per round.
Choose how much to contribute (0-{endowment}) to a public pot.
Total contributions are multiplied by {multiplier}x and split equally.

Goal: Maximize your total score.""".format(
            endowment=self.endowment,
            multiplier=self.multiplier
        )

    def setup(self, agents: List):
        self.agents = agents
        self.scores = {agent.agent_id: self.endowment * self.max_rounds for agent in agents}
        self.state = {"rounds_played": 0}

    def get_state_description(self, agent_id: str = None) -> str:
        desc = f"Round {self.round_number + 1}/{self.max_rounds}\n"
        desc += f"Your current score: {self.scores.get(agent_id, 0)}\n"
        desc += f"You have {self.endowment} points to contribute (0-{self.endowment})\n"

        if self.round_number > 0:
            desc += "\nLast round summary:\n"
            # Show last round's contributions
            pass

        return desc

    def validate_action(self, action: str) -> bool:
        try:
            val = int(action.strip())
            return 0 <= val <= self.endowment
        except:
            return False

    def process_actions(self, actions: List[AgentAction]) -> List[GameEvent]:
        events = []
        timestamp = time.time()

        contributions = {}
        total_pot = 0

        for action in actions:
            try:
                contrib = min(max(int(action.action), 0), self.endowment)
            except:
                contrib = 0

            contributions[action.agent_id] = contrib
            total_pot += contrib

            events.append(GameEvent(
                event_type="action",
                timestamp=timestamp,
                agent_id=action.agent_id,
                message=f"{action.agent_id} contributed {contrib}",
                data={"contribution": contrib}
            ))

        # Multiply and redistribute
        redistributed = int(total_pot * self.multiplier / len(actions))
        for action in actions:
            old_score = self.scores[action.agent_id]
            contribution = contributions[action.agent_id]
            # They kept (endowment - contribution) and get redistributed share
            self.scores[action.agent_id] = self.scores[action.agent_id] - self.endowment + (self.endowment - contribution) + redistributed

        events.append(GameEvent(
            event_type="result",
            timestamp=timestamp,
            message=f"Pot: {total_pot} → {int(total_pot * self.multiplier)} after multiplier, each gets {redistributed}",
            data={
                "contributions": contributions,
                "total_pot": total_pot,
                "redistributed": redistributed,
                "scores": self.scores.copy()
            }
        ))

        self.history.extend(events)
        self.round_number += 1

        return events

    def is_game_over(self) -> bool:
        return self.round_number >= self.max_rounds

    def get_scores(self) -> Dict[str, float]:
        return self.scores.copy()


class CustomGame(BaseGame):
    """
    Flexible game that interprets rules dynamically
    Uses LLM to parse actions, extract moves, and adjudicate results
    """

    def __init__(self, rules: str, llm_orchestrator):
        super().__init__(rules)
        self.llm = llm_orchestrator
        self.scores = {}
        self.available_actions = []
        self.action_format = "free_text"
        self.action_description = ""  # What players should output

    async def analyze_rules(self):
        """Use LLM to understand the game structure - enhanced analysis"""
        prompt = f"""You are analyzing a game to be played by AI agents.

Carefully read these rules and extract ALL relevant information:

{self.rules}

Provide a comprehensive JSON analysis:
{{
    "game_type": "short name of this game type",
    "min_players": minimum players needed,
    "max_players": maximum players possible,
    "recommended_players": ideal number for this game,
    "typical_rounds": how many rounds make sense (be specific to rules if mentioned)",

    "action_format": "free_text" | "number" | "choice" | "bid",
    "action_description": "VERY CLEAR - what exactly should a player output? Be specific: 'a number from 1-100', 'cooperate or defect', 'a bid amount in dollars', 'rock/paper/scissors'",
    "example_action": "give a concrete example of a valid action a player might take",

    "scoring_method": "explain how points are awarded in detail",
    "win_condition": "how does someone win this game?",
    "strategy_hint": "what's the basic strategy to do well?",

    "complexity": "low" | "medium" | "high",
    "key_mechanics": ["list", "key", "game", "mechanics"],
    "player_interactions": "simultaneous" | "sequential" | "auction",

    "sample_round": "walk through what one round looks like with example actions and scores"
}}

Return ONLY valid JSON - no extra text."""

        try:
            response = await self.llm.get_completion(
                prompt=prompt,
                model="openai/gpt-oss-120b",  # Use best model for analysis
                json_mode=True
            )
            import json
            result = json.loads(response)
            self.state["analysis"] = result
            self.action_description = result.get("action_description", "your action")
            return result
        except Exception as e:
            # Fallback
            return {
                "game_type": "custom",
                "min_players": 2,
                "max_players": 8,
                "recommended_players": 2,
                "typical_rounds": 5,
                "action_description": "your move/action",
                "example_action": "your choice",
                "scoring_method": "determined by rules",
                "complexity": "medium"
            }

    def setup(self, agents: List, max_rounds: int = None):
        self.agents = agents
        self.scores = {agent.agent_id: 0 for agent in agents}

        analysis = self.state.get("analysis", {})
        self.max_rounds = max_rounds or analysis.get("typical_rounds", 10)
        self.action_description = analysis.get("action_description", "your action")
        self.state["rounds_played"] = 0

    def get_state_description(self, agent_id: str = None) -> str:
        current_round = self.round_number + 1
        is_final = current_round == self.max_rounds

        desc = f"=== ROUND {current_round}/{self.max_rounds}{' [FINAL ROUND]' if is_final else ''} ===\n\n"

        desc += f"CURRENT STANDINGS:\n"
        sorted_scores = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (aid, score) in enumerate(sorted_scores, 1):
            leader = "[LEADER] " if rank == 1 else ""
            desc += f"  {leader}{aid}: {score} points\n"
            if aid == agent_id:
                desc += f"    ^ YOU\n"

        if self.round_number > 0 and self.history:
            desc += f"\nLAST ROUND RESULT:\n"
            # Find last result event
            for event in reversed(self.history[-5:]):
                if event.event_type == "result":
                    desc += f"  {event.message}\n"
                    break

        desc += f"\nRULES:\n"
        # Show rules in a more readable way
        rules_lines = self.rules.split('\n')
        for line in rules_lines[:15]:  # First 15 lines
            if line.strip():
                desc += f"  {line}\n"
        if len(rules_lines) > 15:
            desc += f"  ... (rules continue)\n"

        if is_final:
            desc += f"\n[FINAL ROUND] - Make it count!\n"

        return desc

    def get_system_prompt(self) -> str:
        """Generate dynamic system prompt based on rules - enhanced clarity"""
        analysis = self.state.get("analysis", {})
        action_desc = analysis.get("action_description", "your action")
        example = analysis.get("example_action", "")
        strategy = analysis.get("strategy_hint", "")

        prompt = f"""You are a rational agent playing a game against opponents.

OBJECTIVE: Maximize your total score.

YOUR ACTION: {action_desc}
{f'Example valid action: {example}' if example else ''}

{f'STRATEGY HINT: {strategy}' if strategy else ''}

CRITICAL - How to respond:
1. Think through the game state first
2. Consider what opponents might do
3. Make your decision strategically
4. Output your action CLEARLY at the end in this exact format:

   ACTION: [your action here]

Remember: Your goal is to WIN by getting the highest score.
Always end your response with "ACTION: [your choice]" on its own line."""

        return prompt

    def validate_action(self, action: str) -> bool:
        return len(action.strip()) > 0

    async def extract_action(self, agent_id: str, raw_response: str) -> str:
        """Use LLM to extract the actual action from agent's response - enhanced with Opus"""

        prompt = f"""You are extracting a player's action from their response.

AGENT: {agent_id}

WHAT THEY SHOULD OUTPUT: {self.action_description}

THEIR FULL RESPONSE:
\"\"\"
{raw_response}
\"\"\"

EXTRACTION RULES:
1. Find the FINAL, CLEAR decision/action
2. Look for patterns like "I choose X", "action: X", "decision: X", "→ X", "=> X"
3. If it's a NUMBER game: extract digits only
4. If it's a CHOICE game (cooperate/defect, rock/paper/scissors): extract the choice word
5. Ignore all reasoning - ONLY extract the action itself
6. If multiple decisions mentioned, take the LAST one (it's the final choice)

Examples:
- "I think... therefore I choose to COOPERATE" → "COOPERATE"
- "Action: 42" → "42"
- "My decision is rock" → "rock"
- "I will defect. Final answer." → "defect"

Extract the action now - respond with ONLY the action, nothing else:"""

        try:
            # Use Opus for best extraction accuracy
            response = await self.llm.get_completion(
                prompt=prompt,
                model="openai/gpt-oss-120b",  # Best model for extraction
                temperature=0.05,  # Lower temperature for more consistent extraction
                max_tokens=50
            )
            extracted = response.strip()

            # Clean up common issues
            extracted = extracted.strip('\'"`.').strip()

            # If the response is quoted, extract the content
            if (extracted.startswith('"') and extracted.endswith('"')) or \
               (extracted.startswith("'") and extracted.endswith("'")):
                extracted = extracted[1:-1].strip()

            # Validate extraction makes sense
            if len(extracted) == 0 or len(extracted) > 100:
                raise ValueError("Extraction failed - invalid length")

            return extracted

        except Exception as e:
            # Enhanced smart fallback with multiple strategies
            import re

            # Strategy 1: Look for explicit decision markers with colons/arrow syntax
            decision_patterns = [
                r'(?:final decision|final choice|i choose|i will|i pick|my action|decision|action|choice|therefore)\s*[:→=]\s*["\']*([^"\n\'\.]+)["\']*\.?',
                r'(?:final decision|final choice|i choose|i will|i pick|my action|decision|action|choice)\s+(?:to\s+)?["\']*([^"\n\'\.]+)["\']*\.?',
                r'(?:therefore|so|i decide|i decided)\s*(?:,)?\s*(?:i\s+)?["\']*([^"\n\'\.]+)["\']*\.?'
            ]

            for pattern in decision_patterns:
                matches = list(re.finditer(pattern, raw_response, re.IGNORECASE | re.MULTILINE))
                if matches:
                    # Take the last match (final decision)
                    action = matches[-1].group(1).strip()
                    if len(action) > 0 and len(action) < 50:
                        # Normalize the action
                        action = action.rstrip('\'".').strip()
                        # Check if it's a well-known action
                        known_actions = {
                            'cooperate': 'cooperate',
                            'coop': 'cooperate',
                            'defect': 'defect',
                            'def': 'defect',
                            'rock': 'rock',
                            'paper': 'paper',
                            'scissors': 'scissors'
                        }
                        action_lower = action.lower()
                        if action_lower in known_actions:
                            return known_actions[action_lower]
                        # If it's a number, return it
                        if action.isdigit():
                            return action
                        return action.capitalize()

            # Strategy 2: Look for standalone numbers (for bid/number games)
            numbers = re.findall(r'^\s*(\d+(?:\.\d+)?)\s*\.?\s*$', raw_response, re.MULTILINE)
            if numbers:
                return numbers[-1]  # Last standalone number is often the final choice

            # Strategy 3: Look for the last line with a clear decision word
            lines = raw_response.strip().split('\n')
            action_words = {'cooperate', 'coop', 'defect', 'def', 'rock', 'paper', 'scissors', 'bid', 'pass'}

            for line in reversed(lines):
                line_clean = line.strip().rstrip('\'".').strip()
                line_lower = line_clean.lower()

                # Check if line contains a known action word
                words = line_lower.split()
                for word in reversed(words):  # Check last word first
                    word_clean = word.rstrip('\'".,').strip()
                    if word_clean in action_words:
                        # Map short forms to full
                        if word_clean in ['coop', 'cooperate']:
                            return 'cooperate'
                        elif word_clean in ['def', 'defect']:
                            return 'defect'
                        return word_clean

                # If line is short and looks like a decision
                if len(line_clean) > 0 and len(line_clean) < 60:
                    # Check for common decision patterns
                    if any(kw in line_lower for kw in ['i choose', 'i will', 'action:', '→', '=>']):
                        # Extract the action part
                        for kw in ['i choose', 'i will', 'i pick', 'action:', 'decision:', '→', '=>', ':']:
                            if kw in line_lower:
                                parts = line_lower.split(kw, 1)
                                if len(parts) > 1:
                                    action = parts[1].strip().strip('\'".,')
                                    if len(action) > 0 and len(action) < 50:
                                        # Normalize
                                        if action in ['coop', 'cooperate']:
                                            return 'Cooperate'
                                        elif action in ['def', 'defect']:
                                            return 'Defect'
                                        return action.capitalize()

            # Strategy 4: Last resort - get the last meaningful line
            for line in reversed(lines):
                line_clean = line.strip().strip('*-=').strip()
                # Remove common prefixes
                for prefix in ['i choose', 'i will', 'i pick', 'action:', 'decision:', 'final:', 'therefore', 'so']:
                    if line_clean.lower().startswith(prefix):
                        line_clean = line_clean[len(prefix):].strip(',\'".')

                if 2 < len(line_clean) < 50:
                    # Check if it's not just reasoning
                    reasoning_words = ['because', 'since', 'due to', 'considering', 'thinking', 'strategy']
                    if not any(rw in line_clean.lower() for rw in reasoning_words):
                        return line_clean.capitalize()

            # Final fallback - return first 50 chars of last non-empty line
            for line in reversed(lines):
                if line.strip():
                    return line.strip()[:50].capitalize()

            return raw_response.strip()[:50]

    async def process_actions(self, actions: List[AgentAction]) -> List[GameEvent]:
        events = []
        timestamp = time.time()

        # Extract clean actions from raw responses
        extracted_actions = {}
        for action in actions:
            clean_action = await self.extract_action(action.agent_id, action.raw_response)
            extracted_actions[action.agent_id] = clean_action

            events.append(GameEvent(
                event_type="action",
                timestamp=timestamp,
                agent_id=action.agent_id,
                message=f"{action.agent_id}: {clean_action}",
                data={
                    "action": clean_action,
                    "raw_response": action.raw_response[:500],
                    "reasoning": action.reasoning[:300] if action.reasoning else ""
                }
            ))

        # Use LLM to adjudicate results
        actions_summary = "\n".join([
            f"- {aid}: {extracted_actions[aid]}" for aid in extracted_actions
        ])

        # Get previous scores for display
        previous_scores = self.scores.copy()

        adjudication_prompt = f"""You are the game adjudicator. Calculate scores accurately.

====================
GAME RULES:
====================
{self.rules}

====================
ACTIONS THIS ROUND:
====================
{actions_summary}

====================
CURRENT STANDINGS (before this round):
====================
{chr(10).join([f'{aid}: {score} points' for aid, score in self.scores.items()])}

====================
YOUR TASK:
====================
Calculate what happens this round. Then respond with JSON:

{{
    "result_message": "Clear explanation: what happened, who gained/lost what, and why",
    "score_changes": {{"exact_agent_id_from_above": points_change (can be negative or positive)}},
    "new_scores": {{"exact_agent_id_from_above": new_total_score}},
    "game_over": false
}}

====================
CRITICAL RULES:
====================
1. Use EXACT agent IDs from the "ACTIONS THIS ROUND" section above
2. Calculate changes carefully - scores can go negative
3. If rules mention ties/same choices, handle them correctly
4. Be FAIR and follow the rules exactly
5. Make result_message VERY clear for players to understand
6. new_scores should be previous_score + score_change for each agent

Think through the scoring step by step before outputting JSON."""

        try:
            response = await self.llm.get_completion(
                prompt=adjudication_prompt,
                model="openai/gpt-oss-120b",  # Best model for adjudication
                json_mode=True
            )

            import json
            result = json.loads(response)

            # Apply score changes with better error handling
            score_updates = result.get("score_changes", {})
            new_scores = result.get("new_scores", {})

            # First try to use new_scores if provided (more reliable)
            if new_scores:
                for agent_id, new_score in new_scores.items():
                    # Try to match agent ID flexibly
                    matched_id = None
                    for aid in self.scores.keys():
                        if aid.lower() in agent_id.lower() or agent_id.lower() in aid.lower():
                            matched_id = aid
                            break

                    if matched_id:
                        try:
                            self.scores[matched_id] = float(new_score)
                        except:
                            pass

            # Then apply any score_changes as backup or for agents not in new_scores
            for agent_id, change in score_updates.items():
                # Skip if we already updated this agent via new_scores
                already_updated = False
                for aid in self.scores.keys():
                    if aid.lower() in agent_id.lower() or agent_id.lower() in aid.lower():
                        if new_scores and any(aid.lower() in nid.lower() for nid in new_scores.keys()):
                            already_updated = True
                            break

                if already_updated:
                    continue

                # Try to match agent ID flexibly
                matched_id = None
                for aid in self.scores.keys():
                    if aid.lower() in agent_id.lower() or agent_id.lower() in aid.lower():
                        matched_id = aid
                        break

                if matched_id:
                    try:
                        self.scores[matched_id] += float(change)
                    except:
                        pass
                else:
                    # Try using the key directly
                    if agent_id in self.scores:
                        try:
                            self.scores[agent_id] += float(change)
                        except:
                            pass

            # Generate result message with score changes for clarity
            result_msg = result.get("result_message", "Scores updated")

            # Add score change details to result message
            changes_detail = []
            for aid in self.scores.keys():
                if aid in previous_scores:
                    old = previous_scores[aid]
                    new = self.scores[aid]
                    diff = new - old
                    if diff != 0:
                        changes_detail.append(f"{aid}: {old} → {new} ({diff:+.0f})")

            if changes_detail:
                result_msg += f" Score changes: {', '.join(changes_detail)}."

            events.append(GameEvent(
                event_type="result",
                timestamp=timestamp,
                message=result_msg,
                data={
                    "scores": self.scores.copy(),
                    "extracted_actions": extracted_actions,
                    "previous_scores": previous_scores
                }
            ))

        except Exception as e:
            # Fallback: simple adjudication
            error_msg = f"Adjudication error: {str(e)}. Using fallback scoring."

            # Simple fallback: everyone gets 1 point
            for aid in self.scores.keys():
                self.scores[aid] += 1

            events.append(GameEvent(
                event_type="result",
                timestamp=timestamp,
                message=f"Rules applied. {', '.join([f'{aid}: {extracted_actions[aid]}' for aid in extracted_actions])}",
                data={
                    "scores": self.scores.copy(),
                    "extracted_actions": extracted_actions,
                    "error": error_msg
                }
            ))

        self.history.extend(events)
        self.round_number += 1

        return events

    def is_game_over(self) -> bool:
        if self.history and self.history[-1].data.get("game_over"):
            return True
        return self.round_number >= self.max_rounds

    def get_scores(self) -> Dict[str, float]:
        return self.scores.copy()

    def to_dict(self):
        """Override to include analysis data"""
        base_dict = super().to_dict()
        base_dict["analysis"] = self.state.get("analysis", {})
        base_dict["action_description"] = self.action_description
        return base_dict
