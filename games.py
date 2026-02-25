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
        """Clear state description for Prisoner's Dilemma"""
        current_round = self.round_number + 1
        is_final = current_round == self.max_rounds

        desc = f"{'='*40}\n"
        desc += f"PRISONER'S DILEMMA - Round {current_round}/{self.max_rounds}{' [FINAL]' if is_final else ''}\n"
        desc += f"{'='*40}\n\n"

        # Your score
        desc += f"YOUR SCORE: {self.scores.get(agent_id, 0)} points\n"

        # Opponent's score
        for other_id in self.scores:
            if other_id != agent_id:
                opp_score = self.scores[other_id]
                diff = self.scores.get(agent_id, 0) - opp_score
                diff_str = f" ({diff:+d})" if diff != 0 else ""
                desc += f"OPPONENT ({other_id}): {opp_score} points{diff_str}\n"

        # Show history
        if self.round_number > 0:
            desc += "\nRECENT HISTORY:\n"
            for event in self.history[-4:]:
                if event.message:
                    desc += f"  {event.message}\n"

        # Payoff reminder
        desc += "\nPAYOFFS (You, Opponent):\n"
        desc += "  Both Cooperate: +3, +3\n"
        desc += "  You Cooperate, Opponent Defects: +0, +5\n"
        desc += "  You Defect, Opponent Cooperates: +5, +0\n"
        desc += "  Both Defect: +1, +1\n"

        if is_final:
            desc += "\n⚠️  FINAL ROUND - No more cooperation!\n"

        return desc

    def get_system_prompt(self) -> str:
        """Get system prompt for Prisoner's Dilemma agents"""
        return """You are a rational agent playing the Prisoner's Dilemma.

OBJECTIVE: Maximize your total score over multiple rounds.

GAME MECHANICS:
- Each round, you and your opponent simultaneously choose: COOPERATE or DEFECT
- Payoffs per round:
  • Both Cooperate: +3 points each
  • You Cooperate, Opponent Defects: +0 for you, +5 for opponent
  • You Defect, Opponent Cooperates: +5 for you, +0 for opponent
  • Both Defect: +1 point each

STRATEGIC CONSIDERATIONS:
- The game is repeated - your opponent will remember your past actions
- Defection dominates in a single round, but cooperation can be better long-term
- Consider tit-for-tat strategies, forgiveness, and reputation

OUTPUT FORMAT:
<reasoning>
Analyze the situation, consider opponent patterns, weigh risks
</reasoning>
<action>
cooperate or defect
</action>"""

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
        # Set these BEFORE calling super().__init__() which calls _default_rules()
        self.multiplier = 1.8
        self.endowment = 10
        self.max_rounds = 10
        self.scores = {}
        super().__init__(rules or self._default_rules())

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
        """Clear state description for Public Goods Game"""
        current_round = self.round_number + 1
        is_final = current_round == self.max_rounds

        desc = f"{'='*40}\n"
        desc += f"PUBLIC GOODS GAME - Round {current_round}/{self.max_rounds}{' [FINAL]' if is_final else ''}\n"
        desc += f"{'='*40}\n\n"

        # Current score
        desc = f"YOUR SCORE: {self.scores.get(agent_id, 0)} points\n"

        # All player scores
        desc += "\nALL SCORES:\n"
        sorted_scores = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        for aid, score in sorted_scores:
            marker = " ← YOU" if aid == agent_id else ""
            desc += f"  {aid}: {score}{marker}\n"

        # Game mechanics reminder
        desc += f"\nTHIS ROUND:\n"
        desc += f"  You have {self.endowment} points to contribute\n"
        desc += f"  Contribution range: 0 to {self.endowment}\n"
        desc += f"  Multiplier: {self.multiplier}x\n"
        desc += f"  Players: {len(self.agents)}\n"

        # Last round results
        if self.round_number > 0:
            desc += "\nLAST ROUND:\n"
            for event in self.history[-4:]:
                if event.message:
                    desc += f"  {event.message}\n"

        return desc

    def get_system_prompt(self) -> str:
        """Get system prompt for Public Goods Game agents"""
        return f"""You are a rational agent playing the Public Goods Game.

OBJECTIVE: Maximize your total score over multiple rounds.

GAME MECHANICS:
- Each round, you have {self.endowment} points to contribute to a public pot
- You choose how much to contribute (0 to {self.endowment})
- All contributions are pooled and multiplied by {self.multiplier}x
- The multiplied pot is divided EQUALLY among all players
- You keep (endowment - contribution) + your share of the pot

STRATEGIC CONSIDERATIONS:
- Full contribution by everyone maximizes total group welfare
- But individual incentive is to free-ride (contribute 0)
- Repeated rounds allow for building trust/reciprocity
- Consider conditional cooperation strategies

OUTPUT FORMAT:
<reasoning>
Analyze contributions, consider others' behavior, decide optimal contribution
</reasoning>
<action>
a number from 0 to {self.endowment}
</action>"""

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
                model="openai/gpt-oss-120b",
                temperature=0.1,  # Low temperature for consistent analysis
                json_mode=True
            )
            import json
            result = json.loads(response)
            self.state["analysis"] = result
            self.action_description = result.get("action_description", "your action")
            return result
        except Exception as e:
            # Enhanced fallback
            return {
                "game_type": "custom",
                "min_players": 2,
                "max_players": 8,
                "recommended_players": 2,
                "typical_rounds": 10,
                "action_format": "free_text",
                "action_description": "your move or action",
                "example_action": "your choice",
                "valid_actions": None,
                "scoring_method": "as specified in rules",
                "win_condition": "highest score",
                "strategy_hint": "play to maximize your score",
                "complexity": "medium",
                "player_interactions": "simultaneous",
                "key_mechanics": ["custom_rules"]
            }

    def setup(self, agents: List, max_rounds: int = None):
        self.agents = agents
        self.scores = {agent.agent_id: 0 for agent in agents}

        analysis = self.state.get("analysis", {})
        self.max_rounds = max_rounds or analysis.get("typical_rounds", 10)
        self.action_description = analysis.get("action_description", "your action")
        self.state["rounds_played"] = 0

    def get_state_description(self, agent_id: str = None) -> str:
        """Generate clear, structured game state description"""
        current_round = self.round_number + 1
        is_final = current_round == self.max_rounds

        desc = f"{'='*50}\n"
        desc += f"ROUND {current_round}/{self.max_rounds}{' [FINAL ROUND]' if is_final else ''}\n"
        desc += f"{'='*50}\n\n"

        # Current standings with clear hierarchy
        desc += "CURRENT STANDINGS:\n"
        sorted_scores = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (aid, score) in enumerate(sorted_scores, 1):
            icon = "👑" if rank == 1 else "  "
            you_marker = " ← YOU" if aid == agent_id else ""
            desc += f"  {icon} {aid}: {score} points{you_marker}\n"

        # Last round results if available
        if self.round_number > 0 and self.history:
            desc += "\nLAST ROUND:\n"
            # Get last few action events and the result
            for event in reversed(self.history[-5:]):
                if event.event_type == "action":
                    desc += f"  • {event.message}\n"
                elif event.event_type == "result":
                    desc += f"  → {event.message}\n"
                    break

        # Rules summary (cleaned up)
        desc += "\nGAME RULES:\n"
        rules_lines = self.rules.split('\n')
        for line in rules_lines[:20]:  # Show first 20 lines
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                desc += f"  {line}\n"
        if len(rules_lines) > 20:
            desc += "  ...\n"

        if is_final:
            desc += "\n⚠️  FINAL ROUND - Give it your all!\n"

        return desc

    def get_system_prompt(self) -> str:
        """Generate clear, structured system prompt with XML-style output format"""
        analysis = self.state.get("analysis", {})
        action_desc = analysis.get("action_description", "your action")
        example = analysis.get("example_action", "")
        strategy = analysis.get("strategy_hint", "")
        valid_actions = analysis.get("valid_actions")

        prompt = """You are a rational AI agent in a game theory competition.

PRIMARY OBJECTIVE: Maximize your total score by the end of the game.

OUTPUT FORMAT (CRITICAL):
You MUST structure your response using XML-style tags:

<reasoning>
Your strategic analysis goes here. Consider:
• Current game state and standings
• Opponent behavior patterns
• Risk vs reward of different actions
• Long-term implications
</reasoning>

<action>
Your final action value ONLY - no explanation
</action>"""

        # Add action-specific guidance
        prompt += f"\n\nYOUR ACTION THIS TURN: {action_desc}"
        if example:
            prompt += f"\nExample: <action>{example}</action>"
        if valid_actions:
            prompt += f"\nValid choices: {', '.join(valid_actions)}"

        # Add strategic hint if available
        if strategy:
            prompt += f"\n\nSTRATEGIC GUIDANCE:\n{strategy}"

        prompt += """

REMEMBER:
- The <action> tag must contain ONLY your action value
- No explanations like "I choose" or "my action is"
- Just the raw action value: "cooperate", "5", "rock", etc.

Your goal is to WIN by achieving the highest score. Think strategically in the <reasoning> section, then output your action in the <action> section."""

        return prompt

    def validate_action(self, action: str) -> bool:
        return len(action.strip()) > 0

    async def extract_action(self, agent_id: str, raw_response: str) -> str:
        """Extract the actual action from agent's response - improved with better prompts"""

        # First, try to extract from XML-style tags (new format)
        import re

        # Look for <action>content</action> tags first
        tag_pattern = r'<action>(.*?)</action>'
        tag_match = re.search(tag_pattern, raw_response, re.DOTALL | re.IGNORECASE)
        if tag_match:
            extracted = tag_match.group(1).strip()
            # Clean up the extracted action
            extracted = extracted.strip('\'"`.').strip()
            if extracted and len(extracted) > 0 and len(extracted) < 100:
                return extracted

        # If no tags found, use LLM-based extraction with improved prompt
        prompt = f"""You are an action extraction system. Your job is to extract ONLY the player's final action from their response.

AGENT: {agent_id}

WHAT THEY SHOULD OUTPUT: {self.action_description}

THEIR FULL RESPONSE:
\"\"\"
{raw_response}
\"\"\"

EXTRACTION PROTOCOL:
1. Find the FINAL, CLEAR decision/action - ignore all reasoning
2. Look for the action at the end of the response (after analysis)
3. Common patterns: "ACTION: X", "I choose X", "My decision: X", "→ X", "=> X"
4. For NUMBER games: extract just the digits
5. For CHOICE games (cooperate/defect, rock/paper/scissors): extract the choice word

IMPORTANT: Respond with ONLY the extracted action - no explanation, no quotes, just the raw action value.

Extract now:"""

        try:
            response = await self.llm.get_completion(
                prompt=prompt,
                model="openai/gpt-oss-120b",
                temperature=0.0,  # Zero temperature for consistent extraction
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
            if 0 < len(extracted) <= 100:
                return extracted

        except Exception as e:
            pass  # Fall through to regex extraction

        # Enhanced regex-based fallback
        # Strategy 1: Look for explicit decision markers with colons/arrows
        decision_patterns = [
            r'(?:final decision|final choice|i choose|i will|i pick|my action|decision|action|choice|therefore)\s*[:→=]\s*["\']*?([^"\n\'\.]+?)["\']*\.?$',
            r'(?:final decision|final choice|i choose|i will|i pick)\s+(?:to\s+)?["\']*?([^"\n\'\.]+?)["\']*\.?$',
        ]

        for pattern in decision_patterns:
            matches = list(re.finditer(pattern, raw_response, re.IGNORECASE | re.MULTILINE))
            if matches:
                action = matches[-1].group(1).strip()
                if 0 < len(action) < 50:
                    action = action.rstrip('\'".').strip()
                    # Normalize known actions
                    known_actions = {
                        'cooperate': 'cooperate', 'coop': 'cooperate',
                        'defect': 'defect', 'def': 'defect',
                        'rock': 'rock', 'paper': 'paper', 'scissors': 'scissors'
                    }
                    action_lower = action.lower()
                    if action_lower in known_actions:
                        return known_actions[action_lower]
                    if action.isdigit():
                        return action
                    return action.capitalize()

        # Strategy 2: Look for standalone numbers
        numbers = re.findall(r'^\s*(\d+(?:\.\d+)?)\s*\.?\s*$', raw_response, re.MULTILINE)
        if numbers:
            return numbers[-1]

        # Strategy 3: Look for action keywords in the last few lines
        lines = raw_response.strip().split('\n')
        action_words = {'cooperate', 'coop', 'defect', 'def', 'rock', 'paper', 'scissors', 'bid', 'pass'}

        for line in reversed(lines[-5:]):  # Check last 5 lines only
            line_clean = line.strip().rstrip('\'".').strip()
            line_lower = line_clean.lower()

            # Check for action words
            for word in reversed(line_lower.split()):
                word_clean = word.rstrip('\'".,').strip()
                if word_clean in action_words:
                    if word_clean in ['coop', 'cooperate']:
                        return 'cooperate'
                    elif word_clean in ['def', 'defect']:
                        return 'defect'
                    return word_clean

            # Check for decision patterns
            if any(kw in line_lower for kw in ['i choose', 'i will', 'action:', '→', '=>']):
                for kw in ['action:', 'decision:', '→', '=>']:
                    if kw in line_lower:
                        parts = line_lower.split(kw, 1)
                        if len(parts) > 1:
                            action = parts[1].strip().strip('\'".,')
                            if 0 < len(action) < 50:
                                return action.capitalize()

        # Strategy 4: Get the last meaningful short line
        for line in reversed(lines):
            line_clean = line.strip().strip('*-=#').strip()
            # Remove common prefixes
            for prefix in ['i choose', 'i will', 'i pick', 'action:', 'decision:', 'final:', 'therefore', 'so']:
                if line_clean.lower().startswith(prefix):
                    line_clean = line_clean[len(prefix):].strip(',\'".')

            if 2 < len(line_clean) < 50:
                reasoning_words = ['because', 'since', 'due to', 'considering', 'thinking', 'strategy', 'analyze']
                if not any(rw in line_clean.lower() for rw in reasoning_words):
                    return line_clean.capitalize()

        # Final fallback
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

        # Use improved LLM adjudication
        actions_summary = "\n".join([
            f"- {aid}: {extracted_actions[aid]}" for aid in extracted_actions
        ])

        previous_scores = self.scores.copy()

        # Improved adjudication prompt - more structured and unbiased
        adjudication_prompt = f"""You are an impartial game adjudicator. Your role is to apply game rules mechanically and fairly.

====================
GAME RULES (Apply These Exactly):
====================
{self.rules}

====================
PLAYER ACTIONS THIS ROUND:
====================
{actions_summary}

====================
CURRENT SCORES (Before This Round):
====================
{chr(10).join([f'Player {aid}: {score} points' for aid, score in self.scores.items()])}

====================
ADJUDICATION PROTOCOL:
====================

Step 1: Interpret the rules literally - do not add or infer anything
Step 2: Apply the rules to the actions taken
Step 3: Calculate score changes for each player
Step 4: Verify calculations are mathematically correct

CRITICAL CONSTRAINTS:
- Apply rules EXACTLY as written - no interpretation bias
- Treat all players equally - no favoritism
- Scores can be negative if rules permit
- Use the EXACT player IDs listed above

====================
REQUIRED OUTPUT FORMAT:
====================
Respond ONLY with valid JSON in this exact format:

{{
    "result_summary": "Brief factual statement of what occurred",
    "score_adjustments": {{
        "player_id_1": change_value,
        "player_id_2": change_value
    }},
    "updated_scores": {{
        "player_id_1": new_total,
        "player_id_2": new_total
    }},
    "calculations": "Brief explanation of how you calculated each score change"
}}

EXAMPLE:
If Player A had 10 points, gains 3, and Player B had 8 points, loses 1:
{{
    "result_summary": "Player A and B both chose Cooperate. Each receives 3 points.",
    "score_adjustments": {{"player_id": 3, "player_id_2": 3}},
    "updated_scores": {{"player_id": 13, "player_id_2": 11}},
    "calculations": "From rules: Both cooperate = +3 each. 10+3=13, 8+3=11"
}}

Apply the rules now:"""

        try:
            # Use lower temperature for more consistent, unbiased adjudication
            response = await self.llm.get_completion(
                prompt=adjudication_prompt,
                model="openai/gpt-oss-120b",
                temperature=0.1,  # Lower temp for more deterministic results
                json_mode=True
            )

            import json
            result = json.loads(response)

            # Apply score updates with careful matching
            score_adjustments = result.get("score_adjustments", {})
            updated_scores = result.get("updated_scores", {})

            # Prefer updated_scores if provided (more reliable)
            if updated_scores:
                for agent_id, new_score in updated_scores.items():
                    matched_id = self._match_agent_id(agent_id)
                    if matched_id:
                        try:
                            self.scores[matched_id] = float(new_score)
                        except (ValueError, TypeError):
                            pass

            # Apply any adjustments not covered by updated_scores
            for agent_id, change in score_adjustments.items():
                matched_id = self._match_agent_id(agent_id)
                if matched_id and matched_id not in updated_scores:
                    try:
                        self.scores[matched_id] += float(change)
                    except (ValueError, TypeError):
                        pass

            # Build result message
            result_msg = result.get("result_summary", "Round processed")

            # Add score change details
            changes_detail = []
            for aid in self.scores.keys():
                if aid in previous_scores:
                    old = previous_scores[aid]
                    new = self.scores[aid]
                    diff = new - old
                    symbol = "±" if diff == 0 else ("+" if diff > 0 else "")
                    changes_detail.append(f"{aid}: {old}→{new} ({symbol}{diff:.0f})")

            if changes_detail:
                result_msg += f" | Changes: {', '.join(changes_detail)}"

            events.append(GameEvent(
                event_type="result",
                timestamp=timestamp,
                message=result_msg,
                data={
                    "scores": self.scores.copy(),
                    "extracted_actions": extracted_actions,
                    "previous_scores": previous_scores,
                    "calculations": result.get("calculations", "")
                }
            ))

        except Exception as e:
            # Enhanced fallback with better error handling
            import json

            # Try a second adjudication attempt with simpler prompt
            fallback_prompt = f"""Apply these rules:

{self.rules}

Actions: {actions_summary}
Current scores: {self.scores}

Give me JSON with score changes only:
{{"player_id": change_number}}"""

            try:
                fallback_response = await self.llm.get_completion(
                    prompt=fallback_prompt,
                    model="llama-3.3-70b-versatile",
                    temperature=0.0,
                    json_mode=True
                )
                fallback_result = json.loads(fallback_response)

                for agent_id, change in fallback_result.items():
                    matched_id = self._match_agent_id(agent_id)
                    if matched_id:
                        try:
                            self.scores[matched_id] += float(change)
                        except (ValueError, TypeError):
                            pass

                events.append(GameEvent(
                    event_type="result",
                    timestamp=timestamp,
                    message=f"Round adjudicated. Actions: {', '.join([f'{aid}: {extracted_actions[aid]}' for aid in extracted_actions])}",
                    data={"scores": self.scores.copy(), "extracted_actions": extracted_actions}
                ))

            except:
                # Ultimate fallback - no score changes, just log actions
                events.append(GameEvent(
                    event_type="result",
                    timestamp=timestamp,
                    message=f"Round processed (scoring unavailable). Actions: {', '.join([f'{aid}: {extracted_actions[aid]}' for aid in extracted_actions])}",
                    data={
                        "scores": self.scores.copy(),
                        "extracted_actions": extracted_actions,
                        "error": str(e)
                    }
                ))

        self.history.extend(events)
        self.round_number += 1

        return events

    def _match_agent_id(self, query_id: str) -> str:
        """Flexibly match agent IDs, handling case and partial matches"""
        query_lower = query_id.lower().strip()

        # Direct match
        if query_id in self.scores:
            return query_id

        # Case-insensitive match
        for aid in self.scores:
            if aid.lower() == query_lower:
                return aid

        # Partial match (query contains actual ID or vice versa)
        for aid in self.scores:
            aid_lower = aid.lower()
            if query_lower in aid_lower or aid_lower in query_lower:
                return aid

        # Extract ID from patterns like "player llama-3.3-70b-1"
        for aid in self.scores:
            if aid.lower() in query_lower:
                return aid

        return None

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
