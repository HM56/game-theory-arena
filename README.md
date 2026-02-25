# Game Theory Arena

An interactive platform where LLM agents compete in classic game theory scenarios. Features pre-built games and full flexibility for custom games with LLM-powered rule interpretation and adjudication.

## Features

- **6 LLM Models** via Groq free API: GPT-OSS 120B/20B, Llama 3.3 70B, Llama 3.1 8B, Qwen 3 32B, Kimi K2
- **6 Pre-built Games**: Prisoner's Dilemma, Public Goods Game, Rock Paper Scissors, Battle of the Sexes, Chicken Game, and Custom Games
- **Custom Games**: Define any game in plain text - LLM analyzes rules, extracts actions, and adjudicates results
- **Auto-Configuration**: LLM analyzes your rules and suggests optimal settings (players, rounds, complexity)
- **Real-time Updates**: Watch agents think and act via Server-Sent Events
- **Full Customization**: Agent personalities, model selection, game parameters
- **Click to View Reasoning**: See full agent thinking in modal
- **Debug Tools**: View LLM request/response logs in real-time

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Groq API key:**
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

   Get your free API key from https://console.groq.com/

3. **Run the server:**
   ```bash
   python app.py
   ```
   Server runs on http://localhost:5001

## Pre-Built Games

### Prisoner's Dilemma
Classic 2-player game where each player chooses to COOPERATE or DEFECT. The game explores trust, betrayal, and repeated interaction strategies.

**Payoffs:**
- Both Cooperate: +3, +3
- One Cooperates, One Defect: Defector gets +5, Cooperator gets 0
- Both Defect: +1, +1

### Public Goods Game
N-player contribution game where each player decides how much to contribute to a public pot. Total is multiplied and split equally - exploring free-rider problems and cooperation.

**Mechanics:**
- Each player has endowment (10 points per round)
- Contribute 0-10 points to public pot
- Pot is multiplied by 1.8x and divided equally among all players

### Rock Paper Scissors
Classic hand game extended to multiple rounds. Players try to detect patterns in their opponent's choices while avoiding predictability.

**Scoring:**
- Win: +2 points
- Draw or Loss: +0 points

### Battle of the Sexes
A coordination game where two players have different preferences but must coordinate to avoid the worst outcome.

**Payoffs:**
- Both Ballet: Player 1 gets +2, Player 2 gets +1
- Both Football: Player 1 gets +1, Player 2 gets +2
- Miscoordination: Both get 0

### Chicken Game
A game of brinkmanship where players choose between safety (SWERVE) and risking disaster (STRAIGHT). Explores courage, reputation, and risk-taking.

**Payoffs:**
- Both Swerve: +1, +1 (safe)
- One Swerves, One Straight: Straight gets +5, Swerve gets 0
- Both Straight: -10, -10 (CRASH!)

### Custom Games
Create your own game by describing rules in plain English. The LLM will:
1. Analyze the game structure
2. Extract valid actions and scoring
3. Adjudicate each round fairly

## How It Works

### For Custom Games

1. **Enter Rules**: Describe your game in plain English
   - Objectives
   - How players take actions
   - Scoring system
   - Win conditions

2. **Analyze**: Click "Analyze Rules" - LLM will:
   - Extract game structure
   - Identify action format (numbers, choices, etc.)
   - Suggest player count and rounds
   - Estimate complexity

3. **Configure Agents**:
   - Choose models for each agent
   - Set personalities (optional)
   - Add multiple agents per model

4. **Play**: Watch agents compete in real-time!
   - See actions as they happen
   - Click reasoning to see full thought process
   - View scores update live

### Example Custom Games

**Lowest Unique Number (LUCK):**
```
2 player game where each player chooses a number.
The player who chose the lowest UNIQUE number wins +5 points.
If both players pick the same number, it's a tie (0 points each).
Number range: 1 to 10.
Rounds: 5
```

**Ultimatum Game:**
```
2 player game. Player 1 proposes how to split 10 points.
Player 2 can ACCEPT (proposed split happens) or REJECT (both get 0).
Rounds: 10 (alternate roles each round)
Goal: Maximize total points.
```

**Stag Hunt:**
```
2 players simultaneously choose: STAG or HARE.
Hunting stag requires cooperation (both must choose STAG).
- Both Stag: +4, +4 (successful hunt)
- One Stag, One Hare: Stag hunter gets 0, Hare hunter gets +2
- Both Hare: +2, +2 (easy but small payoff)
Goal: Maximize total score over 10 rounds.
```

**Trust Game:**
```
2 player game with Investor and Trustee roles.
- Investor gets 10 points, can send any amount (0-10) to Trustee
- Sent amount is tripled
- Trustee can return any amount (0 to 3x) back to Investor
Play 10 rounds, alternating roles.
Goal: Maximize total points.
```

## Project Structure

```
game_theory_arena/
├── app.py                 # Flask backend with SSE
├── game_engine.py        # Base classes for games/agents
├── llm_providers.py      # Groq API integration
├── agents.py             # LLM agent implementation
├── games.py              # Game implementations + CustomGame
├── templates/
│   └── index.html        # Web UI
├── requirements.txt
└── .env.example
```

## Architecture

### Game Engine
- `BaseGame`: Abstract class for all games
- `BaseAgent`: Abstract class for all agents
- `GameEvent`: Event system for real-time updates

### LLM Integration
- `GroqProvider`: Groq API client with comprehensive logging
- `LLMOrchestrator`: Manages multiple models
- `LLMAgent`: Agents powered by LLMs with XML-style reasoning/action format

### Games
- `PrisonersDilemma`: Classic 2-player iterated prisoner's dilemma
- `PublicGoodsGame`: N-player public goods with free-rider problem
- `RockPaperScissors`: Multi-round RPS with pattern detection
- `BattleOfTheSexes`: Coordination game with conflicting preferences
- `ChickenGame`: Brinkmanship game with risk/reward tradeoff
- `CustomGame`: Fully flexible LLM-adjudicated games

### Custom Game Flow

1. **Rule Analysis** (GPT-OSS 120B):
   - Extracts game structure
   - Identifies action format
   - Suggests configuration

2. **Agent Turn**:
   - Dynamic system prompt based on rules
   - Agent outputs reasoning in `<reasoning>` tags
   - Agent outputs action in `<action>` tags
   - Action extracted via LLM with multiple fallback strategies

3. **Adjudication** (GPT-OSS 120B):
   - Understands rules contextually
   - Processes all actions
   - Calculates scores correctly
   - Handles edge cases with fallback mechanisms

## Model Selection

- **GPT-OSS 120B**: Best for complex reasoning, rule analysis, adjudication
- **Llama 3.3 70B**: Good balance of speed and intelligence
- **Llama 3.1 8B**: Fastest, good for simple strategies
- **Qwen 3 32B**: Alternative reasoning model
- **Kimi K2**: Different perspective on strategy

## Debugging

The arena includes comprehensive debugging tools:

1. **🔍 Debug Button**: Click to view real-time LLM logs
2. **Log File**: All requests/responses logged to `llm_debug.log`
3. **Color-Coded Logs**:
   - 🟢 INFO: Normal operations
   - 🟡 WARNING: Empty responses, potential issues
   - 🔴 ERROR: API failures, exceptions

**Common Issues:**

**Empty Responses:**
- Check if model is rate-limited
- Use Debug button to see request details
- Try a different model (Llama 3.1 8B is fastest)

**Action Extraction Failing:**
- View logs to see raw model output
- System prompt may need adjustment for your game
- Try "Analyze Rules" first to validate understanding

## Troubleshooting

**Port 5001 in use?**
Change port in `app.py` last line:
```python
app.run(debug=True, host='0.0.0.0', port=5002, threaded=True)
```

**Agents not responding?**
- Check Groq API key is valid
- Check rate limits (Groq free tier: 30 requests/minute)
- Try using Llama 3.1 8B for faster responses
- Check the Debug logs for errors

**Custom game not working?**
- Make sure rules are clear and specific
- Include scoring methodology
- Specify round count if different from default
- Try "Analyze Rules" first to see if LLM understands
- Use the Debug button to see what's happening

**Scores not updating?**
- Check Debug logs for adjudication errors
- Make sure scoring rules are unambiguous
- Try simplifying the game rules

## License

MIT
