# Game Theory Arena

An interactive platform where LLM agents compete in game theory scenarios. Features full flexibility for custom games with LLM-powered rule interpretation and adjudication.

## Features

- **6 LLM Models** via Groq free API: GPT-OSS 120B/20B, Llama 3.3 70B, Llama 3.1 8B, Qwen 3 32B, Kimi K2
- **Pre-built Games**: Prisoner's Dilemma, Public Goods Game
- **Custom Games**: Define any game in plain text - LLM analyzes rules, extracts actions, and adjudicates results
- **Auto-Configuration**: LLM analyzes your rules and suggests optimal settings (players, rounds, complexity)
- **Real-time Updates**: Watch agents think and act via Server-Sent Events
- **Full Customization**: Agent personalities, model selection, game parameters
- **Click to View Reasoning**: See full agent thinking in modal

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

**Humiliation Prize (Lowest Unique Number):**
```
2 player game where each player chooses a number for a "prize".
The player who chose the lowest unique number wins the prize amount.
Constraints:
- Number can be any natural number, no upper limits
- Rounds: 5
- If both players pick the same number, no one gets the prize
```

**Rock Paper Scissors Tournament:**
```
3 players, each chooses rock, paper, or scissors each round.
Rock beats scissors, scissors beats paper, paper beats rock.
If all 3 choose the same, no points.
If all 3 different, no points.
Otherwise, winners get 2 points each.
Play 10 rounds.
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
- `GroqProvider`: Groq API client
- `LLMOrchestrator`: Manages multiple models
- `LLMAgent`: Agents powered by LLMs

### Games
- `PrisonersDilemma`: Classic 2-player game
- `PublicGoodsGame`: N-player contribution game
- `CustomGame`: Fully flexible LLM-adjudicated games

## Custom Game Flow

1. **Rule Analysis** (GPT-OSS 120B):
   - Extracts game structure
   - Identifies action format
   - Suggests configuration

2. **Agent Turn**:
   - Dynamic system prompt based on rules
   - Agent thinks and responds
   - Action extracted via LLM

3. **Adjudication** (GPT-OSS 120B):
   - Understands rules contextually
   - Processes all actions
   - Calculates scores correctly
   - Handles edge cases

## Model Selection

- **GPT-OSS 120B**: Best for complex reasoning, rule analysis, adjudication
- **Llama 3.3 70B**: Good balance of speed and intelligence
- **Llama 3.1 8B**: Fastest, good for simple strategies
- **Qwen 3 32B**: Alternative reasoning model
- **Kimi K2**: Different perspective on strategy

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

**Custom game not working?**
- Make sure rules are clear and specific
- Include scoring methodology
- Specify round count if different from default
- Try "Analyze Rules" first to see if LLM understands

## License

MIT
