"""
Flask Backend with Server-Sent Events for real-time game updates
"""

import os
import asyncio
import json
import time
from queue import Queue
from threading import Thread
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from game_engine import GameStatus
from llm_providers import LLMOrchestrator, LLMModel
from agents import AgentFactory
from games import CustomGame


app = Flask(__name__)
CORS(app)

# Global state
game_state = {
    "status": "idle",
    "game": None,
    "agents": [],
    "events": [],
    "config": {}
}

# SSE queues for clients
clients = []


def notify_clients(event_data):
    """Send event to all connected SSE clients"""
    for queue in clients:
        queue.put(event_data)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available LLM models"""
    models = [
        {"id": "gpt-oss-120b", "name": "GPT-OSS 120B"},
        {"id": "gpt-oss-20b", "name": "GPT-OSS 20B"},
        {"id": "llama-3.3-70b", "name": "Llama 3.3 70B"},
        {"id": "llama-3.1-8b", "name": "Llama 3.1 8B (Fast)"},
        {"id": "qwen-3-32b", "name": "Qwen 3 32B"},
        {"id": "kimi-k2", "name": "Kimi K2"},
    ]
    return jsonify(models)


@app.route('/api/analyze-rules', methods=['POST'])
def analyze_rules():
    """Analyze game rules and suggest configuration"""
    data = request.json
    rules = data.get('rules', '')

    if not rules:
        return jsonify({"error": "Rules required"}), 400

    async def do_analyze():
        llm = LLMOrchestrator()
        return await llm.analyze_rules(rules)

    result = asyncio.run(do_analyze())
    return jsonify(result)


@app.route('/api/setup', methods=['POST'])
def setup_game():
    """Setup game with configuration"""
    global game_state

    data = request.json
    game_type = data.get('game_type', 'custom')
    rules = data.get('rules', '')
    agent_config = data.get('agents', [])
    config = data.get('config', {})

    try:
        # Initialize LLM orchestrator
        llm = LLMOrchestrator()
        agent_factory = AgentFactory(llm)

        # ALL games use CustomGame now for maximum flexibility
        game = CustomGame(rules, llm)
        asyncio.run(game.analyze_rules())

        # Create agents
        agents = agent_factory.create_multiple_agents(agent_config)

        # Setup game with max_rounds from config
        max_rounds = config.get('maxRounds', None)
        game.setup(agents, max_rounds=max_rounds)

        game_state = {
            "status": "ready",
            "game": game,
            "agents": agents,
            "events": [],
            "config": config
        }

        notify_clients({
            "type": "game_ready",
            "data": {
                "agents": [a.to_dict() for a in agents],
                "game": game.to_dict()
            }
        })

        return jsonify({"status": "ready"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/start', methods=['POST'])
def start_game():
    """Start the game"""
    global game_state

    if game_state["status"] not in ["ready", "paused"]:
        return jsonify({"error": "Game not ready"}), 400

    game_state["status"] = "running"

    # Start game in background thread
    thread = Thread(target=run_game)
    thread.daemon = True
    thread.start()

    return jsonify({"status": "running"})


@app.route('/api/pause', methods=['POST'])
def pause_game():
    """Pause the game"""
    global game_state
    game_state["status"] = "paused"
    return jsonify({"status": "paused"})


@app.route('/api/stop', methods=['POST'])
def stop_game():
    """Stop the game"""
    global game_state
    game_state["status"] = "stopped"
    return jsonify({"status": "stopped"})


@app.route('/api/state', methods=['GET'])
def get_state():
    """Get current game state"""
    global game_state

    game = game_state.get("game")
    return jsonify({
        "status": game_state.get("status", "idle"),
        "agents": [a.to_dict() for a in game_state.get("agents", [])],
        "game": game.to_dict() if game else None,
        "events": [e.to_dict() for e in game_state.get("events", [])]
    })


@app.route('/api/events')
def events():
    """SSE endpoint for real-time events"""
    def event_stream():
        queue = Queue()
        clients.append(queue)

        try:
            while True:
                event = queue.get()
                yield f"data: {json.dumps(event)}\n\n"
        except GeneratorExit:
            clients.remove(queue)

    return app.response_class(
        event_stream(),
        mimetype='text/event-stream'
    )


def run_game():
    """Run game loop (runs in background thread)"""
    global game_state

    game = game_state["game"]
    agents = game_state["agents"]

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        while not game.is_game_over() and game_state["status"] == "running":
            # Get actions from all agents
            actions = []

            # Get system prompt from game if available (for CustomGame)
            system_prompt = game.get_system_prompt() if hasattr(game, 'get_system_prompt') else None

            for agent in agents:
                state_desc = game.get_state_description(agent.agent_id)

                action = loop.run_until_complete(
                    agent.decide_action(
                        game_state=state_desc,
                        available_actions=[],  # Empty for custom games
                        action_format="free_text",
                        system_prompt_override=system_prompt
                    )
                )

                actions.append(action)

            # Process actions (all games are now async CustomGame)
            events = loop.run_until_complete(game.process_actions(actions))

            # Store events and notify
            game_state["events"].extend(events)

            for event in events:
                notify_clients({
                    "type": "game_event",
                    "data": event.to_dict()
                })

            # Check game over
            if game.is_game_over():
                game_state["status"] = "finished"

                scores = game.get_scores()
                max_score = max(scores.values())
                winners = [agent_id for agent_id, score in scores.items() if score == max_score]

                result_data = {
                    "final_scores": scores,
                    "winners": winners,
                    "is_tie": len(winners) > 1
                }

                if len(winners) == 1:
                    result_data["winner"] = winners[0]

                notify_clients({
                    "type": "game_over",
                    "data": result_data
                })
                break

            # Small delay between rounds
            time.sleep(1)

    except Exception as e:
        notify_clients({
            "type": "error",
            "data": {"message": str(e)}
        })
        game_state["status"] = "error"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
