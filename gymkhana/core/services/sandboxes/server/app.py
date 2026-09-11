"""
Gymkhana REPL Server for Inference.

Provides a REST API for executing Python code in a sandboxed environment.
Use this when running inference with a trained model that needs to
interact with a REPL in a request/response loop.

Usage:
    # Start server (as module from project root)
    python -m gymkhana.core.services.sandboxes.server --port 5003

    # Or standalone (in Docker)
    python server.py --port 5003

    # Or with Docker for isolation
    docker-compose up --build

API:
    POST /session/create  - Create a new sandbox session
    POST /session/{id}/execute - Execute code in session
    GET  /session/{id}/state - Get session state (variables, files)
    POST /session/{id}/reset - Reset session
    DELETE /session/{id} - Delete session

    POST /execute - Stateless single execution (creates temp session)
"""

import os
import uuid
import time
import threading
import traceback
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from flask import Flask, jsonify, request

logger = logging.getLogger(__name__)

# Support both module and standalone imports
try:
    from .engine import PythonSandbox, ExecutionResult
except (ImportError, ValueError):
    try:
        from gymkhana.core.services.sandboxes.server.engine import PythonSandbox, ExecutionResult
    except (ImportError, ValueError):
        from engine import PythonSandbox, ExecutionResult


app = Flask(__name__)

# Allow large code payloads (e.g. context/document uploads in Oolong, HotpotQA)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB


@app.errorhandler(500)
def handle_500(e):
    """Return JSON on any 500 so clients see the real error (not HTML)."""
    tb = ""
    if getattr(e, "__traceback__", None) is not None:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    else:
        tb = traceback.format_exc()
    logger.exception("Unhandled 500: %s", e)
    return jsonify({"error": str(e), "traceback": tb}), 500


# Custom JSON provider to handle non-serializable types from the REPL
from flask.json.provider import DefaultJSONProvider
class CustomJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if isinstance(obj, uuid.UUID):
            return str(obj)
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

app.json = CustomJSONProvider(app)

# Session storage
sessions: Dict[str, "REPLSession"] = {}
sessions_lock = threading.Lock()

# Config
SESSION_TIMEOUT = 3600  # 1 hour
MAX_SESSIONS = 100
MAX_OUTPUT_CHARS = 8192


@dataclass
class REPLSession:
    """A REPL session with persistent state."""
    id: str
    sandbox: PythonSandbox
    created_at: float
    last_accessed: float
    execution_count: int = 0


_data_inserter = None

def get_data_inserter():
    """Lazily initialize database inserter from environment variables."""
    global _data_inserter
    if _data_inserter is not None:
        return _data_inserter

    try:
        from gymkhana.core.services.storage.env_storage import EnvStorageService
        db_args = {}
        if os.getenv("DB_NAME"): db_args["db_name"] = os.getenv("DB_NAME")
        if os.getenv("DB_USER"): db_args["user"] = os.getenv("DB_USER")
        if os.getenv("DB_PASSWORD"): db_args["password"] = os.getenv("DB_PASSWORD")
        if os.getenv("DB_HOST"): db_args["host"] = os.getenv("DB_HOST")
        if os.getenv("DB_PORT"): db_args["port"] = int(os.getenv("DB_PORT", 5432))

        if db_args:
            _data_inserter = EnvStorageService(**db_args)
            # Init synchronously since we're in Flask context
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_data_inserter.initialize())
            finally:
                loop.close()
    except Exception as e:
        print(f"Failed to initialize data inserter in REPL server: {e}")

    return _data_inserter


def cleanup_old_sessions():
    """Remove sessions that haven't been accessed recently."""
    now = time.time()
    with sessions_lock:
        expired = [
            sid for sid, session in sessions.items()
            if now - session.last_accessed > SESSION_TIMEOUT
        ]
        for sid in expired:
            sessions[sid].sandbox.cleanup()
            del sessions[sid]


def get_session(session_id: str) -> Optional[REPLSession]:
    """Get a session by ID."""
    with sessions_lock:
        session = sessions.get(session_id)
        if session:
            session.last_accessed = time.time()
        return session


# ============================================================
# Session Management Endpoints
# ============================================================

@app.route("/session/create", methods=["POST"])
def create_session():
    """Create a new REPL session."""
    cleanup_old_sessions()

    if len(sessions) >= MAX_SESSIONS:
        return jsonify({"error": "Too many active sessions"}), 503

    # Be lenient if no JSON body provided
    data = request.get_json(silent=True) or {}

    session_id = data.get("session_id", str(uuid.uuid4())[:8])
    packages = data.get("packages", ["numpy", "pandas", "sympy", "json", "re", "math"])
    max_output = data.get("max_output_chars", MAX_OUTPUT_CHARS)
    max_iterations = data.get("max_iterations", 30)
    enable_bash = data.get("enable_bash", False)  # Disabled by default
    context = data.get("context")

    # Sub-agent config (passed from pipeline config)
    sub_agent_config = data.get("sub_agent_config", {})

    # RL reward config (optional)
    reward_config = data.get("reward_config", {})

    # session_id already set above

    sandbox = PythonSandbox(
        max_output_chars=max_output,
        packages=packages,
        enable_filesystem=True,
        enable_bash=enable_bash,
        reward_on_success=reward_config.get("on_success", 1.0),
        reward_on_iteration=reward_config.get("on_iteration", 0.0),
        reward_on_error=reward_config.get("on_error", -0.05),
        reward_on_failure=reward_config.get("on_failure", -0.1),
    )

    # Set max iterations
    sandbox._max_iterations = max_iterations

    # Store sub-agent config in sandbox for later use
    sandbox.sub_agent_config = sub_agent_config

    if context:
        sandbox._namespace["context"] = context
        context_file = sandbox.workspace_dir / "context.txt"
        context_file.write_text(context)

    session = REPLSession(
        id=session_id,
        sandbox=sandbox,
        created_at=time.time(),
        last_accessed=time.time(),
    )

    with sessions_lock:
        sessions[session_id] = session

    return jsonify({
        "session_id": session_id,
        "workspace": str(sandbox.workspace_dir),
        "max_iterations": max_iterations,
        "available_functions": [
            # Sub-agent
            "sub_agent(task, system_prompt=None, context=None) - Invoke sub-agent for semantic analysis",
            "sub_agent_batch(tasks, system_prompt=None) - Batch sub-agent calls",
            # File I/O (auto-detects json/csv)
            "save_to_file(filename, content) - Save to workspace (auto-serializes json/csv)",
            "read_file(filename, lines=N, raw=False) - Read file (auto-parses json/csv)",
            "list_files(pattern) - List workspace files",
            "file_exists(filename) - Check if file exists",
            # Enhanced file ops
            "get_file_info(filename) - Get file metadata (size, lines, type)",
            "search_files(query, pattern='*', max_results=20) - Search content in files",
            # Scratch vs output organization
            "save_scratch(filename, content) - Save to scratch/ (temporary)",
            "save_output(filename, content) - Save to output/ (artifacts)",
            "list_scratch() - List scratch files",
            "list_output() - List output files",
            # Finalization
            "FINAL(value) - Signal task completion with final answer",
            "FINAL_VAR('var_name') - Signal completion with variable value",
            "answer dict - Set answer['content'] and answer['ready']=True when done",
        ]
    })


@app.route("/session/<session_id>/execute", methods=["POST"])
def execute_in_session(session_id: str):
    """Execute code in an existing session."""
    try:
        session = get_session(session_id)
        if not session:
            return jsonify({"error": f"Session {session_id} not found"}), 404

        data = request.get_json(silent=True) or {}
        code = data.get("code", "")

        if not code:
            return jsonify({"error": "No code provided"}), 400

        result = session.sandbox.execute(code)
        session.execution_count += 1

        state_snapshot = session.sandbox.get_state_snapshot()
        state_formatted = session.sandbox.format_state_for_context()
        episode_state = session.sandbox.get_episode_state()

        return jsonify({
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "truncated": result.truncated,
            "execution_time_ms": result.execution_time_ms,
            "answer": result.answer_state,
            "files_created": result.files_created,
            "variables": session.sandbox.get_namespace_summary(),
            "state": state_snapshot,
            "state_formatted": state_formatted,
            "execution_count": session.execution_count,
            "sub_agent_calls": [
                {"task": c.task, "system_prompt": c.system_prompt, "response": c.response}
                for c in result.sub_agent_calls
            ],
            # RLM-style fields
            "done": result.done,
            "final_answer": result.final_answer,
            "iteration": result.iteration,
            "reward": result.reward,
            "episode_state": episode_state,
        })
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Execute failed for session %s: %s", session_id, e)
        return jsonify({
            "error": str(e),
            "traceback": tb,
        }), 500


@app.route("/session/<session_id>/state", methods=["GET"])
def get_session_state(session_id: str):
    """Get current session state."""
    session = get_session(session_id)
    if not session:
        return jsonify({"error": f"Session {session_id} not found"}), 404

    state_snapshot = session.sandbox.get_state_snapshot()
    state_formatted = session.sandbox.format_state_for_context()
    episode_state = session.sandbox.get_episode_state()

    return jsonify({
        "session_id": session_id,
        "answer": session.sandbox.get_answer(),
        "variables": session.sandbox.get_namespace_summary(),
        "state": state_snapshot,
        "state_formatted": state_formatted,
        "files": session.sandbox._list_files("*"),
        "execution_count": session.execution_count,
        "created_at": session.created_at,
        "last_accessed": session.last_accessed,
        # RLM-style fields
        "done": session.sandbox.done,
        "final_answer": session.sandbox.final_answer,
        "iteration": session.sandbox.iteration,
        "episode_state": episode_state,
    })


@app.route("/session/<session_id>/execute_bash", methods=["POST"])
def execute_bash_in_session(session_id: str):
    """Execute bash commands in an existing session."""
    session = get_session(session_id)
    if not session:
        return jsonify({"error": f"Session {session_id} not found"}), 404

    data = request.json or {}
    code = data.get("code", "")
    timeout = data.get("timeout")

    if not code:
        return jsonify({"error": "No code provided"}), 400

    result = session.sandbox.execute_bash(code, timeout=timeout)
    session.execution_count += 1

    return jsonify({
        "success": result.success,
        "output": result.output,
        "error": result.error,
        "truncated": result.truncated,
        "execution_time_ms": result.execution_time_ms,
        "files_created": result.files_created,
        "execution_count": session.execution_count,
    })


@app.route("/session/<session_id>/reset", methods=["POST"])
def reset_session(session_id: str):
    """Reset session to initial state."""
    session = get_session(session_id)
    if not session:
        return jsonify({"error": f"Session {session_id} not found"}), 404

    session.sandbox.reset()
    session.execution_count = 0

    return jsonify({"status": "reset", "session_id": session_id})


@app.route("/session/<session_id>", methods=["DELETE"])
def delete_session(session_id: str):
    """Delete a session and cleanup resources."""
    with sessions_lock:
        session = sessions.pop(session_id, None)

    if not session:
        return jsonify({"error": f"Session {session_id} not found"}), 404

    session.sandbox.cleanup()
    return jsonify({"status": "deleted", "session_id": session_id})


# ============================================================
# LLM Query Endpoint (for sub-LLM calls from sandbox)
# ============================================================

# ============================================================
# Sub-Agent Endpoint (for async LLM calls from sandbox)
# ============================================================

# Default sub-agent config - can be overridden per-request or per-session
DEFAULT_SUB_MODEL = os.environ.get("SUB_LLM_MODEL", "openai:gpt-4o-mini")
DEFAULT_SUB_TEMPERATURE = float(os.environ.get("SUB_LLM_TEMPERATURE", "0.3"))
DEFAULT_SUB_MAX_TOKENS = int(os.environ.get("SUB_LLM_MAX_TOKENS", "2048"))

@app.route("/sub_agent", methods=["POST"])
def sub_agent():
    """
    Invoke a sub-agent for semantic analysis.

    This endpoint is called by the sandbox's sub_agent() function.
    Uses Gymkhana's provider-neutral Pydantic AI inference service.

    Request body:
        {
            "task": "Count the dice rolls in this text...",
            "system_prompt": "You are a helpful assistant.",  # optional
            "context": "...",  # optional, will be wrapped in <file> tags
            "model": "openai:gpt-4o-mini",  # optional, provider:model
            "client": "openai",  # optional legacy provider hint
            "max_tokens": 2048,  # optional
            "temperature": 0.3  # optional
        }

    Returns:
        {"response": "The count is 84.", "model": "..."}
    """
    data = request.json or {}
    task = data.get("task", "")
    system_prompt = data.get("system_prompt", "You are a helpful assistant. Be concise and accurate.")
    context = data.get("context")

    # Get config from request (passed from pipeline) or use defaults
    model = data.get("model", DEFAULT_SUB_MODEL)
    client = data.get("client")
    max_tokens = data.get("max_tokens", DEFAULT_SUB_MAX_TOKENS)
    temperature = data.get("temperature", DEFAULT_SUB_TEMPERATURE)

    if not task:
        return jsonify({"error": "No task provided"}), 400

    # Build user message with optional context in <file> tags
    user_message = task
    if context:
        user_message = f"<file name=\"context\">\n{context}\n</file>\n\n{task}"

    try:
        from gymkhana.core.services.inference.pydantic_ai import PydanticAIInferenceService

        # Accept the former separate client field during migration, while making
        # provider-qualified Pydantic AI model names the canonical API.
        if ":" not in model:
            provider = {"gemini": "google", "google-gla": "google"}.get(client, client)
            if provider not in {"openai", "anthropic", "google"}:
                provider = "openai"
            model = f"{provider}:{model}"

        inference = PydanticAIInferenceService(
            default_model=model,
            default_max_tokens=max_tokens,
            default_temperature=temperature,
            data_inserter=get_data_inserter(),
        )

        import asyncio
        response_text = asyncio.run(inference.generate(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        ))

        return jsonify({
            "response": response_text,
            "model": model,
            "client": model.split(":", 1)[0],
        })

    except Exception as e:
        import traceback
        print(f"Sub-agent error: {e}")
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "response": f"[Sub-agent call failed: {str(e)}]"
        }), 500


# Legacy endpoint alias
@app.route("/llm_query", methods=["POST"])
def llm_query_legacy():
    """Legacy endpoint - redirects to sub_agent."""
    data = request.json or {}
    # Map old format to new format
    new_data = {
        "task": data.get("prompt", ""),
        "system_prompt": data.get("system_prompt"),
        "max_tokens": data.get("max_tokens"),
        "temperature": data.get("temperature"),
    }
    # Forward to sub_agent
    with app.test_request_context(json=new_data):
        return sub_agent()


# ============================================================
# Stateless Execution
# ============================================================

@app.route("/execute", methods=["POST"])
def execute_stateless():
    """Execute code without session management."""
    data = request.json or {}
    code = data.get("code", "")
    context = data.get("context")

    if not code:
        return jsonify({"error": "No code provided"}), 400

    sandbox = PythonSandbox(max_output_chars=MAX_OUTPUT_CHARS, enable_filesystem=True)

    try:
        if context:
            sandbox._namespace["context"] = context

        result = sandbox.execute(code)

        return jsonify({
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "execution_time_ms": result.execution_time_ms,
            "answer": result.answer_state,
        })
    finally:
        sandbox.cleanup()


# ============================================================
# Health Check
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "active_sessions": len(sessions),
        "max_sessions": MAX_SESSIONS,
    })


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Gymkhana REPL Server")
    parser.add_argument("--port", type=int, default=5003, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    print(f"Starting Gymkhana REPL Server on {args.host}:{args.port}")
    print(f"Max sessions: {MAX_SESSIONS}, Session timeout: {SESSION_TIMEOUT}s")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
