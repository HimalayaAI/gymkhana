# Sandbox Service Architecture

This document describes the refactored sandbox service abstraction in Gymkhana,
designed for RLM-style agents that use Python REPL servers.

## Overview

The sandbox service provides isolated code execution environments for reasoning agents.
It supports multiple backends (HTTP REPL, Docker containers) with a unified interface
that includes resource management, session lifecycle tracking, and RLM-compatible
state management.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Environment Layer                           │
│   (MathPythonEnv, OolongEnv, HotpotQAEnv, SWEEnv)              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SandboxService (ABC)                         │
│   - create_session()    - reset_session()                       │
│   - execute()           - delete_session()                       │
│   - execute_bash()      - health_check()                        │
│   - get_state()         - session lifecycle mgmt                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│    REPLSandbox      │         │ DockerSandboxService│
│  (HTTP REPL server) │         │ (Docker container)  │
└─────────┬───────────┘         └─────────┬───────────┘
          │                               │
          ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│    REPLClient       │         │DockerContainerSession│
│  (HTTP API client)  │         │ (container lifecycle)│
└─────────────────────┘         └─────────────────────┘
```

## Key Components

### Configuration Models (`config.py`)

| Model | Purpose |
|-------|---------|
| `SandboxConfig` | Complete sandbox configuration |
| `ResourceConfig` | CPU, memory, GPU, disk limits |
| `TimeoutConfig` | Execution and startup timeouts |
| `RewardConfig` | RLM reward configuration |
| `SessionConfig` | Session-level settings |
| `SubAgentConfig` | Sub-agent LLM configuration |

### State Tracking Models (`state.py`)

| Model | Purpose |
|-------|---------|
| `SessionState` | Complete session state |
| `EpisodeState` | RLM episode tracking (iteration, done, final_answer) |
| `InterpreterState` | Python namespace snapshot |
| `SessionMetrics` | Execution statistics |
| `SessionStatus` | Session lifecycle enum |

### Service Implementations

| Class | Backend | Use Case |
|-------|---------|----------|
| `REPLSandbox` | HTTP REPL | Math, long-context tasks |
| `DockerSandboxService` | Docker | SWE tasks with repo access |

## Usage Examples

### Basic Math Environment

```python
from gymkhana.core.services.sandboxes import REPLSandbox, SandboxConfig

# Create config for math problems
config = SandboxConfig.for_math()

# Initialize sandbox
sandbox = REPLSandbox(config=config)

# Use with async context manager
async with sandbox.session(context="data") as state:
    result = sandbox.execute("print(2 + 2)")
    print(f"Output: {result.output}")
    print(f"Iteration: {state.episode.iteration}")
    print(f"Reward: {result.reward}")
```

### SWE Environment with Docker

```python
from gymkhana.core.services.sandboxes import DockerSandboxService, SandboxConfig

# Create config for software engineering
config = SandboxConfig.for_swe("swebench/swesmith.x86_64.repo")

# Initialize Docker sandbox
sandbox = DockerSandboxService(
    config=config,
    instance_id="task-001",
)

async with sandbox.session() as state:
    # Execute Python code
    result = sandbox.execute("import os; print(os.getcwd())")

    # Execute bash commands
    bash_result = sandbox.execute_bash("ls -la /testbed")

    # Check episode state
    if state.episode.done:
        print(f"Episode complete: {state.episode.final_answer}")
```

### Long-context with Sub-agents

```python
config = SandboxConfig.for_long_context()
config = config.model_copy(update={
    "sub_agent": SubAgentConfig(
        enabled=True,
        model="Hermes-4-70B",
        temperature=0.3,
    )
})

sandbox = REPLSandbox(config=config)
```

### Resource-constrained Sandbox

```python
config = SandboxConfig(
    resources=ResourceConfig(
        cpu_cores=4,
        memory_gb=16.0,
        disk_size_gb=50.0,
    ),
    timeouts=TimeoutConfig(
        execution_seconds=300,
        startup_seconds=180,
    ),
    rewards=RewardConfig(
        on_success=2.0,
        on_iteration=-0.01,  # Per-step penalty
    ),
)
```

## RLM Compatibility

The sandbox service is designed for Reinforcement Learning from Model (RLM) training:

### Episode Tracking

```python
state = sandbox.current_session
print(f"Iteration: {state.episode.iteration}/{state.episode.max_iterations}")
print(f"Progress: {state.episode.progress_ratio * 100:.1f}%")
print(f"Done: {state.episode.done}")
print(f"Final Answer: {state.episode.final_answer}")
```

### Reward Collection

```python
# Per-step rewards from execution
result = sandbox.execute(code)
step_reward = result.reward

# Session-level metrics
metrics = state.metrics
print(f"Total reward: {metrics.total_reward}")
print(f"Step rewards: {metrics.step_rewards}")
```

### Session Metrics

```python
metrics = state.metrics
print(f"Total executions: {metrics.total_executions}")
print(f"Success rate: {metrics.success_rate * 100:.1f}%")
print(f"Avg execution time: {metrics.avg_execution_time_ms:.1f}ms")
print(f"Total reward: {metrics.total_reward}")
```

## Integration with Environments

Environments use the sandbox service through the `ServiceContainer`:

```python
from gymkhana.core.services import ServiceContainer

services = ServiceContainer(
    sandbox=REPLSandbox(config=config),
    inference=inference_service,
    storage=storage_service,
)

env = MathPythonEnv(config=env_config, services=services)
await env.setup()
result = await env.run_task(task)
```

## Backward Compatibility

Legacy code using `REPLClient` directly still works:

```python
# Old API (still supported)
client = REPLClient(server_url="http://localhost:5003")
client.create_session(context="data")
result = client.execute("print('hello')")
client.delete_session()
```

## Testing

Run sandbox service tests:

```bash
pytest tests/services/test_sandbox_services.py -v
pytest tests/services/test_repl.py -v
```

## Files

| File | Description |
|------|-------------|
| `config.py` | Configuration models |
| `state.py` | State tracking models |
| `sandbox.py` | Abstract base class |
| `repl.py` | HTTP REPL implementation |
| `docker_sandbox.py` | Docker implementation |
| `client.py` | Low-level HTTP client |
| `docker_repl.py` | Legacy Docker utilities |
