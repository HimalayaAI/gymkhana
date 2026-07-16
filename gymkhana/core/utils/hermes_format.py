"""Hermes tool-calling format converter for ShareGPT datasets.

Converts native OpenAI-style tool-call data into the Hermes/NousResearch
XML-based format used for post-training.

Format reference (NousResearch / Hermes):
  - Tools in system prompt:  <tools> [{...}] </tools>
  - Tool calls:              <tool_call> {"name": ..., "arguments": ...} </tool_call>
  - Tool responses:          <tool_response> {...} </tool_response>
  - Reasoning:               <think> ... </think>

Supports three conversation patterns:
  1. Single-turn:      system → human → gpt (with tool_calls)
  2. Multi-turn:       system → human → gpt → tool → gpt → human → gpt → ...
  3. Multi-step:       system → human → gpt → tool → gpt → tool → gpt (interleaved)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hermes system prompt template
# ---------------------------------------------------------------------------

HERMES_SYSTEM_TEMPLATE = """\
You are a deep thinking AI, you may use extremely long chains of thought to \
deeply consider the problem and deliberate with yourself via systematic \
reasoning processes to help come to a correct solution prior to answering. \
You should enclose your thoughts and internal monologue inside <think> </think> \
tags, and then provide your solution or response to the problem.

You are a function calling AI model. You are provided with function signatures \
within <tools> </tools> XML tags. You may call one or more functions to assist \
with the user query. If available tools are not relevant in assisting with user \
query, just respond in natural conversational language. Don't make assumptions \
about what values to plug into functions. After calling & executing the \
functions, you will be provided with function results within \
<tool_response> </tool_response> XML tags. Here are the available tools:
<tools>
{tools_json}
</tools>
For each function call return a JSON object, with the following pydantic model \
json schema for each:
{{"title": "FunctionCall", "type": "object", "properties": {{"name": {{"title": \
"Name", "type": "string"}}, "arguments": {{"title": "Arguments", "type": \
"object"}}}}, "required": ["name", "arguments"]}}
Each function call should be enclosed within <tool_call> </tool_call> XML tags.
Example:
<tool_call>
{{"name": <function-name>, "arguments": <args-dict>}}
</tool_call>"""


# ---------------------------------------------------------------------------
# Tool schema converters
# ---------------------------------------------------------------------------

def openai_tools_to_hermes(openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI function-calling format tools to Hermes format.

    OpenAI format:
        {"type": "function", "function": {"name": ..., "description": ...,
         "parameters": {"type": "object", "properties": {...}, "required": [...]}}}

    Hermes format:
        {"name": ..., "description": ..., "parameters": {"param": {"description": ..., "type": ...}}}
    """
    hermes_tools = []
    for tool in openai_tools:
        func = tool.get("function", tool)  # Handle both wrapped and unwrapped
        name = func.get("name", "unknown")
        description = func.get("description", "")
        params_schema = func.get("parameters", {})

        # Convert JSON Schema properties to Hermes-style flat params
        properties = params_schema.get("properties", {})
        hermes_params = {}
        for param_name, param_info in properties.items():
            hermes_params[param_name] = {
                "description": param_info.get("description", ""),
                "type": param_info.get("type", "string"),
            }

        hermes_tools.append({
            "name": name,
            "description": description,
            "parameters": hermes_params,
        })
    return hermes_tools


def xlam_tools_to_hermes(xlam_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert xlam-format tools to Hermes format (passthrough with cleanup).

    xlam format is already close to Hermes — just ensure consistent structure.
    """
    hermes_tools = []
    for tool in xlam_tools:
        hermes_tools.append({
            "name": tool.get("name", "unknown"),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {}),
        })
    return hermes_tools


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def build_hermes_system_prompt(
    tools: List[Dict[str, Any]],
    source_format: str = "openai",
) -> str:
    """Build the Hermes system prompt with tools embedded in <tools> XML tags.

    Args:
        tools: Tool definitions (OpenAI, xlam, or Hermes format)
        source_format: "openai", "xlam", or "hermes"

    Returns:
        Complete system prompt string with tools embedded.
    """
    if source_format == "openai":
        hermes_tools = openai_tools_to_hermes(tools)
    elif source_format == "xlam":
        hermes_tools = xlam_tools_to_hermes(tools)
    else:
        hermes_tools = tools

    tools_json = json.dumps(hermes_tools, indent=2)
    return HERMES_SYSTEM_TEMPLATE.format(tools_json=tools_json)


# ---------------------------------------------------------------------------
# Message formatters
# ---------------------------------------------------------------------------

def format_tool_calls(tool_calls: List[Dict[str, Any]]) -> str:
    """Format tool calls as Hermes <tool_call> XML blocks.

    Args:
        tool_calls: List of {"name": ..., "arguments": {...}} dicts

    Returns:
        String with each tool call wrapped in <tool_call> tags.
    """
    blocks = []
    for tc in tool_calls:
        call_json = json.dumps({
            "name": tc.get("name", ""),
            "arguments": tc.get("arguments", {}),
        })
        blocks.append(f"<tool_call>\n{call_json}\n</tool_call>")
    return "\n".join(blocks)


def format_tool_response(
    tool_call_id: Optional[str],
    name: str,
    content: Any,
) -> str:
    """Format a tool response as Hermes <tool_response> XML block.

    Args:
        tool_call_id: The ID of the tool call this responds to
        name: Name of the function that was called
        content: The result/output from executing the tool

    Returns:
        String with tool response wrapped in <tool_response> tags.
    """
    response_data = {
        "tool_call_id": tool_call_id or "",
        "name": name,
        "content": content,
    }
    return f"<tool_response>\n{json.dumps(response_data, indent=2)}\n</tool_response>"


def format_assistant_message(
    content: str,
    reasoning_content: Optional[str] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Format an assistant message with optional reasoning and tool calls.

    The output follows the Hermes convention:
        <think>reasoning</think>
        <tool_call>...</tool_call>
        OR
        <think>reasoning</think>
        Natural language response

    Args:
        content: The text content of the assistant response
        reasoning_content: Optional chain-of-thought reasoning
        tool_calls: Optional list of tool call dicts

    Returns:
        Formatted assistant message string.
    """
    parts = []

    # Add reasoning if present
    if reasoning_content:
        parts.append(f"<think>\n{reasoning_content}\n</think>")

    # Add tool calls or content
    if tool_calls:
        parts.append(format_tool_calls(tool_calls))
    elif content:
        parts.append(content)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# ShareGPT conversation builders
# ---------------------------------------------------------------------------

def build_singleturn_sharegpt(
    *,
    tools: List[Dict[str, Any]],
    user_message: str,
    tool_calls: List[Dict[str, Any]],
    reasoning_content: Optional[str] = None,
    content: Optional[str] = None,
    source_format: str = "openai",
) -> List[Dict[str, str]]:
    """Build a single-turn tool-calling ShareGPT conversation.

    Pattern: system → human → gpt

    Args:
        tools: Tool definitions
        user_message: The user's query
        tool_calls: The predicted/expected tool calls
        reasoning_content: Optional reasoning/thinking content
        content: Optional assistant text content alongside tool calls
        source_format: Format of tool definitions ("openai", "xlam", "hermes")

    Returns:
        ShareGPT conversation as list of {"from": role, "value": content} dicts.
    """
    system_prompt = build_hermes_system_prompt(tools, source_format)

    # Build assistant value
    assistant_parts = []
    if reasoning_content:
        assistant_parts.append(f"<think>\n{reasoning_content}\n</think>")
    if content:
        assistant_parts.append(content)
    assistant_parts.append(format_tool_calls(tool_calls))
    assistant_value = "\n".join(assistant_parts)

    return [
        {"from": "system", "value": system_prompt},
        {"from": "human", "value": user_message},
        {"from": "gpt", "value": assistant_value},
    ]


def build_multiturn_sharegpt(
    *,
    tools: List[Dict[str, Any]],
    turns: List[Dict[str, Any]],
    source_format: str = "openai",
) -> List[Dict[str, str]]:
    """Build a multi-turn tool-calling ShareGPT conversation.

    Pattern: system → human → gpt → tool → gpt → human → gpt → ...

    Each turn dict should have:
        - role: "user", "assistant", or "tool"
        - content: The message content (str)
        - reasoning_content: Optional reasoning (for assistant turns)
        - tool_calls: Optional list of tool call dicts (for assistant turns)
        - tool_call_id: Optional tool call ID (for tool turns)
        - name: Optional tool name (for tool turns)

    Args:
        tools: Tool definitions
        turns: List of turn dicts
        source_format: Format of tool definitions

    Returns:
        ShareGPT conversation.
    """
    system_prompt = build_hermes_system_prompt(tools, source_format)
    conversation = [{"from": "system", "value": system_prompt}]

    role_map = {"user": "human", "assistant": "gpt", "tool": "tool"}

    for turn in turns:
        role = turn.get("role", "user")
        sharegpt_role = role_map.get(role, "human")
        content = turn.get("content", "")
        reasoning = turn.get("reasoning_content")
        tool_calls = turn.get("tool_calls")

        if role == "assistant":
            value = format_assistant_message(
                content=content,
                reasoning_content=reasoning,
                tool_calls=tool_calls,
            )
        elif role == "tool":
            value = format_tool_response(
                tool_call_id=turn.get("tool_call_id"),
                name=turn.get("name", ""),
                content=content if isinstance(content, (dict, list)) else content,
            )
        else:
            value = content

        conversation.append({"from": sharegpt_role, "value": value})

    return conversation


def build_multistep_sharegpt(
    *,
    tools: List[Dict[str, Any]],
    user_message: str,
    steps: List[Dict[str, Any]],
    source_format: str = "openai",
) -> List[Dict[str, str]]:
    """Build a multi-step (interleaved) tool-calling ShareGPT conversation.

    Pattern: system → human → gpt(tool_call) → tool → gpt(tool_call) → tool → gpt(final)

    Each step dict should have:
        - tool_calls: List of tool call dicts (if this step calls tools)
        - tool_responses: List of tool response dicts (results from executing tools)
        - content: Assistant text content (for final response or intermediate text)
        - reasoning_content: Optional reasoning

    Args:
        tools: Tool definitions
        user_message: Initial user query
        steps: Ordered list of step dicts
        source_format: Format of tool definitions

    Returns:
        ShareGPT conversation.
    """
    system_prompt = build_hermes_system_prompt(tools, source_format)
    conversation = [
        {"from": "system", "value": system_prompt},
        {"from": "human", "value": user_message},
    ]

    for step in steps:
        reasoning = step.get("reasoning_content")
        tool_calls = step.get("tool_calls", [])
        tool_responses = step.get("tool_responses", [])
        content = step.get("content", "")

        # Assistant turn: tool calls or final response
        if tool_calls:
            assistant_value = format_assistant_message(
                content=content,
                reasoning_content=reasoning,
                tool_calls=tool_calls,
            )
            conversation.append({"from": "gpt", "value": assistant_value})

            # Tool response turns
            for resp in tool_responses:
                tool_value = format_tool_response(
                    tool_call_id=resp.get("tool_call_id"),
                    name=resp.get("name", ""),
                    content=resp.get("content", {}),
                )
                conversation.append({"from": "tool", "value": tool_value})
        elif content:
            # Final assistant response (no tool calls)
            assistant_value = format_assistant_message(
                content=content,
                reasoning_content=reasoning,
            )
            conversation.append({"from": "gpt", "value": assistant_value})

    return conversation


# ---------------------------------------------------------------------------
# Conversion from TrajectoryResult
# ---------------------------------------------------------------------------

def trajectory_to_hermes_sharegpt(
    result: 'TrajectoryResult',
    task: 'Task',
    source_format: str = "openai",
) -> Optional[List[Dict[str, str]]]:
    """Convert a TrajectoryResult to Hermes-format ShareGPT conversation.

    Automatically detects conversation pattern (single-turn vs multi-turn)
    based on the number of turns and presence of tool responses.

    Args:
        result: The TrajectoryResult from environment execution
        task: The original Task with metadata (tools, expected_tool_calls)
        source_format: Format of tool definitions in task.metadata

    Returns:
        ShareGPT conversation list, or None if conversion fails.
    """
    # Get tools from task metadata
    if source_format == "openai":
        tools = task.metadata.get("tools_openai", [])
    elif source_format == "xlam":
        tools = task.metadata.get("tools_raw", [])
    else:
        tools = task.metadata.get("tools_raw", [])

    if not tools:
        logger.warning(f"No tools found in task {task.id} metadata")
        return None

    # For single-turn: extract tool calls from the final answer
    if len(result.turns) <= 2:
        # Single-turn pattern: user → assistant
        user_turn = next((t for t in result.turns if t.role == "user"), None)
        assistant_turn = next((t for t in result.turns if t.role == "assistant"), None)

        if not user_turn or not assistant_turn:
            logger.warning(f"Missing user or assistant turn in task {task.id}")
            return None

        # Extract tool calls from native response
        from gymkhana.envs.tool_use_singleturn.tool_use_singleturn import (
            extract_tool_calls_from_native_response,
        )
        from gymkhana.envs.parsers import HermesToolCallParser

        tool_calls = extract_tool_calls_from_native_response(assistant_turn.content)
        if not tool_calls:
            tool_calls = HermesToolCallParser.parse(assistant_turn.content)

        if not tool_calls:
            logger.warning(f"No tool calls found in model response for task {task.id} - skipping ShareGPT export")
            return None

        return build_singleturn_sharegpt(
            tools=tools,
            user_message=user_turn.content,
            tool_calls=tool_calls,
            reasoning_content=getattr(assistant_turn, 'reasoning_content', None),
            source_format=source_format,
        )

    # Multi-turn: convert turn-by-turn
    turn_dicts = []
    for turn in result.turns:
        turn_dict: Dict[str, Any] = {
            "role": turn.role,
            "content": turn.content,
        }
        if hasattr(turn, 'reasoning_content') and turn.reasoning_content:
            turn_dict["reasoning_content"] = turn.reasoning_content

        # For assistant turns, try to extract tool calls
        if turn.role == "assistant":
            from gymkhana.envs.tool_use_singleturn.tool_use_singleturn import (
                extract_tool_calls_from_native_response,
            )
            from gymkhana.envs.parsers import HermesToolCallParser

            tc = extract_tool_calls_from_native_response(turn.content)
            if not tc:
                tc = HermesToolCallParser.parse(turn.content)
            if tc:
                turn_dict["tool_calls"] = tc

        turn_dicts.append(turn_dict)

    return build_multiturn_sharegpt(
        tools=tools,
        turns=turn_dicts,
        source_format=source_format,
    )
