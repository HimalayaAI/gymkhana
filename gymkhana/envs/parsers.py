"""Common parsing utilities for Gymkhana environment responses."""

from __future__ import annotations

import re
import json
from typing import List, Optional, Tuple


PYTHON_BLOCK_PATTERN = re.compile(r"<python>\s*(.*?)\s*</python>", re.DOTALL)
BASH_BLOCK_PATTERN = re.compile(r"<bash>\s*(.*?)\s*</bash>", re.DOTALL)
MD_PYTHON_BLOCK_PATTERN = re.compile(r"```(?:python|py)\s*\n?(.*?)\s*```", re.DOTALL)
MD_BASH_BLOCK_PATTERN = re.compile(r"```(?:bash|sh)\s*\n?(.*?)\s*```", re.DOTALL)
FINAL_ANSWER_PATTERN = re.compile(r"<final_answer>\s*(.*?)\s*</final_answer>", re.DOTALL)
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
REPL_TAG_PATTERN = re.compile(r"<repl>")
STATE_TAG_PATTERN = re.compile(r"<state>")
SUB_AGENT_TAG_PATTERN = re.compile(r"<sub_agent")
THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
# Also detect orphaned think tags (opening or closing without pair)
ORPHANED_THINK_PATTERN = re.compile(r"(?:^|[^<])(?:</think>|<think>(?!.*?</think>))", re.DOTALL)


def extract_python_blocks(text: str) -> List[str]:
    """Return the contents of all <python> (and markdown code) blocks."""

    blocks = PYTHON_BLOCK_PATTERN.findall(text)
    if not blocks:
        blocks = MD_PYTHON_BLOCK_PATTERN.findall(text)
    return blocks


def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    """Return ordered list of (block_type, code) pairs for python/bash blocks."""

    blocks: List[Tuple[int, str, str]] = []

    # Try XML tags first
    for match in PYTHON_BLOCK_PATTERN.finditer(text):
        blocks.append((match.start(), "python", match.group(1).strip()))
    for match in BASH_BLOCK_PATTERN.finditer(text):
        blocks.append((match.start(), "bash", match.group(1).strip()))

    # Fallback to markdown blocks if nothing found via XML
    if not blocks:
        for match in MD_PYTHON_BLOCK_PATTERN.finditer(text):
            blocks.append((match.start(), "python", match.group(1).strip()))
        for match in MD_BASH_BLOCK_PATTERN.finditer(text):
            blocks.append((match.start(), "bash", match.group(1).strip()))

    blocks.sort(key=lambda item: item[0])
    return [(block_type, code) for _, block_type, code in blocks]


def extract_final_answer(text: str) -> Optional[str]:
    """Return the content inside the <final_answer> tag, if present."""

    match = FINAL_ANSWER_PATTERN.search(text)
    return match.group(1).strip() if match else None


def extract_boxed_answers(text: str) -> List[str]:
    """Return all values found inside \boxed{...} expressions."""

    return [match.strip() for match in BOXED_PATTERN.findall(text)]


def strip_final_answer(text: str) -> str:
    """Remove <final_answer>...</final_answer> from the text."""

    return FINAL_ANSWER_PATTERN.sub("", text).strip()


def has_hallucinated_repl(text: str, allow_think: bool = False) -> bool:
    """Return True when the assistant response contains system-provided tags.

    Args:
        text: The response text to check
        allow_think: If True, <think> tags are allowed and not considered hallucinated
    """
    has_system_tags = bool(
        REPL_TAG_PATTERN.search(text)
        or STATE_TAG_PATTERN.search(text)
        or SUB_AGENT_TAG_PATTERN.search(text)
    )

    if not allow_think:
        has_system_tags = has_system_tags or bool(THINK_TAG_PATTERN.search(text)) or bool(ORPHANED_THINK_PATTERN.search(text))

    return has_system_tags


def strip_hallucinated_repl(text: str, allow_think: bool = False) -> str:
    """Remove hallucinated system tags from text.

    Args:
        text: The response text to clean
        allow_think: If True, <think> tags are preserved
    """
    cleaned = re.sub(r"<repl>.*?</repl>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<state>.*?</state>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<sub_agent[^>]*>.*?</sub_agent>", "", cleaned, flags=re.DOTALL)

    if not allow_think:
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
        # Also remove orphaned think tags
        cleaned = re.sub(r"</think>", "", cleaned)
        cleaned = re.sub(r"<think>", "", cleaned)

    return cleaned.strip()


def extract_think_blocks(text: str) -> List[str]:
    """Extract content from <think>...</think> blocks."""
    return THINK_TAG_PATTERN.findall(text)



# ---------------------------------------------------------------------------
# Tool-Use Parsing & Validation
# ---------------------------------------------------------------------------

class HermesToolCallParser:
    """Parser for Hermes/NousResearch-style XML tool calls.

    Hermes models emit tool calls in XML format:
        <tool_call>{"name": "fn", "arguments": {"key": "val"}}</tool_call>

    This parser is provided for models that don't support native
    OpenAI-style function calling but use Hermes XML format instead.

    Supports:
    - Single tool call: <tool_call>{...}</tool_call>
    - Multiple tool calls: <tool_call>{...}</tool_call><tool_call>{...}</tool_call>
    - Tool calls with list format: <tool_call>[{...}, {...}]</tool_call>
    """

    TAG_PATTERN = re.compile(
        r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL
    )

    @classmethod
    def parse(cls, response: str) -> List[dict]:
        """Parse <tool_call> XML tags from response text.

        Args:
            response: The model's response text containing tool calls

        Returns:
            List of dicts with 'name' and 'arguments' keys.

        Example:
            >>> response = '<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>'
            >>> HermesToolCallParser.parse(response)
            [{"name": "search", "arguments": {"q": "test"}}]
        """
        calls = []
        for match in cls.TAG_PATTERN.findall(response):
            try:
                parsed = json.loads(match.strip())
                if isinstance(parsed, dict) and "name" in parsed:
                    calls.append({
                        "name": parsed["name"],
                        "arguments": parsed.get("arguments", {}),
                    })
                elif isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "name" in item:
                            calls.append({
                                "name": item["name"],
                                "arguments": item.get("arguments", {}),
                            })
            except (json.JSONDecodeError, TypeError):
                continue
        return calls


def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract tool calls from text.

    Supports:
    1. XML format: <tool_call>{JSON}</tool_call>
    2. Markdown JSON: ```json ... ``` (if generic parsing needed)

    Returns list of dicts: [{"name": str, "arguments": dict, ...}]
    """
    calls = []

    # XML pattern
    xml_pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(xml_pattern, text, flags=re.DOTALL | re.IGNORECASE)

    for match in matches:
        try:
            # Try parsing inner content as JSON
            # Clean up potential markdown code blocks inside
            clean_json = match.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:]
            if clean_json.endswith("```"):
                clean_json = clean_json[:-3]

            call_data = json.loads(clean_json.strip())
            if isinstance(call_data, dict) and "name" in call_data:
                calls.append(call_data)
        except json.JSONDecodeError:
            pass

    return calls


def validate_think_plus_calls(text: str) -> bool:
    """Validate 'Regular' Tool-Use format: <think>...</think> then <tool_call>...

    Rules:
    1. Must have <think> block
    2. Must be closed </think>
    3. Tool calls (if any) must appear AFTER </think>
    """
    if "<think>" not in text or "</think>" not in text:
        return False

    # Check order
    think_end = text.rfind("</think>")
    tool_start = text.find("<tool_call>")

    if tool_start != -1 and tool_start < think_end:
        return False  # Tool call inside or before think block

    return True


def validate_interleaved_thinking(text: str) -> bool:
    """Validate 'Interleaved' Tool-Use format.

    Rules:
    1. Must have <think> block
    2. Tool calls can be inside <think>
    """
    if "<think>" not in text:
        return False
    return True


__all__ = [
    "extract_python_blocks",
    "extract_code_blocks",
    "extract_final_answer",
    "extract_boxed_answers",
    "extract_think_blocks",
    "extract_tool_calls",
    "HermesToolCallParser",
    "validate_think_plus_calls",
    "validate_interleaved_thinking",
    "strip_final_answer",
    "has_hallucinated_repl",
    "strip_hallucinated_repl",
]
