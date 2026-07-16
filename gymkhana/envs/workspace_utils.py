"""Common utilities for workspace filesystem formatting across environments.

Provides standardized formatting for file descriptors and workspace context
that can be embedded in system prompts or user messages.
"""

from typing import Dict, List, Optional


def format_file_descriptor(
    filename: str,
    content_length: int,
    file_type: str = "txt",
    title: Optional[str] = None,
    additional_attrs: Optional[Dict[str, str]] = None,
) -> str:
    """Format a single file descriptor in standardized XML-like format.

    Args:
        filename: Name of the file in the workspace
        content_length: Size of the file content in characters
        file_type: Type of file (default: "txt")
        title: Optional human-readable title for the file
        additional_attrs: Optional additional attributes to include in the tag

    Returns:
        Formatted file descriptor string

    Example:
        >>> format_file_descriptor("context.txt", 12345)
        '<file name="context.txt" type="txt" chars="12345">\\n[Content saved to workspace - use read_file(\\'context.txt\\') to load]\\n</file>'

        >>> format_file_descriptor("doc_01_Title.txt", 5000, title="Document Title")
        '<file name="doc_01_Title.txt" type="txt" chars="5000" title="Document Title">\\n[Content saved to workspace - use read_file(\\'doc_01_Title.txt\\') to load]\\n</file>'
    """
    attrs = [
        f'name="{filename}"',
        f'type="{file_type}"',
        f'chars="{content_length}"',
    ]

    if title:
        attrs.append(f'title="{title}"')

    if additional_attrs:
        for key, value in additional_attrs.items():
            attrs.append(f'{key}="{value}"')

    attrs_str = " ".join(attrs)

    return (
        f"<file {attrs_str}>\n"
        f"[Content saved to workspace - use read_file('{filename}') to load]\n"
        f"</file>"
    )


def format_multiple_files(
    files: List[Dict[str, any]],
    file_type: str = "txt",
) -> str:
    """Format multiple file descriptors with consistent spacing.

    Args:
        files: List of file info dicts, each containing:
            - filename: str (required)
            - content_length: int (required)
            - title: str (optional)
            - additional_attrs: Dict[str, str] (optional)
        file_type: Default file type for all files

    Returns:
        Formatted string with all file descriptors separated by blank lines

    Example:
        >>> files = [
        ...     {"filename": "doc_01.txt", "content_length": 1000, "title": "Doc 1"},
        ...     {"filename": "doc_02.txt", "content_length": 2000, "title": "Doc 2"},
        ... ]
        >>> print(format_multiple_files(files))
    """
    file_tags = []

    for file_info in files:
        filename = file_info["filename"]
        content_length = file_info["content_length"]
        title = file_info.get("title")
        additional_attrs = file_info.get("additional_attrs")

        tag = format_file_descriptor(
            filename=filename,
            content_length=content_length,
            file_type=file_type,
            title=title,
            additional_attrs=additional_attrs,
        )
        file_tags.append(tag)

    return "\n\n".join(file_tags)


def format_workspace_context(
    prompt: str,
    files: List[Dict[str, any]],
    file_type: str = "txt",
    instructions: Optional[str] = None,
) -> str:
    """Format a complete user message with prompt, file descriptors, and optional instructions.

    Args:
        prompt: The main task prompt/question
        files: List of file info dicts (see format_multiple_files)
        file_type: Default file type for all files
        instructions: Optional environment-specific instructions to append

    Returns:
        Complete formatted message with prompt and file context

    Example:
        >>> prompt = "What is the answer to the question?"
        >>> files = [{"filename": "context.txt", "content_length": 5000}]
        >>> message = format_workspace_context(prompt, files)
    """
    parts = [prompt]

    if files:
        files_section = format_multiple_files(files, file_type=file_type)
        parts.append(files_section)

    if instructions:
        parts.append(instructions)

    return "\n\n".join(parts)


def format_single_file_context(
    prompt: str,
    filename: str,
    content_length: int,
    file_type: str = "txt",
    instructions: Optional[str] = None,
) -> str:
    """Convenience function for formatting context with a single file.

    Args:
        prompt: The main task prompt/question
        filename: Name of the file in the workspace
        content_length: Size of the file content in characters
        file_type: Type of file (default: "txt")
        instructions: Optional environment-specific instructions to append

    Returns:
        Complete formatted message with prompt and single file descriptor

    Example:
        >>> message = format_single_file_context(
        ...     "Answer the question",
        ...     "context.txt",
        ...     12345
        ... )
    """
    file_descriptor = format_file_descriptor(
        filename=filename,
        content_length=content_length,
        file_type=file_type,
    )

    parts = [prompt, file_descriptor]

    if instructions:
        parts.append(instructions)

    return "\n\n".join(parts)


__all__ = [
    "format_file_descriptor",
    "format_multiple_files",
    "format_workspace_context",
    "format_single_file_context",
]
