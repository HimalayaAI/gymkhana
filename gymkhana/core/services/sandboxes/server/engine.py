"""
Sandboxed Python REPL with Filesystem Access.

Provides isolated code execution with:
- State persistence across turns
- Filesystem access for dynamic context management
- Output truncation
- Pre-injected answer variable and llm_query function

Inspired by Cursor's "dynamic context discovery" pattern where
files are the primitive for context management.
"""

import ast
import sys
import io
import os
import tempfile
import shutil
import traceback
import asyncio
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass, field
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path


@dataclass
class SubAgentCall:
    """Record of a sub-agent invocation."""
    task: str
    system_prompt: Optional[str]
    response: str


@dataclass
class ExecutionResult:
    """Result of code execution in sandbox."""
    success: bool
    output: str
    error: Optional[str] = None
    answer_state: Optional[Dict[str, Any]] = None
    truncated: bool = False
    execution_time_ms: int = 0
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    sub_agent_calls: List[SubAgentCall] = field(default_factory=list)
    # RLM-style fields
    done: bool = False  # Episode complete (FINAL called or answer ready)
    final_answer: Optional[str] = None  # Extracted final answer
    iteration: int = 0  # Current iteration count
    reward: float = 0.0  # Step reward for RL training


class PythonSandbox:
    """
    Sandboxed Python execution environment with filesystem access.

    Features:
    - Persistent state across multiple executions
    - Workspace filesystem for dynamic context management
    - Pre-injected `answer` variable: {"content": "", "ready": False}
    - Pre-injected `FINAL()` and `FINAL_VAR()` helper functions (RLM-style)
    - Pre-injected `sub_agent()` function for sub-LLM calls
    - Output truncation to prevent context explosion
    - Iteration tracking and reward signals for RL training

    Finalization Patterns (RLM-compatible):
    1. FINAL(answer) - Direct call with value
    2. print('FINAL(answer)') - Print pattern (regex detected)
    3. FINAL_VAR("var_name") - Variable lookup
    4. answer = {"content": "...", "ready": True} - Prime Intellect style

    Filesystem Design (inspired by Cursor's dynamic context discovery):
    - Long outputs are written to files instead of bloating context
    - Agent can read/write files to manage working memory
    - Files persist across turns for multi-step reasoning
    """

    def __init__(
        self,
        max_output_chars: int = 8192,
        max_output_lines: int = 500,
        timeout_seconds: int = 120,
        llm_batch_callback: Optional[Callable] = None,
        packages: Optional[List[str]] = None,
        workspace_dir: Optional[str] = None,
        enable_filesystem: bool = True,
        enable_bash: bool = False,  # Disabled by default, enable for SWE tasks
        # RL reward configuration
        reward_on_success: float = 1.0,
        reward_on_iteration: float = 0.0,
        reward_on_error: float = -0.05,
        reward_on_failure: float = -0.1,
        # Context preview
        context_preview_length: int = 500,
    ):
        self.max_output_chars = max_output_chars
        self.max_output_lines = max_output_lines
        self.timeout_seconds = timeout_seconds
        self.llm_batch_callback = llm_batch_callback
        self.enable_filesystem = enable_filesystem
        self.enable_bash = enable_bash

        # RL reward configuration
        self.reward_on_success = reward_on_success
        self.reward_on_iteration = reward_on_iteration
        self.reward_on_error = reward_on_error
        self.reward_on_failure = reward_on_failure

        # Context preview
        self.context_preview_length = context_preview_length

        # Episode state
        self._iteration = 0
        self._max_iterations = 30  # Default, can be set via reset()
        self._done = False
        self._final_answer: Optional[str] = None

        # Setup workspace directory
        if workspace_dir:
            self.workspace_dir = Path(workspace_dir)
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
            self._owns_workspace = False
        else:
            self.workspace_dir = Path(tempfile.mkdtemp(prefix="gymkhana_"))
            self._owns_workspace = True

        # Save cwd to restore after each execute(); avoid getcwd() during execute() when cwd may be deleted
        try:
            self._restore_cwd = os.getcwd()
        except FileNotFoundError:
            self._restore_cwd = os.path.expanduser("~") or tempfile.gettempdir()

        # Track file changes
        self._initial_files: set = set()
        self._scan_workspace()

        # Track sub-agent calls during execution
        self._sub_agent_calls: List[SubAgentCall] = []

        # Initialize sandbox namespace
        self._namespace: Dict[str, Any] = {}
        self._init_namespace(packages or [])

        # Execution history
        self.history: List[Tuple[str, ExecutionResult]] = []

    def _scan_workspace(self) -> set:
        """Scan workspace for existing files."""
        if not self.enable_filesystem:
            return set()
        files = set()
        for f in self.workspace_dir.rglob("*"):
            if f.is_file():
                files.add(str(f.relative_to(self.workspace_dir)))
        self._initial_files = files
        return files

    def _get_file_changes(self) -> Tuple[List[str], List[str]]:
        """Get files created and modified since last scan."""
        if not self.enable_filesystem:
            return [], []

        current_files = set()
        for f in self.workspace_dir.rglob("*"):
            if f.is_file():
                current_files.add(str(f.relative_to(self.workspace_dir)))

        created = list(current_files - self._initial_files)
        modified = []

        return created, modified

    def _init_namespace(self, packages: List[str]) -> None:
        """Initialize the sandbox namespace with builtins and helpers."""
        import builtins

        safe_builtins = {
            'abs': abs, 'all': all, 'any': any, 'ascii': ascii,
            'bin': bin, 'bool': bool, 'bytearray': bytearray, 'bytes': bytes,
            'callable': callable, 'chr': chr, 'complex': complex,
            'dict': dict, 'divmod': divmod, 'enumerate': enumerate,
            'filter': filter, 'float': float, 'format': format,
            'frozenset': frozenset, 'getattr': getattr, 'hasattr': hasattr,
            'hash': hash, 'hex': hex, 'int': int, 'isinstance': isinstance,
            'issubclass': issubclass, 'iter': iter, 'len': len, 'list': list,
            'map': map, 'max': max, 'min': min, 'next': next,
            'oct': oct, 'ord': ord, 'pow': pow, 'print': print,
            'range': range, 'repr': repr, 'reversed': reversed, 'round': round,
            'set': set, 'slice': slice, 'sorted': sorted, 'str': str,
            'sum': sum, 'tuple': tuple, 'type': type, 'zip': zip,
            'True': True, 'False': False, 'None': None,
            '__import__': __import__,
            '__build_class__': __build_class__,  # Required for class definitions
            'open': self._sandboxed_open if self.enable_filesystem else open,
            'input': input,
        }

        self._namespace['__builtins__'] = safe_builtins
        # Add required module-level attributes
        self._namespace['__name__'] = '__main__'
        self._namespace['__doc__'] = None
        self._namespace['answer'] = {"content": "", "ready": False}

        # RLM-style finalization helpers
        self._namespace['FINAL'] = self._final_helper
        self._namespace['FINAL_VAR'] = self._final_var_helper

        # Sub-agent for additional analysis (async LLM calls)
        self._namespace['sub_agent'] = self._sub_agent_wrapper
        self._namespace['sub_agent_batch'] = self._sub_agent_batch_wrapper
        self._namespace['sub_llm'] = self._sub_agent_wrapper
        self._namespace['sub_llm_batch'] = self._sub_agent_batch_wrapper
        # Legacy alias
        self._namespace['llm_query'] = self._sub_agent_wrapper
        self._namespace['llm_query_batched'] = self._sub_agent_batch_wrapper

        if self.enable_filesystem:
            self._namespace['workspace'] = str(self.workspace_dir)
            # File operations (auto-detect type for json/csv)
            self._namespace['save_to_file'] = self._save_to_file
            self._namespace['read_file'] = self._read_file
            self._namespace['list_files'] = self._list_files
            self._namespace['file_exists'] = self._file_exists
            # Enhanced operations
            self._namespace['get_file_info'] = self._get_file_info
            self._namespace['search_files'] = self._search_files
            # Scratch vs output organization
            self._namespace['save_scratch'] = self._save_scratch
            self._namespace['save_output'] = self._save_output
            self._namespace['list_scratch'] = self._list_scratch
            self._namespace['list_output'] = self._list_output

        self._import_packages(packages)

    def _final_helper(self, value: Any) -> str:
        """
        FINAL(answer) - RLM-style finalization helper.

        Marks the episode as done and sets the final answer.
        Returns a string representation for print detection.
        """
        self._done = True
        self._final_answer = str(value)
        self._namespace['answer'] = {"content": str(value), "ready": True}
        return f"FINAL({value})"

    def _final_var_helper(self, var_name: str) -> str:
        """
        FINAL_VAR("var_name") - RLM-style variable lookup finalization.

        Looks up the variable by name and calls FINAL with its value.
        """
        if var_name not in self._namespace:
            raise NameError(f"Variable '{var_name}' not found in namespace")
        value = self._namespace[var_name]
        return self._final_helper(value)

    def _sandboxed_open(self, path, mode='r', *args, **kwargs):
        """Sandboxed open() that restricts to workspace directory.

        For SWE tasks (enable_bash=True), also allows access to /testbed.
        """
        path = Path(path)
        if not path.is_absolute():
            path = self.workspace_dir / path

        # For SWE tasks, allow access to /testbed
        if self.enable_bash:
            testbed_path = Path('/testbed')
            path_str = str(path)

            # Check if path starts with /testbed
            if path_str.startswith('/testbed'):
                # Allow access to /testbed
                if 'w' in mode or 'a' in mode:
                    path.parent.mkdir(parents=True, exist_ok=True)
                return open(path, mode, *args, **kwargs)

        # Check workspace access
        try:
            path.resolve().relative_to(self.workspace_dir.resolve())
        except ValueError:
            raise PermissionError(f"Access denied: {path} is outside workspace")

        if 'w' in mode or 'a' in mode:
            path.parent.mkdir(parents=True, exist_ok=True)

        return open(path, mode, *args, **kwargs)

    def _save_to_file(self, filename: str, content: Any) -> str:
        """
        Save content to a file in the workspace.

        Auto-detects how to serialize based on file extension:
        - .json -> JSON serializes dict/list
        - .csv -> saves DataFrame as CSV
        - others -> saves as string

        Args:
            filename: Target filename
            content: Content to save (str, dict, list, or DataFrame)

        Returns:
            Full path to saved file
        """
        filepath = self.workspace_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        ext = filepath.suffix.lower()

        if ext == '.json':
            import json
            if isinstance(content, str):
                filepath.write_text(content)
            else:
                filepath.write_text(json.dumps(content, indent=2, ensure_ascii=False))

        elif ext == '.csv':
            # Check if it's a DataFrame
            if hasattr(content, 'to_csv'):
                content.to_csv(filepath, index=False)
            else:
                filepath.write_text(str(content))

        else:
            filepath.write_text(str(content))

        return str(filepath)

    def _read_file(self, filename: str, lines: Optional[int] = None, raw: bool = False) -> Any:
        """
        Read content from a file in the workspace.

        Auto-detects file type and parses accordingly:
        - .json -> returns dict/list
        - .csv -> returns pandas DataFrame
        - .txt, .md, etc -> returns string

        Args:
            filename: File to read
            lines: Optional, return only last N lines (text files only)
            raw: If True, always return raw text regardless of extension

        Returns:
            Parsed content based on file type, or raw string if raw=True
        """
        filepath = self.workspace_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filename}")

        ext = filepath.suffix.lower()

        # Return raw text if requested or for line-limited reads
        if raw or lines is not None:
            content = filepath.read_text(encoding="utf-8")
            if lines:
                content_lines = content.split('\n')
                content = '\n'.join(content_lines[-lines:])
            return content

        # Auto-detect and parse based on extension
        if ext == '.json':
            import json
            return json.loads(filepath.read_text(encoding="utf-8"))

        elif ext == '.csv':
            pd = self._namespace.get('pd')
            if pd is None:
                import pandas as pd
            return pd.read_csv(filepath)

        elif ext == '.tsv':
            pd = self._namespace.get('pd')
            if pd is None:
                import pandas as pd
            return pd.read_csv(filepath, sep='\t')

        else:
            # Default: return as text
            return filepath.read_text(encoding="utf-8")

    def _list_files(self, pattern: str = "*") -> List[str]:
        """List files in workspace matching pattern."""
        files = []
        for f in self.workspace_dir.rglob(pattern):
            if f.is_file():
                files.append(str(f.relative_to(self.workspace_dir)))
        return sorted(files)

    def _file_exists(self, filename: str) -> bool:
        """Check if file exists in workspace."""
        return (self.workspace_dir / filename).exists()

    # ============================================================
    # Enhanced Filesystem Operations
    # ============================================================

    def _get_file_info(self, filename: str) -> Dict[str, Any]:
        """
        Get metadata about a file.

        Returns:
            Dict with size, lines, type, modified time
        """
        filepath = self.workspace_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filename}")

        stat = filepath.stat()
        content = filepath.read_text(encoding="utf-8")
        lines = content.count('\n') + 1 if content else 0

        # Detect file type from extension
        ext = filepath.suffix.lower()
        file_type = {
            '.txt': 'text', '.md': 'markdown',
            '.json': 'json', '.csv': 'csv', '.tsv': 'tsv',
            '.py': 'python', '.js': 'javascript',
            '.yaml': 'yaml', '.yml': 'yaml',
            '.xml': 'xml', '.html': 'html',
        }.get(ext, 'text')

        return {
            "name": filename,
            "size": stat.st_size,
            "lines": lines,
            "type": file_type,
            "chars": len(content),
        }

    def _search_files(
        self,
        query: str,
        pattern: str = "*",
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for content across files in workspace.

        Args:
            query: Regex pattern to search for
            pattern: Glob pattern for files to search (default: all)
            max_results: Maximum number of matches to return

        Returns:
            List of matches with file, line number, and context
        """
        import re
        results = []

        try:
            regex = re.compile(query, re.IGNORECASE)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")

        for filepath in self.workspace_dir.rglob(pattern):
            if not filepath.is_file():
                continue

            try:
                content = filepath.read_text(encoding="utf-8")
                for i, line in enumerate(content.split('\n'), 1):
                    if regex.search(line):
                        results.append({
                            "file": str(filepath.relative_to(self.workspace_dir)),
                            "line": i,
                            "match": line[:200] + "..." if len(line) > 200 else line
                        })
                        if len(results) >= max_results:
                            return results
            except (UnicodeDecodeError, PermissionError):
                continue

        return results

    # ============================================================
    # Scratch vs Output File Management
    # ============================================================

    def _save_scratch(self, filename: str, content: Any) -> str:
        """
        Save temporary/intermediate content to scratch directory.
        Scratch files are for reasoning steps, not final outputs.
        """
        scratch_path = Path("scratch") / filename
        return self._save_to_file(str(scratch_path), content)

    def _save_output(self, filename: str, content: Any) -> str:
        """
        Save final output/artifact to output directory.
        Output files are important results to preserve.
        """
        output_path = Path("output") / filename
        return self._save_to_file(str(output_path), content)

    def _list_scratch(self) -> List[str]:
        """List files in scratch directory."""
        scratch_dir = self.workspace_dir / "scratch"
        if not scratch_dir.exists():
            return []
        return [str(f.relative_to(scratch_dir)) for f in scratch_dir.rglob("*") if f.is_file()]

    def _list_output(self) -> List[str]:
        """List files in output directory."""
        output_dir = self.workspace_dir / "output"
        if not output_dir.exists():
            return []
        return [str(f.relative_to(output_dir)) for f in output_dir.rglob("*") if f.is_file()]

    def _import_packages(self, packages: List[str]) -> None:
        """Import pre-approved packages into namespace."""
        import_map = {
            'numpy': 'np', 'pandas': 'pd', 'sympy': 'sympy',
            'scipy': 'scipy', 'json': 'json', 're': 're',
            'collections': 'collections', 'math': 'math',
            'itertools': 'itertools', 'functools': 'functools',
            'datetime': 'datetime', 'pathlib': 'pathlib', 'os': 'os',
        }

        for pkg in packages:
            try:
                module = __import__(pkg)
                alias = import_map.get(pkg, pkg)
                self._namespace[alias] = module
            except ImportError:
                pass

    def _sub_agent_wrapper(
        self,
        task: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None
    ) -> str:
        """
        Invoke a sub-agent for semantic analysis.

        Args:
            task: The task/question for the sub-agent
            system_prompt: Optional system prompt (defaults to helpful assistant)
            context: Optional context to include (will be wrapped in <file> tags)

        Returns:
            Sub-agent response string
        """
        import requests

        # Get the server URL from environment or use default
        server_url = os.environ.get("LLM_QUERY_URL", "http://localhost:5003/sub_agent")

        # Get sub-agent config from sandbox (set during session creation)
        sub_config = getattr(self, 'sub_agent_config', {})

        response_text = ""
        try:
            payload = {
                "task": task,
                "model": sub_config.get("model", "Hermes-4-70B"),
                "client": sub_config.get("client", "litellm"),
                "max_tokens": sub_config.get("max_tokens", 2048),
                "temperature": sub_config.get("temperature", 0.3),
            }
            if system_prompt:
                payload["system_prompt"] = system_prompt
            if context:
                payload["context"] = context

            resp = requests.post(server_url, json=payload, timeout=120)

            if resp.status_code == 200:
                data = resp.json()
                response_text = data.get("response", "[No response]")
            else:
                response_text = f"[Sub-agent call failed: HTTP {resp.status_code}]"

        except requests.exceptions.ConnectionError:
            # If server endpoint not available, return placeholder
            response_text = f"[sub_agent not available - task: {task[:100]}...]"
        except Exception as e:
            response_text = f"[sub_agent error: {str(e)}]"

        # Record the call for observation formatting
        self._sub_agent_calls.append(SubAgentCall(
            task=task[:200] + "..." if len(task) > 200 else task,
            system_prompt=system_prompt,
            response=response_text
        ))

        return response_text

    def _sub_agent_batch_wrapper(
        self,
        tasks: List[str],
        system_prompt: Optional[str] = None
    ) -> List[str]:
        """Invoke sub-agents with multiple tasks (sequential for now)."""
        results = []
        for task in tasks:
            result = self._sub_agent_wrapper(task, system_prompt)
            results.append(result)
        return results

    def _truncate_output(self, output: str) -> Tuple[str, bool]:
        """Truncate output to configured limits."""
        truncated = False

        lines = output.split('\n')
        if len(lines) > self.max_output_lines:
            lines = lines[:self.max_output_lines]
            lines.append(f"\n... [truncated: {len(lines)} lines shown]")
            output = '\n'.join(lines)
            truncated = True

        if len(output) > self.max_output_chars:
            output = output[:self.max_output_chars]
            output += f"\n... [truncated at {self.max_output_chars} chars]"
            truncated = True

        return output, truncated

    def execute(self, code: str) -> ExecutionResult:
        """Execute Python code in the sandbox."""
        import time
        import re
        start_time = time.time()

        # Increment iteration counter
        self._iteration += 1

        # Clear sub-agent calls from previous execution
        self._sub_agent_calls = []

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        answer_val = self._namespace.get('answer', {})
        answer_state = answer_val.copy() if isinstance(answer_val, dict) else {"content": str(answer_val), "ready": False}
        result = ExecutionResult(
            success=False, output="",
            answer_state=answer_state,
            iteration=self._iteration,
        )
        original_cwd = getattr(self, "_restore_cwd", None)
        if original_cwd is None:
            try:
                original_cwd = os.getcwd()
                self._restore_cwd = original_cwd
            except FileNotFoundError:
                original_cwd = os.path.expanduser("~") or tempfile.gettempdir()
                self._restore_cwd = original_cwd
        if self.enable_filesystem:
            if not self.workspace_dir.exists():
                self.workspace_dir.mkdir(parents=True, exist_ok=True)
            try:
                os.chdir(self.workspace_dir)
            except FileNotFoundError:
                self.workspace_dir.mkdir(parents=True, exist_ok=True)
                os.chdir(self.workspace_dir)

        try:
            ast.parse(code)
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, self._namespace)
            result.success = True
        except SyntaxError as e:
            result.error = f"SyntaxError: {e}"
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        finally:
            if self.enable_filesystem and original_cwd:
                try:
                    os.chdir(original_cwd)
                except (FileNotFoundError, OSError):
                    pass

        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        combined_output = stdout_output
        if stderr_output:
            combined_output += f"\n[stderr]: {stderr_output}"
        if result.error:
            combined_output += f"\n[error]: {result.error}"

        result.output, result.truncated = self._truncate_output(combined_output)

        # Safely get answer state (model might have overwritten the answer variable)
        answer_val = self._namespace.get('answer', {})
        if isinstance(answer_val, dict):
            result.answer_state = answer_val.copy()
        else:
            # Model overwrote answer with a non-dict value
            result.answer_state = {"content": str(answer_val), "ready": True}

        result.execution_time_ms = int((time.time() - start_time) * 1000)

        if self.enable_filesystem:
            result.files_created, result.files_modified = self._get_file_changes()

        # Include sub-agent calls made during this execution
        result.sub_agent_calls = self._sub_agent_calls.copy()

        # ============================================================
        # RLM-style finalization detection
        # ============================================================

        # Check if FINAL() was called directly (sets self._done)
        if self._done:
            result.done = True
            result.final_answer = self._final_answer

        # Check for print('FINAL(...)') pattern in output
        elif not result.done:
            final_match = re.search(r'FINAL\(([^)]+)\)', stdout_output)
            if final_match:
                result.done = True
                result.final_answer = final_match.group(1).strip()
                self._done = True
                self._final_answer = result.final_answer

        # Check for print('FINAL_VAR(var_name)') pattern
        if not result.done:
            final_var_match = re.search(r'FINAL_VAR\(([^)]+)\)', stdout_output)
            if final_var_match:
                var_name = final_var_match.group(1).strip().strip('"\'')
                if var_name in self._namespace:
                    result.done = True
                    result.final_answer = str(self._namespace[var_name])
                    self._done = True
                    self._final_answer = result.final_answer

        # Check for answer["ready"] = True (Prime Intellect style)
        if not result.done and result.answer_state.get("ready", False):
            result.done = True
            result.final_answer = str(result.answer_state.get("content", ""))
            self._done = True
            self._final_answer = result.final_answer

        # Check for max iterations
        if not result.done and self._iteration >= self._max_iterations:
            result.done = True
            result.reward = self.reward_on_failure

        # ============================================================
        # Reward calculation
        # ============================================================

        if result.done and result.final_answer is not None:
            result.reward = self.reward_on_success
        elif result.error:
            result.reward = self.reward_on_error
        else:
            result.reward = self.reward_on_iteration

        self.history.append((code, result))
        return result

    def execute_bash(self, code: str, timeout: Optional[int] = None) -> ExecutionResult:
        """
        Execute bash commands in the sandbox workspace.

        Args:
            code: Bash commands to execute
            timeout: Optional timeout in seconds (defaults to self.timeout_seconds)

        Returns:
            ExecutionResult with stdout/stderr

        Raises:
            RuntimeError: If bash execution is not enabled
        """
        import subprocess
        import time

        # Check if bash is enabled
        if not self.enable_bash:
            return ExecutionResult(
                success=False,
                output="",
                error="Bash execution is not enabled for this sandbox. Set enable_bash=True to use bash commands.",
                answer_state=self._namespace.get('answer', {}).copy(),
                iteration=self._iteration,
            )

        timeout = timeout or self.timeout_seconds
        start_time = time.time()

        # Increment iteration counter
        self._iteration += 1

        result = ExecutionResult(
            success=False,
            output="",
            answer_state=self._namespace.get('answer', {}).copy(),
            iteration=self._iteration,
        )

        try:
            # Run bash command in workspace directory
            proc = subprocess.run(
                code,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.workspace_dir) if self.enable_filesystem else None,
                env=os.environ.copy()
            )

            # Combine stdout and stderr
            output = proc.stdout
            if proc.stderr:
                output += f"\n[stderr]:\n{proc.stderr}"

            result.output, result.truncated = self._truncate_output(output)
            result.success = (proc.returncode == 0)

            if proc.returncode != 0:
                result.error = f"Exit code: {proc.returncode}"

        except subprocess.TimeoutExpired:
            result.error = f"Command timed out after {timeout}s"
            result.output = "[Timeout]"
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            result.output = ""

        result.execution_time_ms = int((time.time() - start_time) * 1000)

        # Track file changes
        if self.enable_filesystem:
            result.files_created, result.files_modified = self._get_file_changes()

        # Bash commands don't affect Python namespace, but track for history
        self.history.append((f"[bash] {code}", result))

        return result

    def get_answer(self) -> Dict[str, Any]:
        """Get current state of the answer variable."""
        return self._namespace.get('answer', {"content": "", "ready": False}).copy()

    def is_ready(self) -> bool:
        """Check if answer["ready"] is True."""
        answer = self._namespace.get('answer', {})
        return answer.get('ready', False)

    def reset(self, max_iterations: int = 30) -> None:
        """Reset the sandbox to initial state."""
        self._namespace.clear()
        self._init_namespace([])
        self.history.clear()

        # Reset episode state
        self._iteration = 0
        self._max_iterations = max_iterations
        self._done = False
        self._final_answer = None

        if self._owns_workspace and self.workspace_dir.exists():
            for f in self.workspace_dir.iterdir():
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)

    def cleanup(self) -> None:
        """Clean up resources (call when done)."""
        if self._owns_workspace and self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir)

    def get_context_preview(self) -> Optional[str]:
        """Get a preview of the context variable (first N chars)."""
        context = self._namespace.get('context')
        if context is None:
            return None
        return str(context)[:self.context_preview_length]

    def get_context_length(self) -> int:
        """Get the total length of the context variable."""
        context = self._namespace.get('context')
        if context is None:
            return 0
        return len(str(context))

    def get_episode_state(self) -> Dict[str, Any]:
        """Get current episode state for RL training."""
        return {
            "iteration": self._iteration,
            "max_iterations": self._max_iterations,
            "done": self._done,
            "final_answer": self._final_answer,
            "context_length": self.get_context_length(),
            "context_preview": self.get_context_preview(),
            "available_variables": list(self.get_namespace_summary().keys()),
        }

    @property
    def done(self) -> bool:
        """Check if episode is complete."""
        return self._done

    @property
    def final_answer(self) -> Optional[str]:
        """Get the final answer if episode is complete."""
        return self._final_answer

    @property
    def iteration(self) -> int:
        """Get current iteration count."""
        return self._iteration

    def get_namespace_summary(self) -> Dict[str, str]:
        """Get a summary of variables in the namespace."""
        summary = {}
        skip_keys = {'__builtins__', 'sub_agent', 'sub_agent_batch', 'llm_query', 'llm_query_batched',
                     'save_to_file', 'read_file', 'list_files', 'file_exists', 'workspace',
                     'get_file_info', 'search_files',
                     'save_scratch', 'save_output', 'list_scratch', 'list_output',
                     'FINAL', 'FINAL_VAR'}

        for key, value in self._namespace.items():
            if key in skip_keys:
                continue
            try:
                type_name = type(value).__name__
                if isinstance(value, (str, int, float, bool, type(None))):
                    summary[key] = f"{type_name}: {repr(value)[:100]}"
                elif isinstance(value, (list, dict, set, tuple)):
                    summary[key] = f"{type_name}: len={len(value)}"
                else:
                    summary[key] = f"{type_name}"
            except:
                summary[key] = "<?>"
        return summary

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get a comprehensive state snapshot."""
        import inspect
        import types

        skip_keys = {'__builtins__', 'sub_agent', 'sub_agent_batch', 'llm_query', 'llm_query_batched',
                     'save_to_file', 'read_file', 'list_files', 'file_exists', 'workspace', 'context',
                     'get_file_info', 'search_files',
                     'save_scratch', 'save_output', 'list_scratch', 'list_output',
                     'FINAL', 'FINAL_VAR'}

        variables, functions, classes, modules = {}, {}, {}, []

        for key, value in self._namespace.items():
            if key in skip_keys or key.startswith('_'):
                continue

            try:
                if isinstance(value, types.FunctionType):
                    try:
                        sig = str(inspect.signature(value))
                        functions[key] = f"def {key}{sig}"
                    except (ValueError, TypeError):
                        functions[key] = f"def {key}(...)"
                elif isinstance(value, type):
                    methods = [m for m in dir(value) if not m.startswith('_') and callable(getattr(value, m, None))]
                    classes[key] = {"name": key, "methods": methods[:10],
                                   "bases": [b.__name__ for b in value.__bases__ if b.__name__ != 'object']}
                elif isinstance(value, types.ModuleType):
                    modules.append(key)
                else:
                    type_name = type(value).__name__
                    if isinstance(value, (str, int, float, bool, type(None))):
                        val_repr = repr(value)[:80]
                        variables[key] = {"type": type_name, "value": val_repr}
                    elif isinstance(value, (list, tuple)):
                        variables[key] = {"type": type_name, "len": len(value)}
                    elif isinstance(value, dict):
                        # Ensure keys are strings for JSON serialization
                        safe_keys = [str(k) for k in list(value.keys())[:5]]
                        variables[key] = {"type": type_name, "len": len(value), "keys": safe_keys}
                    elif isinstance(value, set):
                        variables[key] = {"type": type_name, "len": len(value)}
                    elif hasattr(value, 'shape'):
                        variables[key] = {"type": type_name, "shape": str(value.shape)}
                    else:
                        variables[key] = {"type": type_name}
            except Exception:
                variables[key] = {"type": "unknown"}

        return {"variables": variables, "functions": functions, "classes": classes, "modules": modules}

    def format_state_for_context(self) -> str:
        """Format state snapshot as a concise string for context injection."""
        state = self.get_state_snapshot()
        parts = []

        if state["modules"]:
            parts.append(f"imports: {', '.join(state['modules'])}")
        if state["functions"]:
            parts.append(f"functions: {'; '.join(state['functions'].values())}")
        if state["classes"]:
            class_strs = []
            for name, info in state["classes"].items():
                bases = f"({', '.join(info['bases'])})" if info['bases'] else ""
                methods = f" [{', '.join(info['methods'][:5])}]" if info['methods'] else ""
                class_strs.append(f"class {name}{bases}{methods}")
            parts.append(f"classes: {'; '.join(class_strs)}")
        if state["variables"]:
            var_strs = []
            for name, info in state["variables"].items():
                if "value" in info:
                    var_strs.append(f"{name}={info['value']}")
                elif "shape" in info:
                    var_strs.append(f"{name}: {info['type']}({info['shape']})")
                elif "len" in info:
                    var_strs.append(f"{name}: {info['type']}[{info['len']}]")
                else:
                    var_strs.append(f"{name}: {info['type']}")
            parts.append(f"vars: {', '.join(var_strs)}")

        # Add workspace files
        if self.enable_filesystem:
            files = self._list_files("*")
            if files:
                parts.append(f"files: {', '.join(files)}")

        # Add episode completion status (signals model to output <final_answer>)
        if self._done:
            parts.append(f"done: True")
            if self._final_answer is not None:
                # Truncate long answers in state display
                answer_preview = self._final_answer[:50] + "..." if len(self._final_answer) > 50 else self._final_answer
                parts.append(f"final_answer: {answer_preview}")

        return " | ".join(parts) if parts else "(empty state)"


class AsyncPythonSandbox(PythonSandbox):
    """Async version of PythonSandbox with timeout support."""

    async def execute_async(self, code: str) -> ExecutionResult:
        """Execute code asynchronously with timeout."""
        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, self.execute, code),
                timeout=self.timeout_seconds
            )
            return result
        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False, output="",
                error=f"Execution timed out after {self.timeout_seconds} seconds",
                answer_state=self.get_answer()
            )
