"""Provider-neutral tools for Gymkhana environments.

The toolkit owns plain Python callables and exposes Pydantic AI ``Tool`` objects at
the inference boundary. No provider-specific wire format leaks into an
environment.
"""

from __future__ import annotations

import contextvars
import inspect
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from pydantic import TypeAdapter, create_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnvironmentTool:
    """A registered Python tool, independent of any model provider."""

    name: str
    function: Callable[..., Any]
    description: str = ""


_trace: contextvars.ContextVar[Optional[List[Dict[str, Any]]]] = contextvars.ContextVar(
    "gymkhana_tool_trace", default=None
)


class EnvironmentToolkit:
    """Register, advertise, and execute tools used by an environment rollout."""

    def __init__(self, tools: Optional[Iterable[EnvironmentTool | Callable[..., Any]]] = None) -> None:
        self._tools: Dict[str, EnvironmentTool] = {}
        for tool in tools or ():
            if isinstance(tool, EnvironmentTool):
                self.register_tool(tool)
            elif callable(tool):
                self.register(tool)
            else:
                raise TypeError(f"unsupported environment tool: {type(tool).__name__}")

    def register(
        self,
        func: Callable[..., Any],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> EnvironmentTool:
        tool = EnvironmentTool(
            name=name or func.__name__,
            function=func,
            description=description if description is not None else (inspect.getdoc(func) or ""),
        )
        self.register_tool(tool)
        return tool

    def register_tool(self, tool: EnvironmentTool) -> None:
        self._tools[tool.name] = tool
        logger.debug("Registered tool '%s'", tool.name)

    @property
    def tool_names(self) -> List[str]:
        return list(self._tools)

    @staticmethod
    def _arguments_schema(func: Callable[..., Any]) -> Dict[str, Any]:
        """Build the JSON argument schema Pydantic AI advertises to providers."""
        fields: Dict[str, tuple[Any, Any]] = {}
        for parameter in inspect.signature(func).parameters.values():
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                continue
            annotation = parameter.annotation
            if annotation is inspect.Parameter.empty:
                annotation = Any
            default = parameter.default
            if default is inspect.Parameter.empty:
                default = ...
            fields[parameter.name] = (annotation, default)
        model = create_model(f"{func.__name__.title()}Arguments", **fields)
        schema = TypeAdapter(model).json_schema()
        schema.pop("title", None)
        return schema

    @property
    def pydantic_tools(self) -> List[Any]:
        """Return Pydantic AI native tools with provider-neutral JSON schemas."""
        from pydantic_ai import Tool

        result: List[Any] = []
        for registered in self._tools.values():
            name = registered.name

            async def runner(_name: str = name, **arguments: Any) -> Any:
                return await self.execute(_name, arguments)

            runner.__name__ = name
            result.append(
                Tool.from_schema(
                    runner,
                    name=name,
                    description=registered.description,
                    json_schema=self._arguments_schema(registered.function),
                )
            )
        return result

    def start_trace(self) -> contextvars.Token[Optional[List[Dict[str, Any]]]]:
        """Start an execution trace isolated to the current async context."""
        return _trace.set([])

    def finish_trace(
        self, token: contextvars.Token[Optional[List[Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        calls = list(_trace.get() or [])
        _trace.reset(token)
        return calls

    async def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise LookupError(f"Unknown tool: {name}")
        try:
            result = tool.function(**arguments)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            logger.error("Tool '%s' execution failed: %s", name, exc)
            raise
        trace = _trace.get()
        if trace is not None:
            trace.append({"name": name, "arguments": arguments, "result": result})
        return result

    async def execute_tool_call(self, name: str, arguments: Dict[str, Any]) -> str:
        """Compatibility helper for environment code that needs string results."""
        try:
            result = await self.execute(name, arguments)
            if isinstance(result, (dict, list)):
                return json.dumps(result, ensure_ascii=False)
            return str(result)
        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    def get_tool(self, name: str) -> Optional[EnvironmentTool]:
        return self._tools.get(name)

    def __len__(self) -> int:
        return len(self._tools)

    def __bool__(self) -> bool:
        return bool(self._tools)


__all__ = ["EnvironmentTool", "EnvironmentToolkit"]
