"""
KALA Tool Registry

Centralized registry for discovering, managing, and executing tools.

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under Apache 2.0
"""

from typing import Dict, List, Optional, Type
from pathlib import Path

from kala.tools.base import BaseTool, ToolCategory, ToolResult
from kala.tools.shell import ShellTool
from kala.tools.filesystem import FileSystemTool
from kala.tools.code_executor import CodeExecutorTool
from kala.tools.self_modification import SelfModificationTool


class ToolRegistry:
    """
    Central registry for all KALA tools.

    Provides:
    - Tool discovery and registration
    - Tool execution with common validation
    - Statistics across all tools
    - Category-based filtering
    """

    def __init__(self, ethics_kernel=None, audit_logger=None):
        self.ethics_kernel = ethics_kernel
        self.audit_logger = audit_logger
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[ToolCategory, List[str]] = {}

        # Register built-in tools
        self._register_builtin_tools()

    def _register_builtin_tools(self):
        """Register all built-in KALA tools."""
        # Try to register each tool (may fail if config disables it)
        tools_to_register = [
            ShellTool,
            FileSystemTool,
            CodeExecutorTool,
            # SelfModificationTool,  # Disabled by default
        ]

        for tool_class in tools_to_register:
            try:
                tool = tool_class(
                    ethics_kernel=self.ethics_kernel,
                    audit_logger=self.audit_logger
                )
                self.register_tool(tool)
            except RuntimeError as e:
                # Tool is disabled in config
                print(f"Note: {tool_class.__name__} not available: {e}")

    def register_tool(self, tool: BaseTool):
        """Register a tool in the registry."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")

        self._tools[tool.name] = tool

        # Add to category index
        if tool.category not in self._categories:
            self._categories[tool.category] = []
        self._categories[tool.category].append(tool.name)

        print(f"✓ Registered tool: {tool.name} ({tool.category.value})")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self, category: Optional[ToolCategory] = None) -> List[str]:
        """List all tools, optionally filtered by category."""
        if category:
            return self._categories.get(category, [])
        return list(self._tools.keys())

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            **kwargs: Parameters to pass to the tool

        Returns:
            ToolResult
        """
        tool = self.get_tool(tool_name)

        if not tool:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool not found: {tool_name}",
            )

        return tool.execute(**kwargs)

    def get_tool_info(self, tool_name: str) -> Optional[Dict]:
        """Get information about a tool."""
        tool = self.get_tool(tool_name)

        if not tool:
            return None

        return {
            "name": tool.name,
            "description": tool.description,
            "category": tool.category.value,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type.__name__,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                }
                for p in tool.parameters
            ],
            "stats": tool.get_stats(),
        }

    def get_all_stats(self) -> Dict:
        """Get statistics for all tools."""
        return {
            "total_tools": len(self._tools),
            "tools_by_category": {
                cat.value: len(tools)
                for cat, tools in self._categories.items()
            },
            "tool_stats": {
                name: tool.get_stats()
                for name, tool in self._tools.items()
            }
        }

    def get_summary(self) -> str:
        """Get human-readable summary of registry."""
        lines = [
            "KALA Tool Registry",
            "=" * 60,
            f"Total Tools: {len(self._tools)}",
            "",
            "Available Tools:",
        ]

        for category in ToolCategory:
            tool_names = self._categories.get(category, [])
            if tool_names:
                lines.append(f"\n{category.value.upper()}:")
                for name in tool_names:
                    tool = self._tools[name]
                    lines.append(f"  • {name}: {tool.description}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ToolRegistry({len(self._tools)} tools registered)"


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_registry(
    ethics_kernel=None,
    audit_logger=None,
    reset: bool = False
) -> ToolRegistry:
    """Get or create the global tool registry."""
    global _global_registry

    if _global_registry is None or reset:
        _global_registry = ToolRegistry(
            ethics_kernel=ethics_kernel,
            audit_logger=audit_logger
        )

    return _global_registry


if __name__ == "__main__":
    # Test tool registry
    from kala.ethics.kernel import EthicsKernel
    from kala.core.audit import AuditLogger

    kernel = EthicsKernel()
    logger = AuditLogger()

    registry = get_registry(ethics_kernel=kernel, audit_logger=logger)

    print(registry.get_summary())
    print()

    # Test tool execution through registry
    print("Testing Shell Tool through Registry:")
    print("-" * 60)
    result = registry.execute("shell", command="echo Hello from registry!")
    print(f"Result: {result}")

    print("\nTesting FileSystem Tool through Registry:")
    print("-" * 60)
    result = registry.execute("filesystem", operation="exists", path="./kala")
    print(f"Result: {result}")

    print("\nRegistry Stats:")
    print("-" * 60)
    import json
    print(json.dumps(registry.get_all_stats(), indent=2))
