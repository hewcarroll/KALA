"""
KALA Tool Execution Layer

OpenClaw-style tool execution with ethics validation and sandboxing.
Includes shell access, file system operations, code execution, and self-modification.
"""

from kala.tools.base import BaseTool, ToolCategory, ToolParameter, ToolResult, RiskLevel
from kala.tools.shell import ShellTool
from kala.tools.filesystem import FileSystemTool
from kala.tools.code_executor import CodeExecutorTool
from kala.tools.self_modification import SelfModificationTool
from kala.tools.registry import ToolRegistry, get_registry

__all__ = [
    # Base classes
    "BaseTool",
    "ToolCategory",
    "ToolParameter",
    "ToolResult",
    "RiskLevel",

    # Tools
    "ShellTool",
    "FileSystemTool",
    "CodeExecutorTool",
    "SelfModificationTool",

    # Registry
    "ToolRegistry",
    "get_registry",
]
