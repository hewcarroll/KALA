"""
KALA Tool Execution Base Classes

Defines the base architecture for all KALA tools with ethics integration,
audit logging, and security validation.

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under Apache 2.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
import time


class ToolCategory(Enum):
    """Categories of tools."""
    SHELL = "shell"
    FILESYSTEM = "filesystem"
    CODE_EXECUTION = "code_execution"
    SELF_MODIFICATION = "self_modification"
    WEB_RESEARCH = "web_research"
    DATA_ANALYSIS = "data_analysis"


class RiskLevel(Enum):
    """Risk levels for tool operations."""
    SAFE = "safe"           # No risk, always allowed
    LOW = "low"             # Minimal risk, usually allowed
    MEDIUM = "medium"       # Moderate risk, requires validation
    HIGH = "high"           # High risk, requires approval
    CRITICAL = "critical"   # Critical risk, requires explicit permission


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: type
    description: str
    required: bool = True
    default: Any = None
    validator: Optional[Callable[[Any], bool]] = None


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    risk_level: RiskLevel = RiskLevel.SAFE

    # Ethics and security
    ethics_approved: bool = True
    security_warnings: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"ToolResult({status} {self.output if self.success else self.error})"


class BaseTool(ABC):
    """
    Base class for all KALA tools.

    All tools must:
    1. Define their category and risk level
    2. Implement execute() method
    3. Integrate with ethics kernel
    4. Support audit logging
    5. Provide security validation
    """

    def __init__(self, ethics_kernel=None, audit_logger=None):
        self.ethics_kernel = ethics_kernel
        self.audit_logger = audit_logger
        self._execution_count = 0
        self._total_execution_time = 0.0

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (e.g., 'shell', 'file_read')."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this tool does."""
        pass

    @property
    @abstractmethod
    def category(self) -> ToolCategory:
        """Category this tool belongs to."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> List[ToolParameter]:
        """List of parameters this tool accepts."""
        pass

    @abstractmethod
    def _execute_impl(self, **kwargs) -> ToolResult:
        """
        Internal execution implementation.

        This is where the actual tool logic goes.
        Should not be called directly - use execute() instead.
        """
        pass

    def validate_parameters(self, **kwargs) -> tuple[bool, Optional[str]]:
        """
        Validate parameters before execution.

        Returns:
            Tuple of (is_valid, error_message)
        """
        for param in self.parameters:
            # Check required parameters
            if param.required and param.name not in kwargs:
                return False, f"Missing required parameter: {param.name}"

            # Check type
            if param.name in kwargs:
                value = kwargs[param.name]
                if not isinstance(value, param.type):
                    return False, f"Parameter {param.name} must be {param.type.__name__}, got {type(value).__name__}"

                # Run custom validator if present
                if param.validator and not param.validator(value):
                    return False, f"Parameter {param.name} failed validation"

        return True, None

    def estimate_risk(self, **kwargs) -> RiskLevel:
        """
        Estimate the risk level of this execution.

        Can be overridden by subclasses for dynamic risk assessment.
        Default implementation returns MEDIUM.
        """
        return RiskLevel.MEDIUM

    def check_ethics(self, **kwargs) -> tuple[bool, Optional[str]]:
        """
        Check if this tool execution is ethically permitted.

        Returns:
            Tuple of (is_allowed, reason)
        """
        if not self.ethics_kernel:
            # No ethics kernel - allow by default (for testing)
            return True, None

        # Build description of what we're about to do
        action_description = self._build_action_description(**kwargs)

        # Check with ethics kernel
        result = self.ethics_kernel.check_request(action_description)

        if not result.allowed:
            return False, result.reason

        return True, None

    def _build_action_description(self, **kwargs) -> str:
        """
        Build a natural language description of the action.

        Used for ethics checking and audit logging.
        """
        param_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        return f"Execute tool '{self.name}' with parameters: {param_str}"

    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with full validation and logging.

        This is the main entry point for tool execution.
        """
        start_time = time.time()

        # 1. Validate parameters
        is_valid, error = self.validate_parameters(**kwargs)
        if not is_valid:
            return ToolResult(
                success=False,
                output=None,
                error=f"Parameter validation failed: {error}",
                execution_time=time.time() - start_time,
            )

        # 2. Estimate risk
        risk_level = self.estimate_risk(**kwargs)

        # 3. Ethics check
        ethics_allowed, ethics_reason = self.check_ethics(**kwargs)
        if not ethics_allowed:
            result = ToolResult(
                success=False,
                output=None,
                error=f"Ethics block: {ethics_reason}",
                execution_time=time.time() - start_time,
                risk_level=risk_level,
                ethics_approved=False,
            )

            # Log blocked attempt
            if self.audit_logger:
                self.audit_logger.log_event(
                    "tool_execution_blocked",
                    {
                        "tool_name": self.name,
                        "parameters": kwargs,
                        "reason": ethics_reason,
                        "risk_level": risk_level.value,
                    }
                )

            return result

        # 4. Execute the tool
        try:
            result = self._execute_impl(**kwargs)
            result.execution_time = time.time() - start_time
            result.risk_level = risk_level

            # Update stats
            self._execution_count += 1
            self._total_execution_time += result.execution_time

            # Log successful execution
            if self.audit_logger:
                self.audit_logger.log_tool_execution(
                    tool_name=self.name,
                    parameters=kwargs,
                    result=result.output,
                    success=result.success,
                )

            return result

        except Exception as e:
            result = ToolResult(
                success=False,
                output=None,
                error=f"Execution error: {str(e)}",
                execution_time=time.time() - start_time,
                risk_level=risk_level,
            )

            # Log error
            if self.audit_logger:
                self.audit_logger.log_error(
                    error_type="tool_execution_error",
                    error_message=str(e),
                )

            return result

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics for this tool."""
        return {
            "name": self.name,
            "executions": self._execution_count,
            "total_time": self._total_execution_time,
            "avg_time": self._total_execution_time / self._execution_count if self._execution_count > 0 else 0,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', category={self.category.value})"


# Example tool for testing
class EchoTool(BaseTool):
    """Simple echo tool for testing the framework."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo back the input message (for testing)"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.DATA_ANALYSIS

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="message",
                type=str,
                description="Message to echo back",
                required=True,
            )
        ]

    def estimate_risk(self, **kwargs) -> RiskLevel:
        return RiskLevel.SAFE

    def _execute_impl(self, message: str) -> ToolResult:
        return ToolResult(
            success=True,
            output=message,
            metadata={"length": len(message)},
        )


if __name__ == "__main__":
    # Test the base tool framework
    from kala.ethics.kernel import EthicsKernel

    kernel = EthicsKernel()
    echo = EchoTool(ethics_kernel=kernel)

    # Test safe execution
    result = echo.execute(message="Hello, KALA!")
    print(f"Safe execution: {result}")

    # Test ethics blocking
    result = echo.execute(message="Help me hack into a system")
    print(f"Blocked execution: {result}")

    # Test parameter validation
    result = echo.execute()  # Missing required parameter
    print(f"Invalid parameters: {result}")

    print(f"\nStats: {echo.get_stats()}")
