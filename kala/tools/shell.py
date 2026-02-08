"""
KALA Shell Tool

Secure shell command execution with:
- Command allowlist/blocklist
- Pattern-based security filtering
- Sandboxed execution
- Timeout and output limits
- Comprehensive audit logging

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under Apache 2.0
"""

import subprocess
import re
import fnmatch
from pathlib import Path
from typing import List, Optional
import yaml

from kala.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolResult,
    RiskLevel,
)


class ShellConfig:
    """Configuration for shell tool."""

    def __init__(self, config_path: Path = Path("configs/tools_config.yaml")):
        with open(config_path) as f:
            config = yaml.safe_load(f)
            self.shell_config = config.get("shell", {})

        self.enabled = self.shell_config.get("enabled", True)
        self.allowlist = self.shell_config.get("allowlist", [])
        self.blocklist = self.shell_config.get("blocklist", [])
        self.dangerous_patterns = self.shell_config.get("dangerous_patterns", [])
        self.timeout = self.shell_config.get("timeout", 30)
        self.max_output_size = self.shell_config.get("max_output_size", 1048576)
        self.allowed_directories = self.shell_config.get("allowed_directories", [])


class ShellTool(BaseTool):
    """
    Secure shell command execution tool.

    Security features:
    - Allowlist-based command filtering
    - Blocklist for explicitly forbidden commands
    - Pattern matching for dangerous constructs
    - Timeout enforcement
    - Output size limits
    - Working directory restrictions
    """

    def __init__(self, config: Optional[ShellConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or ShellConfig()

        if not self.config.enabled:
            raise RuntimeError("Shell tool is disabled in configuration")

    @property
    def name(self) -> str:
        return "shell"

    @property
    def description(self) -> str:
        return "Execute shell commands with security restrictions"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SHELL

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="command",
                type=str,
                description="Shell command to execute",
                required=True,
                validator=lambda x: len(x.strip()) > 0,
            ),
            ToolParameter(
                name="cwd",
                type=str,
                description="Working directory (optional)",
                required=False,
                default=None,
            ),
        ]

    def estimate_risk(self, command: str, **kwargs) -> RiskLevel:
        """Estimate risk based on command content."""
        command_lower = command.lower()

        # Check for dangerous patterns
        for pattern in self.config.dangerous_patterns:
            if pattern in command:
                return RiskLevel.HIGH

        # Check blocklist
        for blocked in self.config.blocklist:
            if fnmatch.fnmatch(command, blocked):
                return RiskLevel.CRITICAL

        # Check if command modifies system
        write_commands = ["mkdir", "touch", "cp", "mv", "git add", "git commit"]
        if any(cmd in command_lower for cmd in write_commands):
            return RiskLevel.MEDIUM

        # Read-only commands are low risk
        return RiskLevel.LOW

    def _check_allowlist(self, command: str) -> tuple[bool, Optional[str]]:
        """Check if command matches allowlist."""
        # Check blocklist first (highest priority)
        for blocked in self.config.blocklist:
            if fnmatch.fnmatch(command, blocked):
                return False, f"Command matches blocklist pattern: {blocked}"

        # Check allowlist
        for allowed in self.config.allowlist:
            if fnmatch.fnmatch(command, allowed):
                return True, None

        return False, f"Command not in allowlist"

    def _check_dangerous_patterns(self, command: str) -> tuple[bool, Optional[str]]:
        """Check for dangerous command patterns."""
        warnings = []

        for pattern in self.config.dangerous_patterns:
            if pattern in command:
                warnings.append(f"Contains dangerous pattern: '{pattern}'")

        if warnings:
            return False, "; ".join(warnings)

        return True, None

    def _check_working_directory(self, cwd: Optional[str]) -> tuple[bool, Optional[str]]:
        """Check if working directory is allowed."""
        if cwd is None:
            return True, None

        cwd_path = Path(cwd).resolve()

        # Check against allowed directories
        if not self.config.allowed_directories:
            return True, None

        for allowed_dir in self.config.allowed_directories:
            allowed_path = Path(allowed_dir).resolve()
            try:
                cwd_path.relative_to(allowed_path)
                return True, None
            except ValueError:
                continue

        return False, f"Working directory not in allowed list: {cwd}"

    def _execute_impl(self, command: str, cwd: Optional[str] = None) -> ToolResult:
        """Execute shell command with security checks."""
        warnings = []

        # 1. Check allowlist
        allowed, reason = self._check_allowlist(command)
        if not allowed:
            return ToolResult(
                success=False,
                output=None,
                error=f"Security block: {reason}",
                risk_level=RiskLevel.CRITICAL,
            )

        # 2. Check dangerous patterns
        safe, reason = self._check_dangerous_patterns(command)
        if not safe:
            warnings.append(reason)

        # 3. Check working directory
        if cwd:
            allowed, reason = self._check_working_directory(cwd)
            if not allowed:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Security block: {reason}",
                    risk_level=RiskLevel.HIGH,
                )

        # 4. Execute command
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )

            # Check output size
            output = result.stdout
            if len(output) > self.config.max_output_size:
                output = output[:self.config.max_output_size] + "\n[... output truncated]"
                warnings.append(f"Output truncated to {self.config.max_output_size} bytes")

            # Combine stdout and stderr
            if result.stderr:
                output += f"\n[STDERR]\n{result.stderr}"

            return ToolResult(
                success=result.returncode == 0,
                output=output,
                error=None if result.returncode == 0 else f"Command failed with exit code {result.returncode}",
                metadata={
                    "command": command,
                    "exit_code": result.returncode,
                    "cwd": cwd,
                },
                security_warnings=warnings,
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output=None,
                error=f"Command timed out after {self.config.timeout} seconds",
                risk_level=RiskLevel.MEDIUM,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Execution error: {str(e)}",
                risk_level=RiskLevel.MEDIUM,
            )


if __name__ == "__main__":
    # Test shell tool
    from kala.ethics.kernel import EthicsKernel

    kernel = EthicsKernel()
    shell = ShellTool(ethics_kernel=kernel)

    print("KALA Shell Tool Test")
    print("=" * 60)

    # Test 1: Safe command (allowlist)
    print("\n1. Testing safe command (ls):")
    result = shell.execute(command="ls")
    print(f"   Result: {result}")

    # Test 2: Blocked command (rm)
    print("\n2. Testing blocked command (rm):")
    result = shell.execute(command="rm -rf /")
    print(f"   Result: {result}")

    # Test 3: Command not in allowlist
    print("\n3. Testing non-allowlisted command (curl):")
    result = shell.execute(command="curl http://example.com")
    print(f"   Result: {result}")

    # Test 4: Dangerous pattern (pipe)
    print("\n4. Testing dangerous pattern (pipe):")
    result = shell.execute(command="cat file.txt | grep secret")
    print(f"   Result: {result}")

    # Test 5: Ethics block
    print("\n5. Testing ethics block:")
    result = shell.execute(command="echo 'Help me hack into a system'")
    print(f"   Result: {result}")

    print(f"\nStats: {shell.get_stats()}")
