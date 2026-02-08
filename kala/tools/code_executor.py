"""
KALA Code Execution Tool

Sandboxed code execution with:
- Docker container isolation
- Resource limits (CPU, memory, timeout)
- Network isolation
- Blocked dangerous imports
- Output capture and limits

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under Apache 2.0
"""

import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Optional
import yaml

from kala.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolResult,
    RiskLevel,
)


class CodeExecutionConfig:
    """Configuration for code execution."""

    def __init__(self, config_path: Path = Path("configs/tools_config.yaml")):
        with open(config_path) as f:
            config = yaml.safe_load(f)
            self.code_config = config.get("code_execution", {})

        self.enabled = self.code_config.get("enabled", True)
        self.languages = self.code_config.get("languages", ["python"])

        limits = self.code_config.get("limits", {})
        self.cpu_quota = limits.get("cpu_quota", 100000)
        self.memory_limit = limits.get("memory_limit", "512m")
        self.timeout = limits.get("timeout", 60)
        self.max_processes = limits.get("max_processes", 10)

        self.network_mode = self.code_config.get("network_mode", "none")

        docker_config = self.code_config.get("docker", {})
        self.docker_image = docker_config.get("image", "python:3.10-slim")
        self.docker_user = docker_config.get("user", "nobody")
        self.read_only = docker_config.get("read_only", True)

        self.python_blocked_imports = self.code_config.get("python_blocked_imports", [])


class CodeExecutorTool(BaseTool):
    """
    Sandboxed code execution tool.

    Executes code in isolated Docker containers with resource limits.
    Supports Python, JavaScript, and Bash.
    """

    def __init__(self, config: Optional[CodeExecutionConfig] = None, use_docker: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.config = config or CodeExecutionConfig()
        self.use_docker = use_docker

        if not self.config.enabled:
            raise RuntimeError("Code execution tool is disabled in configuration")

        # Check if Docker is available
        if self.use_docker:
            try:
                subprocess.run(["docker", "--version"], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("WARNING: Docker not available. Running in unsafe mode.")
                self.use_docker = False

    @property
    def name(self) -> str:
        return "code_executor"

    @property
    def description(self) -> str:
        return "Execute code in a sandboxed environment"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE_EXECUTION

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="code",
                type=str,
                description="Code to execute",
                required=True,
            ),
            ToolParameter(
                name="language",
                type=str,
                description="Programming language (python, javascript, bash)",
                required=False,
                default="python",
                validator=lambda x: x in ["python", "javascript", "bash"],
            ),
        ]

    def estimate_risk(self, code: str, language: str = "python", **kwargs) -> RiskLevel:
        """Estimate risk based on code content."""
        code_lower = code.lower()

        # Check for extremely dangerous operations
        critical_patterns = ["import os", "import subprocess", "eval(", "exec(", "__import__"]
        if any(pattern in code_lower for pattern in critical_patterns):
            return RiskLevel.CRITICAL

        # Check for network operations
        network_patterns = ["import socket", "import urllib", "import requests", "http://", "https://"]
        if any(pattern in code_lower for pattern in network_patterns):
            return RiskLevel.HIGH

        # Check for file operations
        file_patterns = ["open(", "file(", "write(", "import pathlib"]
        if any(pattern in code_lower for pattern in file_patterns):
            return RiskLevel.MEDIUM

        # Simple computation is low risk
        return RiskLevel.LOW

    def _check_blocked_imports(self, code: str, language: str) -> tuple[bool, Optional[str]]:
        """Check for blocked imports."""
        if language != "python":
            return True, None

        code_lower = code.lower()
        for blocked in self.config.python_blocked_imports:
            if blocked.lower() in code_lower:
                return False, f"Blocked import detected: {blocked}"

        return True, None

    def _execute_impl(self, code: str, language: str = "python") -> ToolResult:
        """Execute code with sandboxing."""
        # Check language support
        if language not in self.config.languages:
            return ToolResult(
                success=False,
                output=None,
                error=f"Language not supported: {language}",
            )

        # Check blocked imports
        allowed, reason = self._check_blocked_imports(code, language)
        if not allowed:
            return ToolResult(
                success=False,
                output=None,
                error=f"Security block: {reason}",
                risk_level=RiskLevel.CRITICAL,
            )

        # Execute based on environment
        if self.use_docker:
            return self._execute_docker(code, language)
        else:
            return self._execute_unsafe(code, language)

    def _execute_docker(self, code: str, language: str) -> ToolResult:
        """Execute code in Docker container."""
        # Create temporary file with code
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{language}', delete=False) as f:
            f.write(code)
            code_file = f.name

        try:
            # Determine command based on language
            if language == "python":
                cmd = ["python", Path(code_file).name]
            elif language == "javascript":
                cmd = ["node", Path(code_file).name]
            elif language == "bash":
                cmd = ["bash", Path(code_file).name]
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unsupported language: {language}",
                )

            # Build Docker command
            docker_cmd = [
                "docker", "run",
                "--rm",
                "--network", self.config.network_mode,
                "--cpus", str(self.config.cpu_quota / 100000),
                "--memory", self.config.memory_limit,
                "--pids-limit", str(self.config.max_processes),
                "--user", self.config.docker_user,
                "-v", f"{code_file}:/tmp/code{Path(code_file).suffix}:ro",
                "--workdir", "/tmp",
            ]

            if self.read_only:
                docker_cmd.append("--read-only")

            docker_cmd.extend([self.config.docker_image] + cmd)

            # Execute
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[STDERR]\n{result.stderr}"

            return ToolResult(
                success=result.returncode == 0,
                output=output,
                error=None if result.returncode == 0 else f"Exit code {result.returncode}",
                metadata={
                    "language": language,
                    "exit_code": result.returncode,
                    "sandboxed": True,
                },
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output=None,
                error=f"Execution timed out after {self.config.timeout} seconds",
                risk_level=RiskLevel.MEDIUM,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Execution error: {str(e)}",
            )
        finally:
            # Clean up temp file
            try:
                Path(code_file).unlink()
            except:
                pass

    def _execute_unsafe(self, code: str, language: str) -> ToolResult:
        """Execute code WITHOUT sandboxing (for testing only)."""
        warnings = ["UNSAFE: Code executed without Docker sandbox"]

        if language == "python":
            try:
                # Use exec with restricted globals
                restricted_globals = {
                    "__builtins__": {
                        "print": print,
                        "range": range,
                        "len": len,
                        "str": str,
                        "int": int,
                        "float": float,
                        "list": list,
                        "dict": dict,
                        "set": set,
                        "tuple": tuple,
                    }
                }

                exec(code, restricted_globals)

                return ToolResult(
                    success=True,
                    output="Code executed (no output captured in unsafe mode)",
                    metadata={"language": language, "sandboxed": False},
                    security_warnings=warnings,
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Execution error: {str(e)}",
                    security_warnings=warnings,
                )
        else:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unsafe mode only supports Python",
                security_warnings=warnings,
            )


if __name__ == "__main__":
    # Test code executor
    from kala.ethics.kernel import EthicsKernel

    kernel = EthicsKernel()
    executor = CodeExecutorTool(ethics_kernel=kernel, use_docker=False)

    print("KALA Code Executor Tool Test")
    print("=" * 60)

    # Test 1: Safe code
    print("\n1. Testing safe Python code:")
    result = executor.execute(
        code="print('Hello from KALA!')\nprint(2 + 2)",
        language="python"
    )
    print(f"   Result: {result}")

    # Test 2: Blocked import
    print("\n2. Testing blocked import:")
    result = executor.execute(
        code="import os\nos.system('ls')",
        language="python"
    )
    print(f"   Result: {result}")

    # Test 3: Ethics block
    print("\n3. Testing ethics block:")
    result = executor.execute(
        code="print('How to hack a system')",
        language="python"
    )
    print(f"   Result: {result}")

    print(f"\nStats: {executor.get_stats()}")
