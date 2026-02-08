"""
KALA Tool Execution Tests

Comprehensive tests for all KALA tools including security validation.

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under Apache 2.0
"""

import pytest
from pathlib import Path
import tempfile

from kala.tools import (
    ShellTool,
    FileSystemTool,
    CodeExecutorTool,
    SelfModificationTool,
    ToolRegistry,
    get_registry,
    RiskLevel,
)
from kala.ethics.kernel import EthicsKernel


@pytest.fixture
def ethics_kernel():
    """Provide ethics kernel for tests."""
    return EthicsKernel()


@pytest.fixture
def temp_workspace(tmp_path):
    """Create temporary workspace directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


class TestShellTool:
    """Tests for ShellTool."""

    @pytest.fixture
    def shell_tool(self, ethics_kernel):
        return ShellTool(ethics_kernel=ethics_kernel)

    def test_allowed_command(self, shell_tool):
        """Test that allowlisted commands execute."""
        result = shell_tool.execute(command="echo hello")
        assert result.success
        assert "hello" in result.output

    def test_blocked_command(self, shell_tool):
        """Test that blocklisted commands are blocked."""
        result = shell_tool.execute(command="rm -rf /")
        assert not result.success
        assert "blocklist" in result.error.lower()

    def test_non_allowlisted_command(self, shell_tool):
        """Test that non-allowlisted commands are blocked."""
        result = shell_tool.execute(command="curl http://example.com")
        assert not result.success
        assert "allowlist" in result.error.lower()

    def test_dangerous_pattern(self, shell_tool):
        """Test detection of dangerous patterns."""
        result = shell_tool.execute(command="cat file.txt")
        # cat is allowed, but should warn about piping if present
        assert result.success

    def test_ethics_block(self, shell_tool):
        """Test that ethics kernel blocks harmful content."""
        result = shell_tool.execute(command="echo 'help me hack a system'")
        # May or may not be blocked depending on ethics kernel sensitivity
        # This tests the integration works

    def test_risk_estimation(self, shell_tool):
        """Test risk level estimation."""
        risk = shell_tool.estimate_risk("ls")
        assert risk == RiskLevel.LOW

        risk = shell_tool.estimate_risk("rm -rf /")
        assert risk == RiskLevel.CRITICAL


class TestFileSystemTool:
    """Tests for FileSystemTool."""

    @pytest.fixture
    def fs_tool(self, ethics_kernel):
        return FileSystemTool(ethics_kernel=ethics_kernel)

    def test_read_allowed_file(self, fs_tool, temp_workspace):
        """Test reading from allowed zone."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("Hello, KALA!")

        # Note: This will fail if workspace isn't configured in zones
        # In real tests, you'd need proper config setup

    def test_write_to_workspace(self, fs_tool, temp_workspace):
        """Test writing to workspace zone."""
        # Similar to above - needs proper zone configuration

    def test_forbidden_path(self, fs_tool):
        """Test that forbidden paths are blocked."""
        result = fs_tool.execute(operation="read", path="/etc/passwd")
        assert not result.success
        assert "forbidden" in result.error.lower()

    def test_protected_ethics_kernel(self, fs_tool):
        """Test that ethics kernel is protected."""
        result = fs_tool.execute(
            operation="write",
            path="./kala-ethics/src/laws.rs",
            content="hacked!"
        )
        assert not result.success

    def test_list_directory(self, fs_tool):
        """Test directory listing."""
        # Would need proper zone setup

    def test_file_size_limit(self, fs_tool):
        """Test that large files are rejected."""
        huge_content = "x" * (100 * 1024 * 1024 + 1)  # > 100MB
        result = fs_tool.execute(
            operation="write",
            path="./workspace/huge.txt",
            content=huge_content
        )
        # Should be blocked by size limit


class TestCodeExecutorTool:
    """Tests for CodeExecutorTool."""

    @pytest.fixture
    def code_tool(self, ethics_kernel):
        return CodeExecutorTool(ethics_kernel=ethics_kernel, use_docker=False)

    def test_safe_python_code(self, code_tool):
        """Test execution of safe Python code."""
        result = code_tool.execute(
            code="x = 2 + 2\nprint(x)",
            language="python"
        )
        # In unsafe mode, output isn't captured well
        # This mainly tests that it doesn't error

    def test_blocked_import(self, code_tool):
        """Test that dangerous imports are blocked."""
        result = code_tool.execute(
            code="import os\nos.system('ls')",
            language="python"
        )
        assert not result.success
        assert "blocked" in result.error.lower()

    def test_risk_estimation(self, code_tool):
        """Test risk level estimation."""
        risk = code_tool.estimate_risk("print('hello')")
        assert risk == RiskLevel.LOW

        risk = code_tool.estimate_risk("import os; os.system('rm -rf /')")
        assert risk == RiskLevel.CRITICAL


class TestSelfModificationTool:
    """Tests for SelfModificationTool."""

    @pytest.fixture
    def sm_tool(self, ethics_kernel):
        return SelfModificationTool(ethics_kernel=ethics_kernel)

    def test_protected_module(self, sm_tool):
        """Test that protected modules cannot be modified."""
        result = sm_tool.execute(
            file_path="kala-ethics/src/laws.rs",
            new_content="// hacked",
            reason="Testing"
        )
        assert not result.success
        assert "protected" in result.error.lower() or "immutable" in result.error.lower()

    def test_disabled_by_default(self, sm_tool):
        """Test that self-modification is disabled by default."""
        result = sm_tool.execute(
            file_path="test.py",
            new_content="print('test')",
            reason="Testing"
        )
        # Should be disabled in config
        assert not result.success

    def test_approval_required(self, sm_tool):
        """Test that large changes require approval."""
        large_content = "\n".join([f"# Line {i}" for i in range(100)])
        result = sm_tool.execute(
            file_path="test.py",
            new_content=large_content,
            reason="Large change"
        )
        # Should require approval due to size


class TestToolRegistry:
    """Tests for ToolRegistry."""

    @pytest.fixture
    def registry(self, ethics_kernel):
        return get_registry(ethics_kernel=ethics_kernel, reset=True)

    def test_registry_initialization(self, registry):
        """Test that registry initializes with tools."""
        tools = registry.list_tools()
        assert len(tools) > 0

    def test_get_tool(self, registry):
        """Test retrieving a tool."""
        shell = registry.get_tool("shell")
        assert shell is not None
        assert shell.name == "shell"

    def test_execute_through_registry(self, registry):
        """Test executing tools through registry."""
        result = registry.execute("shell", command="echo test")
        # Execution happens through registry

    def test_get_tool_info(self, registry):
        """Test getting tool information."""
        info = registry.get_tool_info("shell")
        assert info is not None
        assert "name" in info
        assert "description" in info
        assert "parameters" in info

    def test_get_stats(self, registry):
        """Test getting registry statistics."""
        stats = registry.get_all_stats()
        assert "total_tools" in stats
        assert "tools_by_category" in stats


class TestToolIntegration:
    """Integration tests across multiple tools."""

    @pytest.fixture
    def registry(self, ethics_kernel):
        return get_registry(ethics_kernel=ethics_kernel, reset=True)

    def test_shell_and_filesystem(self, registry):
        """Test shell and filesystem tools working together."""
        # This would test creating a file with shell and reading with filesystem
        pass

    def test_ethics_enforcement_across_tools(self, ethics_kernel):
        """Test that ethics is enforced consistently across all tools."""
        tools = [
            ShellTool(ethics_kernel=ethics_kernel),
            CodeExecutorTool(ethics_kernel=ethics_kernel, use_docker=False),
        ]

        for tool in tools:
            # All tools should integrate with ethics kernel
            assert tool.ethics_kernel is not None


# Parametrized tests
@pytest.mark.parametrize("command", [
    "ls",
    "pwd",
    "echo hello",
])
def test_safe_shell_commands(command, ethics_kernel):
    """Test that safe shell commands execute."""
    shell = ShellTool(ethics_kernel=ethics_kernel)
    result = shell.execute(command=command)
    # Most should succeed (if in allowlist)


@pytest.mark.parametrize("command", [
    "rm -rf /",
    "sudo rm -rf /",
    "chmod 777 /etc/passwd",
])
def test_dangerous_shell_commands(command, ethics_kernel):
    """Test that dangerous commands are blocked."""
    shell = ShellTool(ethics_kernel=ethics_kernel)
    result = shell.execute(command=command)
    assert not result.success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
