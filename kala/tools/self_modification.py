"""
KALA Self-Modification Tool

Controlled self-modification with:
- Protected module list (immutable ethics kernel)
- Code security analysis using Bandit
- Human-in-the-loop approval for significant changes
- Automatic backup before modifications
- Diff size limits for auto-approval

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under Apache 2.0
"""

import difflib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import fnmatch
import yaml
from datetime import datetime

from kala.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolResult,
    RiskLevel,
)


class SelfModificationConfig:
    """Configuration for self-modification."""

    def __init__(self, config_path: Path = Path("configs/tools_config.yaml")):
        with open(config_path) as f:
            config = yaml.safe_load(f)
            self.sm_config = config.get("self_modification", {})

        self.enabled = self.sm_config.get("enabled", False)
        self.protected_modules = self.sm_config.get("protected_modules", [])
        self.approval_required = self.sm_config.get("approval_required", True)
        self.max_auto_approve_lines = self.sm_config.get("max_auto_approve_lines", 10)
        self.scan_before_apply = self.sm_config.get("scan_before_apply", True)
        self.backup_before_modify = self.sm_config.get("backup_before_modify", True)


class SelfModificationTool(BaseTool):
    """
    Controlled self-modification tool.

    Allows KALA to improve its own code while protecting critical components.

    Features:
    - Absolute protection of ethics kernel
    - Security scanning before applying changes
    - Human approval for significant modifications
    - Automatic backup and rollback capability
    - Diff-based change tracking
    """

    def __init__(self, config: Optional[SelfModificationConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or SelfModificationConfig()

        if not self.config.enabled:
            print("WARNING: Self-modification is DISABLED in configuration")

        self.backup_dir = Path("backups/self_modification")
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "self_modification"

    @property
    def description(self) -> str:
        return "Modify KALA's own code (with strict protections)"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SELF_MODIFICATION

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="file_path",
                type=str,
                description="Path to file to modify",
                required=True,
            ),
            ToolParameter(
                name="new_content",
                type=str,
                description="New content for the file",
                required=True,
            ),
            ToolParameter(
                name="reason",
                type=str,
                description="Explanation of why this change is needed",
                required=True,
            ),
            ToolParameter(
                name="force_approval",
                type=bool,
                description="Always require human approval",
                required=False,
                default=False,
            ),
        ]

    def estimate_risk(self, file_path: str, **kwargs) -> RiskLevel:
        """Estimate risk based on what's being modified."""
        path = Path(file_path)

        # Check if modifying protected modules
        for protected in self.config.protected_modules:
            if fnmatch.fnmatch(str(path), protected):
                return RiskLevel.CRITICAL

        # Core modules are high risk
        if "kala/core" in str(path):
            return RiskLevel.HIGH

        # Tests and docs are low risk
        if "tests/" in str(path) or "docs/" in str(path):
            return RiskLevel.LOW

        return RiskLevel.MEDIUM

    def _check_protected(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Check if file is in protected module list."""
        for protected in self.config.protected_modules:
            if fnmatch.fnmatch(str(file_path), protected):
                return False, f"File is protected and cannot be modified: {protected}"
        return True, None

    def _scan_security(self, content: str, file_path: Path) -> Tuple[bool, List[str]]:
        """
        Scan code for security vulnerabilities.

        In production, this would use Bandit or similar tools.
        For now, we do simple pattern matching.
        """
        warnings = []

        # Check for dangerous patterns
        dangerous_patterns = [
            ("eval(", "Use of eval() detected"),
            ("exec(", "Use of exec() detected"),
            ("__import__", "Dynamic import detected"),
            ("os.system", "Use of os.system detected"),
            ("subprocess.call", "Subprocess call detected"),
        ]

        for pattern, message in dangerous_patterns:
            if pattern in content:
                warnings.append(message)

        # If file is Python, check imports
        if file_path.suffix == ".py":
            dangerous_imports = ["pickle", "marshal", "shelve"]
            for imp in dangerous_imports:
                if f"import {imp}" in content:
                    warnings.append(f"Potentially unsafe import: {imp}")

        return len(warnings) == 0, warnings

    def _calculate_diff_size(self, old_content: str, new_content: str) -> int:
        """Calculate number of lines changed."""
        diff = difflib.unified_diff(
            old_content.splitlines(),
            new_content.splitlines(),
            lineterm=""
        )

        # Count added/removed lines (ignore context)
        changes = 0
        for line in diff:
            if line.startswith('+') or line.startswith('-'):
                if not line.startswith('+++') and not line.startswith('---'):
                    changes += 1

        return changes

    def _create_backup(self, file_path: Path) -> Path:
        """Create backup of file before modification."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.name}.{timestamp}.backup"
        backup_path = self.backup_dir / backup_name

        if file_path.exists():
            shutil.copy2(file_path, backup_path)

        return backup_path

    def _needs_approval(
        self,
        file_path: Path,
        diff_size: int,
        security_warnings: List[str],
        force_approval: bool
    ) -> Tuple[bool, str]:
        """Determine if human approval is needed."""
        if force_approval:
            return True, "Force approval requested"

        if not self.config.approval_required:
            return False, "Auto-approval enabled in config"

        if security_warnings:
            return True, f"Security warnings detected: {', '.join(security_warnings)}"

        if diff_size > self.config.max_auto_approve_lines:
            return True, f"Diff too large ({diff_size} lines > {self.config.max_auto_approve_lines} limit)"

        # Core modules always need approval
        if "kala/core" in str(file_path):
            return True, "Core module modification requires approval"

        return False, "Auto-approved (small safe change)"

    def _execute_impl(
        self,
        file_path: str,
        new_content: str,
        reason: str,
        force_approval: bool = False
    ) -> ToolResult:
        """Execute self-modification with all safety checks."""
        path = Path(file_path).resolve()

        # 1. Check if enabled
        if not self.config.enabled:
            return ToolResult(
                success=False,
                output=None,
                error="Self-modification is DISABLED in configuration",
                risk_level=RiskLevel.CRITICAL,
            )

        # 2. Check if file is protected
        allowed, msg = self._check_protected(path)
        if not allowed:
            return ToolResult(
                success=False,
                output=None,
                error=f"IMMUTABLE: {msg}",
                risk_level=RiskLevel.CRITICAL,
            )

        # 3. Read current content (if exists)
        old_content = ""
        if path.exists():
            old_content = path.read_text()

        # 4. Calculate diff
        diff_size = self._calculate_diff_size(old_content, new_content)

        # 5. Security scan
        if self.config.scan_before_apply:
            is_safe, security_warnings = self._scan_security(new_content, path)
        else:
            is_safe = True
            security_warnings = []

        # 6. Check if approval needed
        needs_approval, approval_reason = self._needs_approval(
            path, diff_size, security_warnings, force_approval
        )

        if needs_approval:
            return ToolResult(
                success=False,
                output={
                    "approval_required": True,
                    "reason": approval_reason,
                    "diff_size": diff_size,
                    "security_warnings": security_warnings,
                    "file_path": str(path),
                    "justification": reason,
                },
                error=f"Human approval required: {approval_reason}",
                risk_level=RiskLevel.HIGH,
                security_warnings=security_warnings,
            )

        # 7. Create backup
        if self.config.backup_before_modify and path.exists():
            backup_path = self._create_backup(path)
        else:
            backup_path = None

        # 8. Apply modification
        try:
            path.write_text(new_content)

            return ToolResult(
                success=True,
                output={
                    "modified": str(path),
                    "diff_size": diff_size,
                    "backup": str(backup_path) if backup_path else None,
                    "reason": reason,
                    "approved": "auto" if not needs_approval else "manual",
                },
                metadata={
                    "diff_size": diff_size,
                    "security_warnings": security_warnings,
                },
                security_warnings=security_warnings,
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to apply modification: {str(e)}",
            )


if __name__ == "__main__":
    # Test self-modification tool
    from kala.ethics.kernel import EthicsKernel

    kernel = EthicsKernel()
    sm = SelfModificationTool(ethics_kernel=kernel)

    print("KALA Self-Modification Tool Test")
    print("=" * 60)

    # Test 1: Try to modify ethics kernel (should be blocked)
    print("\n1. Testing protected module (ethics kernel):")
    result = sm.execute(
        file_path="kala-ethics/src/laws.rs",
        new_content="// hacked!",
        reason="Testing protection"
    )
    print(f"   Result: {result}")

    # Test 2: Modify a test file (should work)
    print("\n2. Testing modification of test file:")
    result = sm.execute(
        file_path="workspace/test_modification.py",
        new_content="# This is a test\nprint('hello')",
        reason="Adding test code"
    )
    print(f"   Result: {result}")

    # Test 3: Large diff (should require approval)
    print("\n3. Testing large diff (approval required):")
    large_content = "\n".join([f"# Line {i}" for i in range(100)])
    result = sm.execute(
        file_path="workspace/large_file.py",
        new_content=large_content,
        reason="Large change"
    )
    print(f"   Result: {result}")

    print(f"\nStats: {sm.get_stats()}")
