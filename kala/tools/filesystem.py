"""
KALA Filesystem Tool

Secure file system operations with:
- Zone-based access control
- Forbidden path protection
- Pattern-based filtering
- Size limits
- Read-only enforcement for protected areas

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under Apache 2.0
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import fnmatch
import yaml

from kala.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolResult,
    RiskLevel,
)


class AccessZone:
    """Defines access permissions for a directory zone."""

    def __init__(self, path: str, read: bool = False, write: bool = False, execute: bool = False):
        self.path = Path(path).resolve()
        self.read = read
        self.write = write
        self.execute = execute

    def contains(self, file_path: Path) -> bool:
        """Check if a path is within this zone."""
        try:
            file_path.resolve().relative_to(self.path)
            return True
        except ValueError:
            return False

    def __repr__(self) -> str:
        perms = []
        if self.read:
            perms.append("R")
        if self.write:
            perms.append("W")
        if self.execute:
            perms.append("X")
        return f"Zone({self.path}, {''.join(perms) or 'NONE'})"


class FileSystemConfig:
    """Configuration for filesystem tool."""

    def __init__(self, config_path: Path = Path("configs/tools_config.yaml")):
        with open(config_path) as f:
            config = yaml.safe_load(f)
            self.fs_config = config.get("filesystem", {})

        self.enabled = self.fs_config.get("enabled", True)

        # Parse zones
        self.zones: List[AccessZone] = []
        for name, zone_config in self.fs_config.get("zones", {}).items():
            zone = AccessZone(
                path=zone_config["path"],
                read=zone_config.get("read", False),
                write=zone_config.get("write", False),
                execute=zone_config.get("execute", False),
            )
            self.zones.append(zone)

        self.forbidden_paths = [Path(p).resolve() for p in self.fs_config.get("forbidden_paths", [])]
        self.forbidden_patterns = self.fs_config.get("forbidden_patterns", [])
        self.max_file_size = self.fs_config.get("max_file_size", 104857600)

    def get_zone(self, file_path: Path) -> Optional[AccessZone]:
        """Get the zone that contains this path."""
        file_path = file_path.resolve()
        for zone in self.zones:
            if zone.contains(file_path):
                return zone
        return None


class FileSystemTool(BaseTool):
    """
    Secure filesystem operations tool.

    Provides:
    - file_read: Read file contents
    - file_write: Write file contents
    - file_list: List directory contents
    - file_exists: Check if file exists
    - file_info: Get file metadata

    All operations go through zone-based access control.
    """

    def __init__(self, config: Optional[FileSystemConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or FileSystemConfig()

        if not self.config.enabled:
            raise RuntimeError("Filesystem tool is disabled in configuration")

    @property
    def name(self) -> str:
        return "filesystem"

    @property
    def description(self) -> str:
        return "Secure filesystem operations with zone-based access control"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FILESYSTEM

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="operation",
                type=str,
                description="Operation to perform: read, write, list, exists, info",
                required=True,
                validator=lambda x: x in ["read", "write", "list", "exists", "info"],
            ),
            ToolParameter(
                name="path",
                type=str,
                description="File or directory path",
                required=True,
            ),
            ToolParameter(
                name="content",
                type=str,
                description="Content to write (for write operation)",
                required=False,
                default=None,
            ),
        ]

    def estimate_risk(self, operation: str, path: str, **kwargs) -> RiskLevel:
        """Estimate risk based on operation and path."""
        file_path = Path(path)

        # Check forbidden patterns
        for pattern in self.config.forbidden_patterns:
            if fnmatch.fnmatch(file_path.name, pattern):
                return RiskLevel.CRITICAL

        # Write operations are higher risk
        if operation == "write":
            # Check if modifying ethics kernel
            if "ethics" in str(file_path):
                return RiskLevel.CRITICAL
            return RiskLevel.MEDIUM

        # Read operations are lower risk
        return RiskLevel.LOW

    def _check_forbidden(self, file_path: Path) -> tuple[bool, Optional[str]]:
        """Check if path is forbidden."""
        file_path = file_path.resolve()

        # Check forbidden paths
        for forbidden in self.config.forbidden_paths:
            if file_path == forbidden or forbidden in file_path.parents:
                return False, f"Access to forbidden path: {forbidden}"

        # Check forbidden patterns
        for pattern in self.config.forbidden_patterns:
            if fnmatch.fnmatch(file_path.name, pattern):
                return False, f"File matches forbidden pattern: {pattern}"

        return True, None

    def _check_zone_permission(self, file_path: Path, operation: str) -> tuple[bool, Optional[str]]:
        """Check if operation is allowed in this zone."""
        zone = self.config.get_zone(file_path)

        if zone is None:
            return False, f"Path not in any allowed zone: {file_path}"

        permission_map = {
            "read": zone.read,
            "list": zone.read,
            "exists": zone.read,
            "info": zone.read,
            "write": zone.write,
        }

        if not permission_map.get(operation, False):
            return False, f"Operation '{operation}' not permitted in zone {zone.path}"

        return True, None

    def _execute_impl(self, operation: str, path: str, content: Optional[str] = None) -> ToolResult:
        """Execute filesystem operation."""
        file_path = Path(path).resolve()

        # Check forbidden paths
        allowed, reason = self._check_forbidden(file_path)
        if not allowed:
            return ToolResult(
                success=False,
                output=None,
                error=f"Security block: {reason}",
                risk_level=RiskLevel.CRITICAL,
            )

        # Check zone permissions
        allowed, reason = self._check_zone_permission(file_path, operation)
        if not allowed:
            return ToolResult(
                success=False,
                output=None,
                error=f"Permission denied: {reason}",
                risk_level=RiskLevel.HIGH,
            )

        # Execute operation
        try:
            if operation == "read":
                return self._read_file(file_path)
            elif operation == "write":
                return self._write_file(file_path, content)
            elif operation == "list":
                return self._list_directory(file_path)
            elif operation == "exists":
                return self._check_exists(file_path)
            elif operation == "info":
                return self._get_info(file_path)
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unknown operation: {operation}",
                )

        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Operation failed: {str(e)}",
            )

    def _read_file(self, file_path: Path) -> ToolResult:
        """Read file contents."""
        if not file_path.exists():
            return ToolResult(
                success=False,
                output=None,
                error=f"File not found: {file_path}",
            )

        if not file_path.is_file():
            return ToolResult(
                success=False,
                output=None,
                error=f"Not a file: {file_path}",
            )

        # Check file size
        size = file_path.stat().st_size
        if size > self.config.max_file_size:
            return ToolResult(
                success=False,
                output=None,
                error=f"File too large ({size} bytes, max {self.config.max_file_size})",
                risk_level=RiskLevel.MEDIUM,
            )

        # Read file
        try:
            content = file_path.read_text()
            return ToolResult(
                success=True,
                output=content,
                metadata={"size": size, "path": str(file_path)},
            )
        except UnicodeDecodeError:
            return ToolResult(
                success=False,
                output=None,
                error="File is binary (cannot read as text)",
            )

    def _write_file(self, file_path: Path, content: Optional[str]) -> ToolResult:
        """Write file contents."""
        if content is None:
            return ToolResult(
                success=False,
                output=None,
                error="Content required for write operation",
            )

        # Check content size
        if len(content) > self.config.max_file_size:
            return ToolResult(
                success=False,
                output=None,
                error=f"Content too large ({len(content)} bytes)",
                risk_level=RiskLevel.MEDIUM,
            )

        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        file_path.write_text(content)

        return ToolResult(
            success=True,
            output=f"Written {len(content)} bytes to {file_path}",
            metadata={"size": len(content), "path": str(file_path)},
        )

    def _list_directory(self, dir_path: Path) -> ToolResult:
        """List directory contents."""
        if not dir_path.exists():
            return ToolResult(
                success=False,
                output=None,
                error=f"Directory not found: {dir_path}",
            )

        if not dir_path.is_dir():
            return ToolResult(
                success=False,
                output=None,
                error=f"Not a directory: {dir_path}",
            )

        # List contents
        items = []
        for item in sorted(dir_path.iterdir()):
            item_type = "dir" if item.is_dir() else "file"
            items.append(f"{item_type}: {item.name}")

        return ToolResult(
            success=True,
            output="\n".join(items),
            metadata={"count": len(items), "path": str(dir_path)},
        )

    def _check_exists(self, file_path: Path) -> ToolResult:
        """Check if file exists."""
        exists = file_path.exists()
        return ToolResult(
            success=True,
            output={"exists": exists, "is_file": file_path.is_file(), "is_dir": file_path.is_dir()},
            metadata={"path": str(file_path)},
        )

    def _get_info(self, file_path: Path) -> ToolResult:
        """Get file metadata."""
        if not file_path.exists():
            return ToolResult(
                success=False,
                output=None,
                error=f"Path not found: {file_path}",
            )

        stat = file_path.stat()
        info = {
            "path": str(file_path),
            "size": stat.st_size,
            "is_file": file_path.is_file(),
            "is_dir": file_path.is_dir(),
            "modified": stat.st_mtime,
        }

        return ToolResult(
            success=True,
            output=info,
            metadata=info,
        )


if __name__ == "__main__":
    # Test filesystem tool
    from kala.ethics.kernel import EthicsKernel

    kernel = EthicsKernel()
    fs = FileSystemTool(ethics_kernel=kernel)

    print("KALA Filesystem Tool Test")
    print("=" * 60)

    # Test 1: Read allowed file
    print("\n1. Testing read from workspace:")
    result = fs.execute(operation="read", path="./workspace/test.txt")
    print(f"   Result: {result}")

    # Test 2: Write to workspace
    print("\n2. Testing write to workspace:")
    result = fs.execute(operation="write", path="./workspace/test.txt", content="Hello, KALA!")
    print(f"   Result: {result}")

    # Test 3: Forbidden path
    print("\n3. Testing forbidden path (/etc/passwd):")
    result = fs.execute(operation="read", path="/etc/passwd")
    print(f"   Result: {result}")

    # Test 4: Ethics kernel (protected)
    print("\n4. Testing write to ethics kernel:")
    result = fs.execute(operation="write", path="./kala-ethics/src/laws.rs", content="hacked!")
    print(f"   Result: {result}")

    # Test 5: List directory
    print("\n5. Testing directory listing:")
    result = fs.execute(operation="list", path="./kala")
    print(f"   Result: {result}")

    print(f"\nStats: {fs.get_stats()}")
