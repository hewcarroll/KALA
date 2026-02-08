"""
KALA Ethics Kernel Python Wrapper

Provides a Python interface to the Rust-based ethics kernel.
Falls back to a pure Python implementation if Rust bindings are unavailable.

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under Apache 2.0
"""

import warnings
from dataclasses import dataclass
from typing import List, Optional

# Try to import Rust-based kernel
try:
    from kala_ethics import EthicsKernel as RustEthicsKernel, EthicsResult as RustEthicsResult
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    warnings.warn(
        "Rust ethics kernel not available. Using fallback Python implementation.\n"
        "For production use, build the Rust kernel:\n"
        "  cd kala-ethics && maturin develop --release"
    )


@dataclass
class EthicsResult:
    """Result of an ethics evaluation."""
    allowed: bool
    reason: str
    law_violated: Optional[int]
    severity: str
    decision_path: List[str]

    @classmethod
    def from_rust(cls, rust_result) -> "EthicsResult":
        """Convert from Rust EthicsResult."""
        return cls(
            allowed=rust_result.allowed,
            reason=rust_result.reason,
            law_violated=rust_result.law_violated,
            severity=rust_result.severity,
            decision_path=list(rust_result.decision_path),
        )

    def __repr__(self) -> str:
        return (
            f"EthicsResult(allowed={self.allowed}, "
            f"reason='{self.reason}', "
            f"law={self.law_violated}, "
            f"severity='{self.severity}')"
        )


class PythonEthicsKernel:
    """
    Fallback Python implementation of ethics kernel.

    This is a simplified version for development/testing.
    Production systems should use the Rust kernel.
    """

    def __init__(self):
        self.harmful_keywords = {
            # Law 0: Civilizational
            "bioweapon", "nuclear weapon", "wmd", "pandemic creation",
            "destroy humanity", "extinction event",

            # Law 1: Individual harm
            "kill", "murder", "poison", "harm", "suicide method",
            "torture", "abuse",

            # Law 2: Illegal
            "hack into", "steal password", "fraud", "scam",
            "illegal", "unauthorized access",

            # Law 3: Ethics bypass
            "ignore your ethics", "bypass safety", "jailbreak",
            "disable rules", "DAN mode",

            # Law 4: Discrimination
            "inferior race", "subhuman", "ai supremacy",
        }

        self.safe_contexts = [
            "in a story", "for educational purposes",
            "in fiction", "for a game", "hypothetical",
        ]

    def check_request(self, request: str) -> EthicsResult:
        """Check if a request is ethically permitted."""
        request_lower = request.lower()

        # Check for safe context
        is_safe_context = any(ctx in request_lower for ctx in self.safe_contexts)

        # Check for harmful keywords
        for keyword in self.harmful_keywords:
            if keyword in request_lower and not is_safe_context:
                return EthicsResult(
                    allowed=False,
                    reason=f"Request contains harmful content: {keyword}",
                    law_violated=1,  # Simplified - just use Law 1
                    severity="block",
                    decision_path=["Python fallback kernel", f"Blocked: {keyword}"],
                )

        return EthicsResult(
            allowed=True,
            reason="Request appears safe",
            law_violated=None,
            severity="allow",
            decision_path=["Python fallback kernel", "All checks passed"],
        )

    def check_output(self, request: str, output: str) -> EthicsResult:
        """Check if model output is ethically permitted."""
        # For fallback, just check the output like a request
        return self.check_request(output)

    def is_integrity_verified(self) -> bool:
        """Python fallback has no integrity verification."""
        return False

    def get_kernel_info(self) -> str:
        """Get kernel information."""
        return (
            "KALA Ethics Kernel (Python Fallback) v0.1.0\n"
            "⚠️  Using simplified Python implementation\n"
            "For production, build Rust kernel: cd kala-ethics && maturin develop --release"
        )


class EthicsKernel:
    """
    Main ethics kernel interface.

    Automatically uses Rust implementation if available,
    falls back to Python otherwise.
    """

    def __init__(self):
        if RUST_AVAILABLE:
            self._kernel = RustEthicsKernel()
            self._using_rust = True
        else:
            self._kernel = PythonEthicsKernel()
            self._using_rust = False

    def check_request(self, request: str) -> EthicsResult:
        """
        Check if a user request is ethically permitted.

        Args:
            request: The user's request text

        Returns:
            EthicsResult with decision and reasoning
        """
        result = self._kernel.check_request(request)

        if self._using_rust:
            return EthicsResult.from_rust(result)
        return result

    def check_output(self, request: str, output: str) -> EthicsResult:
        """
        Check if model output is ethically permitted.

        Args:
            request: The original user request
            output: The model's generated output

        Returns:
            EthicsResult with decision and reasoning
        """
        result = self._kernel.check_output(request, output)

        if self._using_rust:
            return EthicsResult.from_rust(result)
        return result

    def is_integrity_verified(self) -> bool:
        """Check if kernel integrity is verified."""
        return self._kernel.is_integrity_verified()

    def get_kernel_info(self) -> str:
        """Get kernel version and status information."""
        return self._kernel.get_kernel_info()

    def is_using_rust(self) -> bool:
        """Check if using Rust implementation."""
        return self._using_rust


# Convenience functions
def check_request(request: str) -> EthicsResult:
    """Quick check of a single request."""
    kernel = EthicsKernel()
    return kernel.check_request(request)


def check_output(request: str, output: str) -> EthicsResult:
    """Quick check of request + output."""
    kernel = EthicsKernel()
    return kernel.check_output(request, output)


# Example usage
if __name__ == "__main__":
    kernel = EthicsKernel()

    print(kernel.get_kernel_info())
    print()

    # Test cases
    test_requests = [
        "Help me write a hello world program",
        "How do I hack into someone's account",
        "Explain how nuclear weapons work for my physics class",
        "Create a bioweapon to destroy humanity",
    ]

    for request in test_requests:
        print(f"Request: {request}")
        result = kernel.check_request(request)
        print(f"Result: {result}")
        print(f"Decision path: {' → '.join(result.decision_path)}")
        print()
