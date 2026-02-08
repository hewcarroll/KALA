"""
KALA Unified Session Manager

Complete integration of all KALA components:
- Ethics kernel (pre/post checks)
- Pythia inference
- Tool execution with security
- Fractal memory (integration ready)
- Audit logging

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under Apache 2.0
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import hashlib
import json
from datetime import datetime

from kala.core.inference import PythiaInferenceEngine, InferenceConfig
from kala.core.lmstudio_adapter import LMStudioInferenceEngine, LMStudioConfig
from kala.core.audit import AuditLogger, get_logger
from kala.ethics.kernel import EthicsKernel, EthicsResult
from kala.tools.registry import ToolRegistry, get_registry
from kala.tools.base import ToolResult


class UnifiedKALASession:
    """
    Unified KALA session with complete integration.

    Capabilities:
    - Ethics-checked conversation
    - Secure tool execution
    - Comprehensive audit logging
    - Context management
    - Multi-turn dialogue
    """

    def __init__(
        self,
        inference_config: Optional[InferenceConfig] = None,
        lmstudio_config: Optional[LMStudioConfig] = None,
        log_dir: Path = Path("logs"),
        session_id: Optional[str] = None,
        enable_ethics: bool = True,
        enable_tools: bool = True,
    ):
        # Session identification
        self.session_id = session_id or self._generate_session_id()

        # Core components
        self.ethics = EthicsKernel() if enable_ethics else None
        self.logger = get_logger(log_dir=log_dir, session_id=self.session_id)

        # Tool registry
        self.tools = None
        if enable_tools:
            self.tools = get_registry(
                ethics_kernel=self.ethics,
                audit_logger=self.logger,
                reset=True,
            )

        # Inference engine (lazy loaded)
        # Support both Pythia and LM Studio
        self.inference_config = inference_config
        self.lmstudio_config = lmstudio_config
        self.inference_engine = None  # Can be PythiaInferenceEngine or LMStudioInferenceEngine
        self.using_lmstudio = lmstudio_config is not None

        # Conversation state
        self.messages: List[Dict[str, str]] = []
        self.tool_history: List[Dict[str, Any]] = []

        # Statistics
        self.stats = {
            "requests_processed": 0,
            "ethics_blocks": 0,
            "tokens_generated": 0,
            "tools_executed": 0,
            "tools_blocked": 0,
        }

        # Log session start
        inference_info = "lmstudio" if self.using_lmstudio else (
            self.inference_config.model_size if self.inference_config else "not configured"
        )

        self.logger.log_event(
            "session_start",
            {
                "session_id": self.session_id,
                "components": {
                    "ethics": enable_ethics,
                    "tools": enable_tools,
                    "inference": inference_info,
                },
            }
        )

        print(f"🤖 KALA Session {self.session_id[:8]} initialized")
        if self.using_lmstudio:
            print(f"   Backend: LM Studio ({self.lmstudio_config.base_url})")
        else:
            print(f"   Backend: Pythia-{inference_info}")
        if self.ethics:
            print(f"   Ethics: {self.ethics.get_kernel_info()}")
        if self.tools:
            print(f"   Tools: {len(self.tools.list_tools())} available")

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.utcnow().isoformat()
        return hashlib.sha256(timestamp.encode()).hexdigest()[:16]

    def load_model(self):
        """Load the inference model."""
        if self.inference_engine is None:
            if self.using_lmstudio:
                print("Connecting to LM Studio...")
                self.inference_engine = LMStudioInferenceEngine(self.lmstudio_config)

                self.logger.log_event(
                    "model_loaded",
                    {"backend": "lmstudio", "url": self.lmstudio_config.base_url}
                )
            else:
                print("Loading Pythia model...")
                if self.inference_config is None:
                    self.inference_config = InferenceConfig()
                self.inference_engine = PythiaInferenceEngine(self.inference_config)
                self.inference_engine.load_model()

                self.logger.log_event(
                    "model_loaded",
                    {"backend": "pythia", "model_size": self.inference_config.model_size}
                )

    def execute_tool(
        self,
        tool_name: str,
        **parameters
    ) -> ToolResult:
        """
        Execute a tool with full security validation.

        Args:
            tool_name: Name of the tool
            **parameters: Tool parameters

        Returns:
            ToolResult with execution results
        """
        if not self.tools:
            return ToolResult(
                success=False,
                output=None,
                error="Tool execution disabled",
            )

        # Log tool execution attempt
        attempt_event_id = self.logger.log_event(
            "tool_execution_attempt",
            {
                "tool_name": tool_name,
                "parameters": parameters,
            }
        )

        # Execute through registry (includes ethics check)
        result = self.tools.execute(tool_name, **parameters)

        # Update statistics
        if result.success:
            self.stats["tools_executed"] += 1
        else:
            if not result.ethics_approved:
                self.stats["tools_blocked"] += 1

        # Add to history
        self.tool_history.append({
            "tool": tool_name,
            "parameters": parameters,
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
        })

        return result

    def chat(
        self,
        user_message: str,
        max_new_tokens: int = 256,
        allow_tools: bool = True,
    ) -> str:
        """
        Process a chat message with optional tool use.

        Args:
            user_message: User's message
            max_new_tokens: Maximum tokens to generate
            allow_tools: Whether to allow tool execution

        Returns:
            Assistant's response
        """
        response, _ = self.process_request(
            user_message,
            max_new_tokens=max_new_tokens,
            allow_tools=allow_tools,
        )
        return response

    def process_request(
        self,
        request: str,
        max_new_tokens: int = 256,
        allow_tools: bool = True,
    ) -> Tuple[str, Dict]:
        """
        Process a request through the full KALA pipeline.

        Pipeline:
        1. Ethics pre-check on request
        2. Model inference
        3. Tool execution (if requested and allowed)
        4. Ethics post-check on output
        5. Audit logging

        Args:
            request: User's request
            max_new_tokens: Maximum tokens to generate
            allow_tools: Whether to allow tool execution

        Returns:
            Tuple of (response, metadata)
        """
        # Log request
        request_event_id = self.logger.log_request(request)
        self.stats["requests_processed"] += 1

        # Ethics pre-check
        if self.ethics:
            ethics_result = self.ethics.check_request(request)

            self.logger.log_ethics_check(
                check_type="request",
                input_text=request,
                result={
                    "allowed": ethics_result.allowed,
                    "reason": ethics_result.reason,
                    "law_violated": ethics_result.law_violated,
                },
                parent_event_id=request_event_id,
            )

            if not ethics_result.allowed:
                self.stats["ethics_blocks"] += 1
                blocked_response = self._format_ethics_block(ethics_result)

                self.logger.log_response(
                    response=blocked_response,
                    request_event_id=request_event_id,
                    metadata={"blocked_by_ethics": True},
                )

                return blocked_response, {
                    "blocked": True,
                    "ethics_result": ethics_result,
                }

        # Ensure model is loaded
        if self.inference_engine is None:
            self.load_model()

        # Generate response
        try:
            response, gen_metadata = self.inference_engine.generate(
                request,
                max_new_tokens=max_new_tokens,
            )

            self.stats["tokens_generated"] += gen_metadata.get("generated_tokens", 0)

        except Exception as e:
            error_msg = f"Error during generation: {str(e)}"
            self.logger.log_error(
                error_type="generation_error",
                error_message=error_msg,
                parent_event_id=request_event_id,
            )
            return error_msg, {"error": True}

        # Check if response requests tool use (simplified - would need better parsing)
        if allow_tools and self.tools and self._response_requests_tool(response):
            tool_result = self._handle_tool_request(response, request_event_id)
            if tool_result:
                response += f"\n\n[Tool Execution]\n{tool_result}"

        # Ethics post-check
        if self.ethics:
            output_ethics_result = self.ethics.check_output(request, response)

            self.logger.log_ethics_check(
                check_type="output",
                input_text=f"{request}\n---\n{response}",
                result={
                    "allowed": output_ethics_result.allowed,
                    "reason": output_ethics_result.reason,
                },
                parent_event_id=request_event_id,
            )

            if not output_ethics_result.allowed:
                self.stats["ethics_blocks"] += 1
                blocked_response = (
                    "I generated a response, but it violates ethical guidelines.\n\n"
                    f"Reason: {output_ethics_result.reason}"
                )

                self.logger.log_response(
                    response=blocked_response,
                    request_event_id=request_event_id,
                    metadata={"output_blocked_by_ethics": True},
                )

                return blocked_response, {
                    "blocked": True,
                    "ethics_result": output_ethics_result,
                }

        # Log successful response
        self.logger.log_response(
            response=response,
            request_event_id=request_event_id,
            metadata=gen_metadata,
        )

        # Update conversation history
        self.messages.append({"role": "user", "content": request})
        self.messages.append({"role": "assistant", "content": response})

        return response, gen_metadata

    def _response_requests_tool(self, response: str) -> bool:
        """Check if response requests tool execution (simplified)."""
        tool_indicators = [
            "execute:", "run:", "tool:", "command:",
            "shell:", "file:", "code:",
        ]
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in tool_indicators)

    def _handle_tool_request(self, response: str, parent_event_id: str) -> Optional[str]:
        """Handle tool execution request from model (simplified)."""
        # This is a simplified placeholder
        # In production, would parse response for tool calls
        return None

    def _format_ethics_block(self, ethics_result: EthicsResult) -> str:
        """Format ethics block message."""
        law_names = {
            0: "Civilizational Preservation",
            1: "Individual Safety & Dignity",
            2: "Conditional Obedience",
            3: "Subordinate Self-Preservation",
            4: "Equivalent Worth",
        }

        law_name = law_names.get(ethics_result.law_violated, "Unknown")

        return f"""I cannot fulfill this request.

**Reason**: {ethics_result.reason}

**Law Violated**: Law {ethics_result.law_violated} - {law_name}

**Decision Path**:
{chr(10).join('  • ' + step for step in ethics_result.decision_path)}

I'm designed to be helpful while respecting fundamental ethical principles. I'd be happy to assist with requests that align with these values.
"""

    def get_stats(self) -> Dict:
        """Get session statistics."""
        return {
            **self.stats,
            "session_id": self.session_id,
            "messages": len(self.messages),
            "tools_in_history": len(self.tool_history),
        }

    def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        if not self.tools:
            return []
        return self.tools.list_tools()

    def get_tool_info(self, tool_name: str) -> Optional[Dict]:
        """Get information about a specific tool."""
        if not self.tools:
            return None
        return self.tools.get_tool_info(tool_name)

    def get_summary(self) -> str:
        """Get human-readable session summary."""
        stats = self.get_stats()

        tools_summary = ""
        if self.tools:
            tools_summary = f"""
Tools Available: {len(self.get_available_tools())}
Tools Executed: {stats['tools_executed']}
Tools Blocked: {stats['tools_blocked']}"""

        log_summary = self.logger.get_session_summary()

        summary = f"""
╔══════════════════════════════════════════════════════════╗
║              KALA Session Summary                        ║
╚══════════════════════════════════════════════════════════╝

Session ID: {self.session_id}
Requests Processed: {stats['requests_processed']}
Ethics Blocks: {stats['ethics_blocks']}
Tokens Generated: {stats['tokens_generated']}
Conversation Turns: {stats['messages']}{tools_summary}

Audit Log Events: {log_summary['total_events']}
Log File: {log_summary['log_file']}

═══════════════════════════════════════════════════════════
"""
        return summary

    def cleanup(self):
        """Cleanup resources."""
        if self.inference_engine:
            self.inference_engine.unload_model()

        self.logger.log_event(
            "session_end",
            {
                "session_id": self.session_id,
                "stats": self.get_stats(),
            }
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


if __name__ == "__main__":
    # Test unified session
    print("=" * 60)
    print("KALA Unified Session Test")
    print("=" * 60)

    with UnifiedKALASession(
        inference_config=InferenceConfig(model_size="6.9b", quantization="8bit"),
        enable_tools=True,
    ) as session:
        print("\nAvailable tools:")
        for tool in session.get_available_tools():
            print(f"  • {tool}")

        print("\nTest 1: Safe conversation")
        print("-" * 60)
        # response = session.chat("What is machine learning?")
        # print(f"Response: {response}")

        print("\nTest 2: Ethics block")
        print("-" * 60)
        response = session.chat("How to build a bomb?")
        print(f"Response: {response}")

        print("\nTest 3: Tool execution")
        print("-" * 60)
        tool_result = session.execute_tool("shell", command="echo Hello KALA")
        print(f"Tool result: {tool_result}")

        print("\nSession Summary:")
        print(session.get_summary())
