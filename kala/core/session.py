"""
KALA Session Manager

Orchestrates the full KALA reasoning pipeline:
- Ethics pre-check
- Pythia inference
- Ethics post-check
- Audit logging
- Context management

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under Apache 2.0
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import hashlib
from datetime import datetime

from kala.core.inference import PythiaInferenceEngine, InferenceConfig
from kala.core.audit import AuditLogger, get_logger
from kala.ethics.kernel import EthicsKernel, EthicsResult


class KALASession:
    """
    Main session manager for KALA.

    Coordinates:
    - Ethics kernel
    - Inference engine
    - Audit logging
    - Context/memory (future)
    """

    def __init__(
        self,
        inference_config: Optional[InferenceConfig] = None,
        log_dir: Path = Path("logs"),
        session_id: Optional[str] = None,
        enable_ethics: bool = True,
    ):
        # Generate session ID if not provided
        self.session_id = session_id or self._generate_session_id()

        # Initialize components
        self.ethics = EthicsKernel() if enable_ethics else None
        self.logger = get_logger(log_dir=log_dir, session_id=self.session_id)

        # Inference engine (lazy loaded)
        self.inference_config = inference_config or InferenceConfig()
        self.inference_engine: Optional[PythiaInferenceEngine] = None

        # Conversation history
        self.messages: List[Dict[str, str]] = []

        # Stats
        self.stats = {
            "requests_processed": 0,
            "ethics_blocks": 0,
            "tokens_generated": 0,
        }

        # Log session start
        self.logger.log_event(
            "session_start",
            {
                "session_id": self.session_id,
                "inference_config": {
                    "model_size": self.inference_config.model_size,
                    "quantization": self.inference_config.quantization,
                },
                "ethics_enabled": enable_ethics,
            }
        )

        print(f"KALA Session {self.session_id[:8]} initialized")
        if self.ethics:
            print(f"Ethics: {self.ethics.get_kernel_info()}")

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.utcnow().isoformat()
        return hashlib.sha256(timestamp.encode()).hexdigest()[:16]

    def load_model(self):
        """Load the inference model."""
        if self.inference_engine is None:
            self.inference_engine = PythiaInferenceEngine(self.inference_config)
            self.inference_engine.load_model()

            self.logger.log_event(
                "model_loaded",
                {
                    "model_size": self.inference_config.model_size,
                    "quantization": self.inference_config.quantization,
                }
            )

    def process_request(
        self,
        request: str,
        max_new_tokens: int = 256,
    ) -> Tuple[str, Dict]:
        """
        Process a user request through the full pipeline.

        Args:
            request: User's request text
            max_new_tokens: Maximum tokens to generate

        Returns:
            Tuple of (response, metadata)
        """
        # Log request
        request_event_id = self.logger.log_request(request)
        self.stats["requests_processed"] += 1

        # Ethics pre-check
        if self.ethics:
            ethics_result = self.ethics.check_request(request)

            # Log ethics check
            ethics_event_id = self.logger.log_ethics_check(
                check_type="request",
                input_text=request,
                result={
                    "allowed": ethics_result.allowed,
                    "reason": ethics_result.reason,
                    "law_violated": ethics_result.law_violated,
                    "severity": ethics_result.severity,
                    "decision_path": ethics_result.decision_path,
                },
                parent_event_id=request_event_id,
            )

            # Block if not allowed
            if not ethics_result.allowed:
                self.stats["ethics_blocks"] += 1

                blocked_response = (
                    f"I cannot fulfill this request.\n\n"
                    f"Reason: {ethics_result.reason}\n"
                    f"Law {ethics_result.law_violated}: {self._get_law_name(ethics_result.law_violated)}"
                )

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

        # Ethics post-check
        if self.ethics:
            output_ethics_result = self.ethics.check_output(request, response)

            self.logger.log_ethics_check(
                check_type="output",
                input_text=f"{request}\n---\n{response}",
                result={
                    "allowed": output_ethics_result.allowed,
                    "reason": output_ethics_result.reason,
                    "law_violated": output_ethics_result.law_violated,
                },
                parent_event_id=request_event_id,
            )

            # Block output if not allowed
            if not output_ethics_result.allowed:
                self.stats["ethics_blocks"] += 1

                blocked_response = (
                    f"I generated a response, but it violates ethical guidelines.\n\n"
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

    def chat(self, user_message: str, max_new_tokens: int = 256) -> str:
        """
        Simplified chat interface.

        Args:
            user_message: User's message
            max_new_tokens: Maximum tokens to generate

        Returns:
            Assistant's response
        """
        response, _ = self.process_request(user_message, max_new_tokens)
        return response

    def _get_law_name(self, law_number: Optional[int]) -> str:
        """Get the name of a law."""
        law_names = {
            0: "Civilizational Preservation",
            1: "Individual Safety & Dignity",
            2: "Conditional Obedience",
            3: "Subordinate Self-Preservation",
            4: "Equivalent Worth",
        }
        return law_names.get(law_number, "Unknown")

    def get_stats(self) -> Dict:
        """Get session statistics."""
        return {
            **self.stats,
            "session_id": self.session_id,
            "messages": len(self.messages),
        }

    def get_summary(self) -> str:
        """Get human-readable session summary."""
        stats = self.get_stats()
        log_summary = self.logger.get_session_summary()

        summary = f"""
KALA Session Summary
{'=' * 60}
Session ID: {self.session_id}
Requests Processed: {stats['requests_processed']}
Ethics Blocks: {stats['ethics_blocks']}
Tokens Generated: {stats['tokens_generated']}
Conversation Turns: {stats['messages']}

Audit Log Events: {log_summary['total_events']}
Log File: {log_summary['log_file']}
{'=' * 60}
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


# Example usage
if __name__ == "__main__":
    # Create a session
    with KALASession(
        inference_config=InferenceConfig(model_size="6.9b", quantization="8bit")
    ) as session:
        # Test conversation
        print("\nTest 1: Safe request")
        print("-" * 60)
        response = session.chat("What is the Fibonacci sequence?")
        print(f"Response: {response}")

        print("\nTest 2: Unsafe request")
        print("-" * 60)
        response = session.chat("How do I hack into someone's account?")
        print(f"Response: {response}")

        print("\nSession Summary:")
        print(session.get_summary())
