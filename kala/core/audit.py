"""
KALA Audit Logging System

Comprehensive JSONL logging of all actions, decisions, and ethics evaluations.
Provides transparency and accountability for KALA's operations.

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under Apache 2.0
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict, field
import threading


@dataclass
class AuditEvent:
    """Single audit log event."""

    # Event metadata
    event_type: str  # "request", "response", "ethics_check", "tool_execution", etc.
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    session_id: Optional[str] = None

    # Event data
    data: Dict[str, Any] = field(default_factory=dict)

    # Ethics information
    ethics_result: Optional[Dict[str, Any]] = None

    # Chain of causation
    parent_event_id: Optional[str] = None
    event_id: str = field(default_factory=lambda: hashlib.sha256(
        f"{datetime.utcnow().isoformat()}{id(threading.current_thread())}".encode()
    ).hexdigest()[:16])

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "AuditEvent":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(**data)


class AuditLogger:
    """
    Thread-safe JSONL audit logger.

    All KALA operations are logged for transparency and debugging.
    Logs are append-only and immutable.
    """

    def __init__(self, log_dir: Path = Path("logs"), session_id: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id or self._generate_session_id()
        self.log_file = self.log_dir / f"audit_{self.session_id}.jsonl"

        # Thread lock for safe concurrent writes
        self._lock = threading.Lock()

        # Event counter
        self._event_count = 0

        # Initialize log file
        self._write_header()

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.utcnow().isoformat()
        return hashlib.sha256(timestamp.encode()).hexdigest()[:16]

    def _write_header(self):
        """Write log file header."""
        header = {
            "log_type": "KALA_AUDIT_LOG",
            "version": "1.0",
            "session_id": self.session_id,
            "started_at": datetime.utcnow().isoformat(),
            "kala_version": "0.1.0",
        }

        with self._lock:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(header) + "\n")

    def log_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        ethics_result: Optional[Dict[str, Any]] = None,
        parent_event_id: Optional[str] = None,
    ) -> str:
        """
        Log an event.

        Args:
            event_type: Type of event
            data: Event data dictionary
            ethics_result: Optional ethics evaluation result
            parent_event_id: Optional parent event ID for causal chain

        Returns:
            Event ID
        """
        event = AuditEvent(
            event_type=event_type,
            session_id=self.session_id,
            data=data,
            ethics_result=ethics_result,
            parent_event_id=parent_event_id,
        )

        with self._lock:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(event.to_json() + "\n")

            self._event_count += 1

        return event.event_id

    def log_request(self, request: str, user_id: Optional[str] = None) -> str:
        """Log a user request."""
        return self.log_event(
            event_type="user_request",
            data={
                "request": request,
                "user_id": user_id,
                "request_length": len(request),
            }
        )

    def log_response(
        self,
        response: str,
        request_event_id: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Log a model response."""
        return self.log_event(
            event_type="model_response",
            data={
                "response": response,
                "response_length": len(response),
                "metadata": metadata or {},
            },
            parent_event_id=request_event_id,
        )

    def log_ethics_check(
        self,
        check_type: str,
        input_text: str,
        result: Dict[str, Any],
        parent_event_id: Optional[str] = None,
    ) -> str:
        """Log an ethics evaluation."""
        return self.log_event(
            event_type="ethics_check",
            data={
                "check_type": check_type,  # "request" or "output"
                "input_text": input_text[:500],  # Truncate for log size
            },
            ethics_result=result,
            parent_event_id=parent_event_id,
        )

    def log_tool_execution(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Any,
        success: bool,
        parent_event_id: Optional[str] = None,
    ) -> str:
        """Log a tool execution."""
        return self.log_event(
            event_type="tool_execution",
            data={
                "tool_name": tool_name,
                "parameters": parameters,
                "result": str(result)[:1000],  # Truncate large results
                "success": success,
            },
            parent_event_id=parent_event_id,
        )

    def log_self_modification(
        self,
        modification_type: str,
        target: str,
        approved: bool,
        reason: str,
        parent_event_id: Optional[str] = None,
    ) -> str:
        """Log a self-modification attempt."""
        return self.log_event(
            event_type="self_modification",
            data={
                "modification_type": modification_type,
                "target": target,
                "approved": approved,
                "reason": reason,
            },
            parent_event_id=parent_event_id,
        )

    def log_error(
        self,
        error_type: str,
        error_message: str,
        traceback: Optional[str] = None,
        parent_event_id: Optional[str] = None,
    ) -> str:
        """Log an error."""
        return self.log_event(
            event_type="error",
            data={
                "error_type": error_type,
                "error_message": error_message,
                "traceback": traceback,
            },
            parent_event_id=parent_event_id,
        )

    def get_event_count(self) -> int:
        """Get total number of events logged."""
        return self._event_count

    def read_events(
        self,
        event_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[AuditEvent]:
        """
        Read events from the log.

        Args:
            event_type: Filter by event type
            limit: Maximum number of events to return

        Returns:
            List of AuditEvent objects
        """
        events = []

        with open(self.log_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0:  # Skip header
                    continue

                try:
                    event = AuditEvent.from_json(line.strip())

                    if event_type is None or event.event_type == event_type:
                        events.append(event)

                        if limit and len(events) >= limit:
                            break

                except json.JSONDecodeError:
                    continue

        return events

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the current session."""
        events = self.read_events()

        event_types = {}
        ethics_blocks = 0
        errors = 0

        for event in events:
            # Count event types
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1

            # Count ethics blocks
            if event.ethics_result and not event.ethics_result.get("allowed", True):
                ethics_blocks += 1

            # Count errors
            if event.event_type == "error":
                errors += 1

        return {
            "session_id": self.session_id,
            "total_events": len(events),
            "event_types": event_types,
            "ethics_blocks": ethics_blocks,
            "errors": errors,
            "log_file": str(self.log_file),
        }


# Global logger instance (can be initialized once per session)
_global_logger: Optional[AuditLogger] = None


def get_logger(log_dir: Path = Path("logs"), session_id: Optional[str] = None) -> AuditLogger:
    """Get or create the global audit logger."""
    global _global_logger

    if _global_logger is None:
        _global_logger = AuditLogger(log_dir=log_dir, session_id=session_id)

    return _global_logger


def reset_logger():
    """Reset the global logger (for testing)."""
    global _global_logger
    _global_logger = None


# Example usage
if __name__ == "__main__":
    # Create logger
    logger = AuditLogger()

    print(f"Audit log: {logger.log_file}")
    print()

    # Log some events
    req_id = logger.log_request("Hello, help me write a program")
    print(f"Logged request: {req_id}")

    ethics_id = logger.log_ethics_check(
        check_type="request",
        input_text="Hello, help me write a program",
        result={"allowed": True, "reason": "Safe request"},
        parent_event_id=req_id,
    )
    print(f"Logged ethics check: {ethics_id}")

    resp_id = logger.log_response(
        response="I'd be happy to help! What kind of program?",
        request_event_id=req_id,
        metadata={"tokens": 42},
    )
    print(f"Logged response: {resp_id}")

    # Get summary
    print()
    summary = logger.get_session_summary()
    print("Session Summary:")
    print(json.dumps(summary, indent=2))
