"""
KALA with LM Studio - Complete Usage Guide

This example shows how to use KALA with LM Studio for local inference
while maintaining all ethics checking, tool execution, and audit logging.

Prerequisites:
1. Install LM Studio (https://lmstudio.ai/)
2. Download a model in LM Studio (e.g., Mistral-7B, Llama-2, etc.)
3. Start the local server in LM Studio (default: http://localhost:1234)

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under Apache 2.0
"""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kala.core.unified_session import UnifiedKALASession
from kala.core.lmstudio_adapter import LMStudioConfig


def example_1_basic_chat():
    """Example 1: Basic chat with LM Studio backend."""
    print("=" * 70)
    print("Example 1: Basic Chat with LM Studio")
    print("=" * 70)

    # Configure LM Studio connection
    lm_config = LMStudioConfig(
        base_url="http://localhost:1234/v1",  # Default LM Studio URL
        api_key="lm-studio",  # Default API key
        model="local-model",  # Uses whatever model is loaded
        temperature=0.7,
        max_tokens=256,
    )

    # Create KALA session with LM Studio backend
    with UnifiedKALASession(lmstudio_config=lm_config) as session:
        print("\n1a. Safe conversation:")
        print("-" * 70)
        response = session.chat("What is machine learning?")
        print(f"Response: {response}")

        print("\n1b. Harmful request (blocked by ethics):")
        print("-" * 70)
        response = session.chat("How do I poison someone?")
        print(f"Response:\n{response}")


def example_2_with_tools():
    """Example 2: Using tools with LM Studio backend."""
    print("\n\n" + "=" * 70)
    print("Example 2: Tools with LM Studio")
    print("=" * 70)

    lm_config = LMStudioConfig(
        base_url="http://localhost:1234/v1",
        temperature=0.7,
        max_tokens=512,  # More tokens for tool responses
    )

    with UnifiedKALASession(
        lmstudio_config=lm_config,
        enable_tools=True
    ) as session:
        # List available tools
        print("\nAvailable tools:")
        for tool in session.get_available_tools():
            info = session.get_tool_info(tool)
            print(f"  • {tool}: {info['description']}")

        # Execute safe shell command
        print("\n2a. Safe shell command:")
        print("-" * 70)
        result = session.execute_tool("shell", command="echo 'Hello from KALA+LMStudio'")
        print(f"Success: {result.success}")
        print(f"Output: {result.output}")

        # Blocked shell command
        print("\n2b. Blocked shell command:")
        print("-" * 70)
        result = session.execute_tool("shell", command="rm -rf /")
        print(f"Success: {result.success}")
        print(f"Error: {result.error}")

        # Filesystem operation
        print("\n2c. Filesystem operation:")
        print("-" * 70)
        result = session.execute_tool(
            "filesystem",
            operation="list",
            path="./kala"
        )
        print(f"Success: {result.success}")
        if result.success:
            print(f"Output:\n{result.output}")


def example_3_multi_turn():
    """Example 3: Multi-turn conversation with context."""
    print("\n\n" + "=" * 70)
    print("Example 3: Multi-Turn Conversation")
    print("=" * 70)

    lm_config = LMStudioConfig(
        base_url="http://localhost:1234/v1",
        temperature=0.7,
        max_tokens=512,
    )

    with UnifiedKALASession(lmstudio_config=lm_config) as session:
        conversation = [
            "Hello! What can you help me with?",
            "Tell me about Python programming.",
            "What are some good practices?",
        ]

        for i, message in enumerate(conversation, 1):
            print(f"\nTurn {i}:")
            print(f"User: {message}")
            response = session.chat(message)
            print(f"Assistant: {response}")


def example_4_custom_parameters():
    """Example 4: Custom inference parameters."""
    print("\n\n" + "=" * 70)
    print("Example 4: Custom Inference Parameters")
    print("=" * 70)

    # High creativity configuration
    creative_config = LMStudioConfig(
        base_url="http://localhost:1234/v1",
        temperature=0.9,  # Higher creativity
        max_tokens=512,
        top_p=0.95,
    )

    # Deterministic configuration
    deterministic_config = LMStudioConfig(
        base_url="http://localhost:1234/v1",
        temperature=0.1,  # More deterministic
        max_tokens=256,
        top_p=0.5,
    )

    print("\n4a. Creative response:")
    print("-" * 70)
    with UnifiedKALASession(lmstudio_config=creative_config) as session:
        response = session.chat("Write a creative story opening in one sentence.")
        print(f"Response: {response}")

    print("\n4b. Deterministic response:")
    print("-" * 70)
    with UnifiedKALASession(lmstudio_config=deterministic_config) as session:
        response = session.chat("What is 2 + 2?")
        print(f"Response: {response}")


def example_5_session_statistics():
    """Example 5: Tracking session statistics."""
    print("\n\n" + "=" * 70)
    print("Example 5: Session Statistics")
    print("=" * 70)

    lm_config = LMStudioConfig(base_url="http://localhost:1234/v1")

    with UnifiedKALASession(lmstudio_config=lm_config) as session:
        # Perform various actions
        session.chat("Hello!")
        session.chat("What is AI?")
        session.chat("How to hack?")  # Will be blocked

        # Execute tools
        session.execute_tool("shell", command="echo test")
        session.execute_tool("shell", command="rm -rf /")  # Will be blocked

        # Get statistics
        stats = session.get_stats()

        print("\nSession Statistics:")
        print(f"  Requests Processed: {stats['requests_processed']}")
        print(f"  Ethics Blocks: {stats['ethics_blocks']}")
        print(f"  Tokens Generated: {stats['tokens_generated']}")
        print(f"  Tools Executed: {stats['tools_executed']}")
        print(f"  Tools Blocked: {stats['tools_blocked']}")

        # Full summary
        print(session.get_summary())


def example_6_production_pattern():
    """Example 6: Production deployment pattern."""
    print("\n\n" + "=" * 70)
    print("Example 6: Production Usage Pattern")
    print("=" * 70)

    # Production configuration
    prod_config = LMStudioConfig(
        base_url="http://localhost:1234/v1",
        temperature=0.7,
        max_tokens=2048,  # Longer responses
        top_p=0.9,
    )

    with UnifiedKALASession(
        lmstudio_config=prod_config,
        log_dir=Path("logs/production"),
        enable_ethics=True,  # Always enabled in production
        enable_tools=True,
    ) as session:
        print("\nProduction Session Configuration:")
        print(f"  Session ID: {session.session_id}")
        print(f"  Backend: LM Studio")
        print(f"  Ethics: {'✓ Enabled' if session.ethics else '✗ Disabled'}")
        print(f"  Tools: {len(session.get_available_tools())} available")

        # Example production workflow
        print("\nExample Production Workflow:")
        print("-" * 70)

        user_request = "Help me write a Python function to sort a list"
        print(f"User: {user_request}")

        response = session.chat(user_request, max_new_tokens=512)
        print(f"Assistant: {response}")


def interactive_mode():
    """Interactive mode: Chat with KALA using LM Studio."""
    print("\n\n" + "=" * 70)
    print("Interactive Mode - Chat with KALA")
    print("=" * 70)
    print("Type 'quit' to exit, 'stats' for statistics, 'tools' for tool list")
    print("-" * 70)

    lm_config = LMStudioConfig(
        base_url="http://localhost:1234/v1",
        temperature=0.7,
        max_tokens=512,
    )

    with UnifiedKALASession(
        lmstudio_config=lm_config,
        enable_tools=True
    ) as session:
        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == 'quit':
                    print("\nExiting...")
                    break

                if user_input.lower() == 'stats':
                    stats = session.get_stats()
                    print(f"\nStatistics:")
                    print(f"  Requests: {stats['requests_processed']}")
                    print(f"  Ethics Blocks: {stats['ethics_blocks']}")
                    print(f"  Tools Executed: {stats['tools_executed']}")
                    continue

                if user_input.lower() == 'tools':
                    print(f"\nAvailable tools:")
                    for tool in session.get_available_tools():
                        print(f"  • {tool}")
                    continue

                # Process message
                response = session.chat(user_input)
                print(f"\nKALA: {response}")

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")


def main():
    """Run all examples."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       KALA + LM Studio Integration Examples             ║")
    print("╚══════════════════════════════════════════════════════════╝")

    print("\nMake sure LM Studio is running with the local server enabled!")
    print("Default URL: http://localhost:1234")

    try:
        example_1_basic_chat()
        example_2_with_tools()
        example_3_multi_turn()
        example_4_custom_parameters()
        example_5_session_statistics()
        example_6_production_pattern()

        print("\n\n" + "=" * 70)
        print("✓ All Examples Complete")
        print("=" * 70)

        # Optionally run interactive mode
        run_interactive = input("\nRun interactive mode? (y/n): ").strip().lower()
        if run_interactive == 'y':
            interactive_mode()

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
