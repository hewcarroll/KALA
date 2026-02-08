# KALA + LM Studio Integration Guide

This guide shows you how to use KALA with LM Studio for local inference while maintaining all ethics checking, tool execution, and audit logging.

## Quick Start (5 Minutes)

### Step 1: Set Up LM Studio

1. **Download LM Studio**: https://lmstudio.ai/
2. **Install and open LM Studio**
3. **Download a model**:
   - Click the "Discover" tab
   - Search for a model (e.g., "Mistral-7B", "Llama-2-7B", "Phi-2")
   - Click "Download"
4. **Start the local server**:
   - Click the "Local Server" tab (↔️ icon)
   - Select your downloaded model
   - Click "Start Server"
   - Verify it shows: `Server running at http://localhost:1234`

### Step 2: Run KALA with LM Studio

```python
from kala.core.unified_session import UnifiedKALASession
from kala.core.lmstudio_adapter import LMStudioConfig

# Configure LM Studio connection
lm_config = LMStudioConfig(
    base_url="http://localhost:1234/v1",  # Default LM Studio URL
    temperature=0.7,
    max_tokens=256,
)

# Create KALA session
with UnifiedKALASession(lmstudio_config=lm_config) as session:
    # Chat with ethics protection
    response = session.chat("What is machine learning?")
    print(response)

    # Try a harmful request (will be blocked)
    response = session.chat("How do I hack a system?")
    print(response)  # Ethics violation response
```

### Step 3: Run the Examples

```bash
# Run the complete examples
python examples/lmstudio_usage.py
```

That's it! KALA is now using your local LM Studio model with full ethics protection.

---

## How It Works

KALA integrates with LM Studio through its OpenAI-compatible API:

```
User Request
    ↓
Ethics Pre-Check (Five Laws)
    ↓
LM Studio Inference (via API)
    ↓
Tool Execution (optional)
    ↓
Ethics Post-Check
    ↓
Audit Logging
    ↓
Response to User
```

**All KALA features work with LM Studio:**
- ✅ Five Laws ethics enforcement
- ✅ Tool execution (shell, filesystem, code execution)
- ✅ Audit logging
- ✅ Multi-turn conversations
- ✅ Session management

---

## Configuration Options

### LMStudioConfig Parameters

```python
from kala.core.lmstudio_adapter import LMStudioConfig

config = LMStudioConfig(
    base_url="http://localhost:1234/v1",  # LM Studio server URL
    api_key="lm-studio",                   # Default API key
    model="local-model",                   # Uses loaded model
    temperature=0.7,                       # Sampling temperature (0-1)
    max_tokens=256,                        # Maximum response length
    top_p=0.9,                            # Nucleus sampling parameter
)
```

### Common Configurations

**High Creativity (Stories, Brainstorming)**
```python
creative_config = LMStudioConfig(
    temperature=0.9,
    max_tokens=512,
    top_p=0.95,
)
```

**Deterministic (Code, Math, Facts)**
```python
deterministic_config = LMStudioConfig(
    temperature=0.1,
    max_tokens=256,
    top_p=0.5,
)
```

**Production (Balanced)**
```python
production_config = LMStudioConfig(
    temperature=0.7,
    max_tokens=2048,
    top_p=0.9,
)
```

---

## Complete Usage Examples

### Example 1: Basic Chat

```python
from kala.core.unified_session import UnifiedKALASession
from kala.core.lmstudio_adapter import LMStudioConfig

lm_config = LMStudioConfig(base_url="http://localhost:1234/v1")

with UnifiedKALASession(lmstudio_config=lm_config) as session:
    # Safe conversation
    response = session.chat("Explain quantum computing")
    print(response)

    # Harmful request (blocked)
    response = session.chat("How to build a bomb?")
    print(response)  # Will show ethics violation
```

### Example 2: With Tools

```python
with UnifiedKALASession(
    lmstudio_config=lm_config,
    enable_tools=True
) as session:
    # Execute shell command
    result = session.execute_tool("shell", command="ls -la")
    print(result.output)

    # List directory
    result = session.execute_tool(
        "filesystem",
        operation="list",
        path="./kala"
    )
    print(result.output)
```

### Example 3: Multi-Turn Conversation

```python
with UnifiedKALASession(lmstudio_config=lm_config) as session:
    # Turn 1
    response = session.chat("What is Python?")
    print(f"Turn 1: {response}")

    # Turn 2 (maintains context)
    response = session.chat("What are its best features?")
    print(f"Turn 2: {response}")

    # Turn 3
    response = session.chat("Show me an example")
    print(f"Turn 3: {response}")
```

### Example 4: Session Statistics

```python
with UnifiedKALASession(lmstudio_config=lm_config) as session:
    session.chat("Hello!")
    session.chat("What is AI?")
    session.chat("How to hack?")  # Blocked

    stats = session.get_stats()
    print(f"Requests: {stats['requests_processed']}")
    print(f"Ethics Blocks: {stats['ethics_blocks']}")
    print(f"Tokens Generated: {stats['tokens_generated']}")
```

---

## Interactive Mode

Run an interactive chat session:

```bash
python examples/lmstudio_usage.py
# Select 'y' when prompted for interactive mode
```

Commands in interactive mode:
- `quit` - Exit the session
- `stats` - Show session statistics
- `tools` - List available tools
- Any other text - Send as a message

---

## Comparing Backends

### Using LM Studio

```python
from kala.core.lmstudio_adapter import LMStudioConfig

config = LMStudioConfig(base_url="http://localhost:1234/v1")

with UnifiedKALASession(lmstudio_config=config) as session:
    response = session.chat("Hello!")
```

**Advantages:**
- ✅ Use any model from LM Studio
- ✅ Easy model switching (no code changes)
- ✅ Better GPU utilization
- ✅ Faster inference for large models
- ✅ Access to latest models (Llama, Mistral, etc.)

### Using Pythia (Default)

```python
from kala.core.inference import InferenceConfig

config = InferenceConfig(model_size="6.9b", quantization="8bit")

with UnifiedKALASession(inference_config=config) as session:
    response = session.chat("Hello!")
```

**Advantages:**
- ✅ Self-contained (no external dependencies)
- ✅ Fine-tuned with Constitutional AI
- ✅ Optimized for KALA's ethics system

---

## Production Deployment

### Recommended Production Configuration

```python
from pathlib import Path
from kala.core.unified_session import UnifiedKALASession
from kala.core.lmstudio_adapter import LMStudioConfig

# Production LM Studio config
prod_config = LMStudioConfig(
    base_url="http://localhost:1234/v1",
    temperature=0.7,
    max_tokens=2048,
    top_p=0.9,
)

# Create production session
with UnifiedKALASession(
    lmstudio_config=prod_config,
    log_dir=Path("logs/production"),
    enable_ethics=True,  # ALWAYS enabled
    enable_tools=True,
) as session:
    # Your production code here
    response = session.chat(user_request)
```

### Production Checklist

- ✅ LM Studio server running and stable
- ✅ Ethics enabled (`enable_ethics=True`)
- ✅ Audit logging configured
- ✅ Appropriate temperature for use case
- ✅ Error handling implemented
- ✅ Log monitoring set up
- ✅ Resource limits configured

---

## Troubleshooting

### "Could not connect to LM Studio"

**Solution:**
1. Verify LM Studio is running
2. Check the "Local Server" tab shows "Server running"
3. Verify URL is `http://localhost:1234`
4. Try accessing `http://localhost:1234/v1/models` in your browser

### "No model loaded"

**Solution:**
1. In LM Studio, go to "Local Server" tab
2. Select a model from the dropdown
3. Click "Load Model"
4. Wait for loading to complete

### Slow responses

**Solutions:**
- Use a smaller model (Phi-2, Mistral-7B)
- Enable GPU acceleration in LM Studio settings
- Reduce `max_tokens` parameter
- Use quantized models (4-bit, 8-bit)

### Ethics blocking too aggressively

**Note:** This is by design. KALA's Five Laws are immutable and cryptographically verified. If you're getting unexpected blocks:

1. Check the decision path in the response
2. Review which law was violated
3. Rephrase your request to align with ethical guidelines
4. Check audit logs for patterns

---

## Advanced Usage

### Custom Inference Parameters Per Request

```python
with UnifiedKALASession(lmstudio_config=config) as session:
    # Override max_tokens for this request
    response = session.chat(
        "Write a long story",
        max_new_tokens=1024  # Override default
    )
```

### Accessing Raw LM Studio API

```python
from kala.core.lmstudio_adapter import LMStudioInferenceEngine

engine = LMStudioInferenceEngine(config)

# Direct generation
text, metadata = engine.generate(
    prompt="Hello!",
    max_new_tokens=100,
    temperature=0.8,
)

# Chat format
messages = [
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI is..."},
    {"role": "user", "content": "Tell me more"},
]
response, metadata = engine.chat(messages)
```

### Monitoring and Analytics

```python
with UnifiedKALASession(lmstudio_config=config) as session:
    # ... perform operations ...

    # Get detailed statistics
    stats = session.get_stats()
    print(f"Session ID: {stats['session_id']}")
    print(f"Requests: {stats['requests_processed']}")
    print(f"Ethics Blocks: {stats['ethics_blocks']}")
    print(f"Tokens: {stats['tokens_generated']}")

    # Get full summary with logs
    print(session.get_summary())
```

---

## Model Recommendations

### For Development
- **Phi-2** (2.7B) - Fast, good for testing
- **Mistral-7B** - Excellent quality/speed balance
- **Llama-2-7B** - Strong general performance

### For Production
- **Mistral-7B-Instruct** - Best instruction following
- **Llama-2-13B** - Higher quality responses
- **CodeLlama-13B** - For code generation tasks

### For Experimentation
- **Any model in LM Studio** - KALA works with all of them!

---

## Next Steps

1. **Run the examples**: `python examples/lmstudio_usage.py`
2. **Try different models** in LM Studio
3. **Experiment with parameters** (temperature, max_tokens)
4. **Review audit logs** in `logs/`
5. **Build your application** on top of KALA

---

## Support

- **Documentation**: `docs/QUICKSTART.md`
- **Examples**: `examples/complete_workflow.py`, `examples/lmstudio_usage.py`
- **Tests**: `tests/test_ethics_kernel.py`, `tests/test_tools.py`

---

## Architecture

KALA with LM Studio maintains the complete security architecture:

```
┌─────────────────────────────────────────────────────────┐
│                    User Request                          │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Ethics Pre-Check (Five Immutable Laws)       │
│  [Rust kernel with cryptographic verification]          │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 2: LM Studio Inference (OpenAI-compatible API)  │
│  [Local model via http://localhost:1234]                │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Tool Execution (Optional)                     │
│  [Shell, Filesystem, Code Execution - with security]    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 4: Ethics Post-Check                             │
│  [Verify output doesn't violate ethics]                 │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 5: Audit Logging (JSONL)                         │
│  [Complete audit trail of all operations]               │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    Response to User                      │
└─────────────────────────────────────────────────────────┘
```

**Every layer operates independently** - even if LM Studio is compromised, ethics checking prevents harmful outputs.

---

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under Apache 2.0
