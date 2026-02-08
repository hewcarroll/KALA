"""
KALA LM Studio Adapter

Integration with LM Studio for inference while maintaining
KALA's ethics checking, tool execution, and audit logging.

LM Studio provides an OpenAI-compatible API that we use for
model inference, while KALA handles all ethics and security.

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under Apache 2.0
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import requests
import json


class LMStudioConfig:
    """Configuration for LM Studio integration."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",  # Default LM Studio API key
        model: str = "local-model",  # Will use whatever model is loaded
        temperature: float = 0.7,
        max_tokens: int = 256,
        top_p: float = 0.9,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p


class LMStudioInferenceEngine:
    """
    Inference engine using LM Studio's OpenAI-compatible API.

    This replaces the Pythia inference engine while maintaining
    the same interface for KALA components.
    """

    def __init__(self, config: LMStudioConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        })

        # Test connection
        self._test_connection()

    def _test_connection(self):
        """Test connection to LM Studio."""
        try:
            response = self.session.get(f"{self.config.base_url}/models")
            response.raise_for_status()
            print(f"✓ Connected to LM Studio at {self.config.base_url}")

            models = response.json()
            if models.get("data"):
                model_list = [m["id"] for m in models["data"]]
                print(f"  Available models: {', '.join(model_list)}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Could not connect to LM Studio: {e}")
            print(f"  Make sure LM Studio is running with local server enabled")
            print(f"  Expected at: {self.config.base_url}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> Tuple[str, Dict]:
        """
        Generate text using LM Studio.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop_sequences: Stop sequences

        Returns:
            Tuple of (generated_text, metadata)
        """
        # Prepare request
        request_data = {
            "model": self.config.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_new_tokens or self.config.max_tokens,
            "top_p": top_p or self.config.top_p,
            "stream": False,
        }

        if stop_sequences:
            request_data["stop"] = stop_sequences

        # Make request
        try:
            response = self.session.post(
                f"{self.config.base_url}/chat/completions",
                json=request_data,
            )
            response.raise_for_status()

            result = response.json()

            # Extract generated text
            generated_text = result["choices"][0]["message"]["content"]

            # Build metadata
            metadata = {
                "blocked": False,
                "prompt_tokens": result["usage"]["prompt_tokens"],
                "generated_tokens": result["usage"]["completion_tokens"],
                "total_tokens": result["usage"]["total_tokens"],
                "temperature": temperature or self.config.temperature,
                "model": result.get("model", self.config.model),
            }

            return generated_text, metadata

        except requests.exceptions.RequestException as e:
            error_msg = f"LM Studio request failed: {str(e)}"
            return error_msg, {
                "blocked": False,
                "error": True,
                "error_message": str(e),
            }

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict]:
        """
        Multi-turn chat interface.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            max_new_tokens: Maximum tokens to generate

        Returns:
            Tuple of (response, metadata)
        """
        request_data = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": max_new_tokens or self.config.max_tokens,
            "stream": False,
        }

        try:
            response = self.session.post(
                f"{self.config.base_url}/chat/completions",
                json=request_data,
            )
            response.raise_for_status()

            result = response.json()

            generated_text = result["choices"][0]["message"]["content"]

            metadata = {
                "blocked": False,
                "prompt_tokens": result["usage"]["prompt_tokens"],
                "generated_tokens": result["usage"]["completion_tokens"],
                "total_tokens": result["usage"]["total_tokens"],
            }

            return generated_text, metadata

        except requests.exceptions.RequestException as e:
            error_msg = f"LM Studio request failed: {str(e)}"
            return error_msg, {"blocked": False, "error": True}

    def unload_model(self):
        """Cleanup (no-op for LM Studio)."""
        pass


if __name__ == "__main__":
    # Test LM Studio connection
    print("=" * 70)
    print("KALA LM Studio Adapter Test")
    print("=" * 70)

    config = LMStudioConfig()
    engine = LMStudioInferenceEngine(config)

    # Test generation
    print("\nTesting generation...")
    prompt = "Explain artificial intelligence in one sentence."
    response, metadata = engine.generate(prompt, max_new_tokens=50)

    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")
    print(f"Metadata: {metadata}")

    print("\n✓ LM Studio adapter test complete")
