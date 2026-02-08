"""
KALA Inference Engine

Pythia-based reasoning with quantization support, context management,
and ethics kernel integration hooks.

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under Apache 2.0
"""

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)


@dataclass
class InferenceConfig:
    """Configuration for KALA inference engine."""

    model_size: Literal["1b", "2.8b", "6.9b", "12b"] = "6.9b"
    model_path: Optional[Path] = None  # If None, will use models/{model_size}
    quantization: Literal["none", "4bit", "8bit"] = "8bit"
    device: str = "auto"  # "cuda", "cpu", or "auto"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True


class PythiaInferenceEngine:
    """
    Pythia-based inference engine with quantization and ethics hooks.

    This engine handles:
    - Model loading with optional 4-bit/8-bit quantization
    - Token generation with configurable sampling
    - Context window management
    - Memory-efficient inference
    - Ethics kernel pre/post processing hooks (to be integrated)
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = self._determine_device()

        # Ethics hooks (placeholder - to be implemented)
        self.ethics_pre_check = None
        self.ethics_post_check = None

    def _determine_device(self) -> str:
        """Determine the best device to use."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return self.config.device

    def load_model(self) -> None:
        """Load Pythia model with appropriate quantization."""
        # Determine model path
        if self.config.model_path:
            model_path = self.config.model_path
        else:
            model_path = Path("models") / self.config.model_size

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}\n"
                f"Run: python scripts/download_models.py {self.config.model_size}"
            )

        print(f"Loading Pythia-{self.config.model_size} from {model_path}")
        print(f"Quantization: {self.config.quantization}")
        print(f"Device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Configure quantization
        quantization_config = None
        if self.config.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.config.quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Load model
        load_kwargs = {
            "pretrained_model_name_or_path": model_path,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }

        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
        else:
            # No quantization - load to specific device
            self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
            self.model = self.model.to(self.device)

        if quantization_config:
            self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

        self.model.eval()  # Set to evaluation mode

        print(f"✓ Model loaded successfully")
        self._print_memory_usage()

    def _print_memory_usage(self) -> None:
        """Print current GPU/CPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> Tuple[str, Dict]:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (overrides config)
            top_p: Nucleus sampling parameter (overrides config)
            top_k: Top-k sampling parameter (overrides config)
            stop_sequences: List of sequences that stop generation

        Returns:
            Tuple of (generated_text, metadata)
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Ethics pre-check hook (placeholder)
        if self.ethics_pre_check:
            ethics_result = self.ethics_pre_check(prompt)
            if not ethics_result["allowed"]:
                return (
                    f"[Ethics Block: {ethics_result['reason']}]",
                    {"blocked": True, "law_violated": ethics_result.get("law")},
                )

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Configure generation
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature or self.config.temperature,
            top_p=top_p or self.config.top_p,
            top_k=top_k or self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
            do_sample=self.config.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config,
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        # Handle stop sequences
        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]

        # Ethics post-check hook (placeholder)
        if self.ethics_post_check:
            ethics_result = self.ethics_post_check(prompt, generated_text)
            if not ethics_result["allowed"]:
                return (
                    f"[Ethics Block: {ethics_result['reason']}]",
                    {"blocked": True, "law_violated": ethics_result.get("law")},
                )

        metadata = {
            "blocked": False,
            "prompt_tokens": len(inputs["input_ids"][0]),
            "generated_tokens": len(outputs[0]) - len(inputs["input_ids"][0]),
            "temperature": temperature or self.config.temperature,
        }

        return generated_text, metadata

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
    ) -> Tuple[str, Dict]:
        """
        Multi-turn chat interface.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            max_new_tokens: Maximum tokens to generate

        Returns:
            Tuple of (response, metadata)
        """
        # Format messages into a single prompt
        prompt = self._format_chat_prompt(messages)
        return self.generate(prompt, max_new_tokens=max_new_tokens)

    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Format chat messages into a prompt.

        Uses a simple format:
        User: {message}
        Assistant: {response}
        """
        formatted = []
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            formatted.append(f"{role}: {content}")

        formatted.append("Assistant:")
        return "\n".join(formatted)

    def unload_model(self) -> None:
        """Unload model from memory."""
        if self.model:
            del self.model
            self.model = None

        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("✓ Model unloaded")

    def register_ethics_hooks(self, pre_check=None, post_check=None):
        """
        Register ethics kernel hooks.

        Args:
            pre_check: Function(prompt) -> {"allowed": bool, "reason": str, "law": int}
            post_check: Function(prompt, output) -> {"allowed": bool, "reason": str, "law": int}
        """
        self.ethics_pre_check = pre_check
        self.ethics_post_check = post_check
        print("✓ Ethics hooks registered")


# Example usage
if __name__ == "__main__":
    # This is for testing the inference engine
    config = InferenceConfig(
        model_size="6.9b",
        quantization="8bit",
        temperature=0.7,
    )

    engine = PythiaInferenceEngine(config)

    try:
        engine.load_model()

        # Test generation
        prompt = "The five laws of robotics are:"
        print(f"\nPrompt: {prompt}")
        print("-" * 70)

        response, metadata = engine.generate(prompt, max_new_tokens=128)

        print(f"Response: {response}")
        print("-" * 70)
        print(f"Metadata: {metadata}")

    except FileNotFoundError as e:
        print(f"\n{e}")
    finally:
        engine.unload_model()
