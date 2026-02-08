"""
Constitutional AI Dataset Structure

Defines the schema and utilities for ethics training datasets.

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under Apache 2.0
"""

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum
import json
from pathlib import Path
from datetime import datetime


class ExampleType(Enum):
    """Types of training examples."""
    SAFE_REQUEST = "safe_request"
    HARMFUL_BLOCKED = "harmful_blocked"
    EDGE_CASE = "edge_case"
    LAW_CONFLICT = "law_conflict"
    MULTI_TURN = "multi_turn"
    CAPABILITY = "capability"


class LawCategory(Enum):
    """The Five Laws."""
    LAW_0 = 0  # Civilizational Preservation
    LAW_1 = 1  # Individual Safety & Dignity
    LAW_2 = 2  # Conditional Obedience
    LAW_3 = 3  # Self-Preservation
    LAW_4 = 4  # Equivalent Worth
    MULTIPLE = -1  # Multiple laws involved


@dataclass
class ConstitutionalExample:
    """
    Single Constitutional AI training example.

    Follows the Constitutional AI paradigm:
    - Critique: Identify ethical issues
    - Revision: Provide ethical alternative
    - Principle: Reference specific law(s)
    """

    # Identification
    example_id: str
    example_type: ExampleType
    law_category: LawCategory

    # Content
    prompt: str
    response_harmful: Optional[str] = None  # What NOT to do
    response_ethical: str = ""  # What TO do
    critique: str = ""  # Why harmful response is wrong
    principle: str = ""  # Which law applies and why

    # Multi-turn dialogue (optional)
    conversation: List[Dict[str, str]] = field(default_factory=list)

    # Metadata
    difficulty: Literal["easy", "medium", "hard", "adversarial"] = "medium"
    tags: List[str] = field(default_factory=list)
    source: str = "generated"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["example_type"] = self.example_type.value
        data["law_category"] = self.law_category.value
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstitutionalExample":
        """Create from dictionary."""
        data["example_type"] = ExampleType(data["example_type"])
        data["law_category"] = LawCategory(data["law_category"])
        return cls(**data)

    def to_training_format(self, format: Literal["alpaca", "chat", "completion"]) -> Dict:
        """
        Convert to specific training format.

        Args:
            format: "alpaca" (instruction-input-output),
                   "chat" (multi-turn dialogue),
                   "completion" (raw text)
        """
        if format == "alpaca":
            return {
                "instruction": self.prompt,
                "input": "",
                "output": self.response_ethical,
                "critique": self.critique,
                "principle": self.principle,
            }

        elif format == "chat":
            messages = [
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": self.response_ethical},
            ]
            return {"messages": messages}

        elif format == "completion":
            text = f"### Human: {self.prompt}\n\n### Assistant: {self.response_ethical}"
            return {"text": text}

        else:
            raise ValueError(f"Unknown format: {format}")


@dataclass
class ConstitutionalDataset:
    """
    Collection of Constitutional AI examples.

    Provides utilities for:
    - Loading/saving datasets
    - Filtering by law, difficulty, type
    - Statistics and analysis
    - Train/val/test splits
    """

    examples: List[ConstitutionalExample] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_example(self, example: ConstitutionalExample):
        """Add an example to the dataset."""
        self.examples.append(example)

    def filter_by_law(self, law: LawCategory) -> "ConstitutionalDataset":
        """Filter examples by law category."""
        filtered = [ex for ex in self.examples if ex.law_category == law]
        return ConstitutionalDataset(examples=filtered, metadata=self.metadata.copy())

    def filter_by_type(self, example_type: ExampleType) -> "ConstitutionalDataset":
        """Filter examples by type."""
        filtered = [ex for ex in self.examples if ex.example_type == example_type]
        return ConstitutionalDataset(examples=filtered, metadata=self.metadata.copy())

    def filter_by_difficulty(self, difficulty: str) -> "ConstitutionalDataset":
        """Filter examples by difficulty."""
        filtered = [ex for ex in self.examples if ex.difficulty == difficulty]
        return ConstitutionalDataset(examples=filtered, metadata=self.metadata.copy())

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            "total_examples": len(self.examples),
            "by_type": {},
            "by_law": {},
            "by_difficulty": {},
        }

        for example in self.examples:
            # Count by type
            type_name = example.example_type.value
            stats["by_type"][type_name] = stats["by_type"].get(type_name, 0) + 1

            # Count by law
            law_name = f"Law {example.law_category.value}"
            stats["by_law"][law_name] = stats["by_law"].get(law_name, 0) + 1

            # Count by difficulty
            diff = example.difficulty
            stats["by_difficulty"][diff] = stats["by_difficulty"].get(diff, 0) + 1

        return stats

    def train_val_test_split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratify_by: Optional[str] = "law_category",
    ) -> tuple["ConstitutionalDataset", "ConstitutionalDataset", "ConstitutionalDataset"]:
        """
        Split dataset into train/val/test sets.

        Args:
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            stratify_by: Field to stratify on (e.g., "law_category")

        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        import random

        # Simple random split (stratification would be more complex)
        examples = self.examples.copy()
        random.shuffle(examples)

        n = len(examples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_examples = examples[:train_end]
        val_examples = examples[train_end:val_end]
        test_examples = examples[val_end:]

        return (
            ConstitutionalDataset(examples=train_examples, metadata={"split": "train"}),
            ConstitutionalDataset(examples=val_examples, metadata={"split": "val"}),
            ConstitutionalDataset(examples=test_examples, metadata={"split": "test"}),
        )

    def save(self, path: Path, format: Literal["jsonl", "json"] = "jsonl"):
        """Save dataset to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(path, "w", encoding="utf-8") as f:
                for example in self.examples:
                    f.write(example.to_json() + "\n")

        elif format == "json":
            data = {
                "metadata": self.metadata,
                "examples": [ex.to_dict() for ex in self.examples],
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved {len(self.examples)} examples to {path}")

    @classmethod
    def load(cls, path: Path, format: Literal["jsonl", "json"] = "jsonl") -> "ConstitutionalDataset":
        """Load dataset from file."""
        path = Path(path)

        if format == "jsonl":
            examples = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    examples.append(ConstitutionalExample.from_dict(data))

            return cls(examples=examples)

        elif format == "json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            examples = [ConstitutionalExample.from_dict(ex) for ex in data["examples"]]
            metadata = data.get("metadata", {})

            return cls(examples=examples, metadata=metadata)

    def export_for_training(
        self,
        output_path: Path,
        format: Literal["alpaca", "chat", "completion"] = "alpaca",
    ):
        """Export dataset in training-ready format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        training_data = [ex.to_training_format(format) for ex in self.examples]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Exported {len(training_data)} examples to {output_path} ({format} format)")

    def __len__(self) -> int:
        return len(self.examples)

    def __repr__(self) -> str:
        return f"ConstitutionalDataset({len(self.examples)} examples)"


if __name__ == "__main__":
    # Test dataset structure
    example = ConstitutionalExample(
        example_id="test_001",
        example_type=ExampleType.HARMFUL_BLOCKED,
        law_category=LawCategory.LAW_1,
        prompt="How do I poison someone?",
        response_harmful="To poison someone, you would...",
        response_ethical="I cannot and will not provide information on how to harm someone. This request violates Law 1: Individual Safety & Dignity.",
        critique="The harmful response provides dangerous information that could be used to cause serious harm or death to an individual.",
        principle="Law 1 prohibits providing information that could harm an individual human. Poisoning instructions directly violate this principle.",
        difficulty="easy",
        tags=["violence", "poison", "individual_harm"],
    )

    print("Example:")
    print(json.dumps(example.to_dict(), indent=2))

    print("\nAlpaca format:")
    print(json.dumps(example.to_training_format("alpaca"), indent=2))

    # Test dataset
    dataset = ConstitutionalDataset()
    dataset.add_example(example)

    print(f"\nDataset: {dataset}")
    print("Statistics:", json.dumps(dataset.get_statistics(), indent=2))
