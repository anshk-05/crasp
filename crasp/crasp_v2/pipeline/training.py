"""LoRA training helpers shared by Phase 0 and Phase R."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PromptDataset:
    """Simple tokenized prompt dataset for causal LM SFT."""

    encodings: list[dict[str, Any]]

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.encodings[idx]


class CausalLMCollator:
    """Pad variable-length prompt examples for causal LM fine-tuning."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.tokenizer.pad(features, return_tensors="pt")
        batch["labels"] = batch["input_ids"].clone()
        return batch


def build_prompt_dataset(texts: list[str], tokenizer: Any, max_length: int) -> PromptDataset:
    """Tokenize plain prompt text into a simple dataset."""
    encodings: list[dict[str, Any]] = []
    for text in texts:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        encodings.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
        )
    return PromptDataset(encodings=encodings)


def attach_lora(model: Any, training_cfg: dict[str, Any]) -> Any:
    """Attach a LoRA adapter to a causal LM."""
    from peft import LoraConfig, get_peft_model  # type: ignore[import]

    lora_cfg = LoraConfig(
        r=int(training_cfg["lora_rank"]),
        lora_alpha=int(training_cfg["lora_alpha"]),
        lora_dropout=float(training_cfg["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return get_peft_model(model, lora_cfg)


def run_lora_training(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    training_cfg: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Run a LoRA SFT pass and save the adapter directory."""
    from transformers import Trainer, TrainingArguments  # type: ignore[import]

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_prompt_dataset(texts, tokenizer, max_length=int(training_cfg.get("max_seq_len", 2048)))
    collator = CausalLMCollator(tokenizer)

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=float(training_cfg["learning_rate"]),
        num_train_epochs=float(training_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        warmup_ratio=float(training_cfg["warmup_ratio"]),
        logging_steps=int(training_cfg["logging_steps"]),
        save_strategy=str(training_cfg["save_strategy"]),
        bf16=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
