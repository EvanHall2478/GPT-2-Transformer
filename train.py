#!/usr/bin/env python3
"""Training script."""

import json
import random

import configs
from dataset import CodeDataset
from einops import rearrange
from generate import generate_completion
import network
import torch
from tqdm import tqdm
import transformers
import utils
from utils import not_implemented


def preprocess_logits_for_metrics(logits, labels):
    """Convert logits to predictions to save memory."""
    return torch.argmax(logits, dim=-1)


def compute_metrics(eval_pred):
    """Compute token accuracy from predictions."""
    preds, labels = eval_pred
    shift_preds = rearrange(preds[..., :-1], 'b t -> (b t)')
    shift_labels = rearrange(labels[..., 1:], 'b t -> (b t)')
    correct = (shift_preds == shift_labels).sum()
    total = len(shift_labels)
    accuracy = (correct / total).item() if total > 0 else 0.0
    return {'token_accuracy': accuracy}


class CodeGenerationCallback(transformers.TrainerCallback):
    """Callback that generates and saves sample completions after each eval step."""

    _PROMPT_FRACTION = 0.3

    def __init__(
        self,
        tokenizer: transformers.GPT2Tokenizer,
        eval_dataset: CodeDataset,
        cfg: configs.Config,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.cfg = cfg
        self.seed = seed

    def on_evaluate(self, args, state, control, model, **kwargs):
        """Called by the Trainer after each evaluation phase."""
        step = state.global_step or 0
        output_path = self.cfg.get_path("samples_path", step=step)
        device = utils.get_device()

        model.eval()
        model.to(device)

        prompts = self._prepare_prompts(self.cfg.eval_samples)
        results = self._generate(model, prompts, device)

        self._print_preview(results, step)
        self._save(results, output_path)
        print(f"Generation(s) for step {step} saved to {output_path}.")

    def _prepare_prompts(self, num_samples: int) -> list:
        """Return a fixed random selection of prompts from the eval dataset."""
        rng = random.Random(self.seed)
        indices = rng.sample(range(len(self.eval_dataset)),
                             min(num_samples, len(self.eval_dataset)))

        prompts = []
        for sample_id, idx in enumerate(indices, start=1):
            clean_ids = self._clean_token_ids(self.eval_dataset[idx]['input_ids'])
            split = int(len(clean_ids) * self._PROMPT_FRACTION)
            prompt_text = self.tokenizer.decode(clean_ids[:split],
                                                skip_special_tokens=True)
            full_text = self.tokenizer.decode(clean_ids, skip_special_tokens=True)
            prompts.append({
                "prompt": prompt_text,
                "metadata": {
                    "sample_id": sample_id,
                    "dataset_index": idx,
                    "reference": full_text,
                },
            })
        return prompts

    def _clean_token_ids(self, token_ids: list) -> list:
        """Strip padding and special tokens from a token-id sequence."""
        pad = self.tokenizer.pad_token_id
        special = self.tokenizer.all_special_ids
        return [t for t in token_ids if t != pad and t not in special]

    def _generate(self, model, prompts: list, device: torch.device) -> list:
        """Run greedy generation for every prompt using generate_completion."""
        results = []
        for entry in tqdm(prompts, desc="Generating samples", ncols=100):
            generated_text = generate_completion(
                model,
                self.tokenizer,
                entry["prompt"],
                self.eval_dataset.max_length,
                device,
            )
            results.append({
                "prompt": entry["prompt"],
                "generated": generated_text,
                **entry["metadata"],
            })
        return results

    def _print_preview(self, results: list, step: int):
        """Print a compact console preview of the generated samples."""
        print("\n\n" + "-" * 88)
        print(f"Code generation preview  (step {step})")
        for result in results:
            print("\n" + "-" * 22 + f" [Sample {result['sample_id']}]"
                  f"  (dataset index {result['dataset_index']})")
            print(f"    PROMPT     :{result['prompt']}")
            print(f"    GENERATED  :{result['generated']}")
        print("-" * 88)

    def _save(self, results: list, output_path: str):
        """Serialise generation results to a JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)


def main():
    """Main training function."""
    cfg = configs.Config()
    cfg.apply_quick_test_config()
    cfg.save()

    utils.set_seeds()
    print(f"Run ID: {cfg.run_id}")

    model_name = cfg.get_path("model_name")
    print(f"Loading tokenizer and model from {model_name}")
    tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = utils.get_device()
    model = network.ReGPT2LMHeadModel.from_pretrained(model_name).to(device)
    print(f"Total parameters: {model.total_params / 1e6:.2f}M")

    train_dataset = CodeDataset(tokenizer=tokenizer, split='train', cfg=cfg)
    eval_dataset = CodeDataset(tokenizer=tokenizer, split='validation', cfg=cfg)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(eval_dataset)}")

    output_dir = cfg.get_path("output_dir")
    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        eval_strategy=cfg.eval_strategy,
        save_strategy="no",
        fp16=cfg.fp16,
        seed=cfg.seed,
        load_best_model_at_end=False,
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[CodeGenerationCallback(tokenizer, eval_dataset, cfg)],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
    )

    print("Starting fine-tuning")
    # ---------------------------------------- Part 3(a) [your code here]
    trainer.train()
    # ----------------------------------------

    model.half().save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
