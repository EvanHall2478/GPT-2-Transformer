"""Configuration definitions for this assignment.

Modify this file to set the allowed hyperparameters and options you believe are
appropriate for your experiments.
"""
import argparse
import dataclasses
import json
import os
import sys
import time
from typing import List

import utils


@dataclasses.dataclass
class Config:
    """Configuration for model training with command-line argument parsing."""

    # yapf: disable
    # ---------------------------------------- [you may modify this]
    # Training configs
    num_train_epochs: int = 10              # Total number of passes through the training dataset
    per_device_train_batch_size: int = 4    # Number of training samples processed per GPU/CPU per step
    per_device_eval_batch_size: int = 4     # Number of evaluation samples processed per GPU/CPU per step
    learning_rate: float = 5e-5             # Step size for the optimizer (AdamW default for fine-tuning)
    weight_decay: float = 0.01              # L2 regularization strength to reduce overfitting
    warmup_ratio: float = 0.2               # Fraction of total steps used for linear LR warmup before decay

    # Data configs
    max_length: int = 1024                  # Maximum token sequence length; longer sequences are truncated

    # Eval configs
    eval_strategy: str = "epoch"            # When to run evaluation: "epoch" evaluates after each epoch ("steps" is the alternative)
    eval_samples: int = 3                   # Number of text samples to generate and log during each evaluation
    # ----------------------------------------
    # yapf: enable

    # Experiment configs
    run_id: str = f"{int(time.time() * 1000)}"
    quick_test: bool = False
    hf_root_dir: str = "/mnt/hf_cache" if utils.is_cluster_node() else "a3/hf_cache"
    ds_root_dir: str = "/mnt/code_cache" if utils.is_cluster_node() else "a3/code_cache"

    # Data configs
    data_dir: str = f"{ds_root_dir}/Python"
    val_frac: float = 0.1
    stride: int = 256

    # Model configs
    model_repo_id = "gpt2"
    model_name: str = f"{hf_root_dir}/models/{model_repo_id}"

    # Paths
    output_dir: str = "a3/runs/{run_id}/model"
    samples_path: str = "a3/runs/{run_id}/generations/{step}.json"
    configs_path: str = "a3/runs/{run_id}/configs.json"

    # Other settings
    fp16: bool = False if str(utils.get_device()) == "mps" else True
    seed: int = 42

    # Generation config.
    sample_prompts: List[str] = dataclasses.field(default_factory=lambda: [
        "def is_palindrome(head: ListNode | None) -> bool:\n    \"\"\"\n    Check if a linked list is a palindrome.\n    \"\"\"\n    if not head:\n        return True\n    # split the list to two parts\n    fast: ListNode | None = head.next_node\n    slow: ListNode | None = head\n    while fast and fast.next_node:\n        fast = fast.next_node.next_node\n        slow = slow.next_node if slow else None\n    if slow:",
    ])

    def get_path(self, key, **kwargs):
        if "run_id" not in kwargs:
            kwargs["run_id"] = self.run_id
        path = getattr(self, key).format(**kwargs)
        dir_name = os.path.dirname(path) if "." in path else path
        os.makedirs(dir_name, exist_ok=True)
        return path

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--quick_test",
            action="store_true",
            help="run a fast, lightweight configuration for testing",
        )
        parser.add_argument(
            "--run_id",
            type=str,
            help="Only use for generation (using generate.py)",
        )
        return parser.parse_args()

    def apply_quick_test_config(self, quick_test=False):
        args = self.parse_args()
        if args.quick_test or quick_test:
            self.quick_test = True
            self.run_id = "quick_test"
            self.num_train_epochs = 1
            self.eval_samples = 1
            self.per_device_eval_batch_size = 1
            self.test_samples = 16
            self.max_length = 126

    def apply_generation_config(self):
        args = self.parse_args()
        if not args.run_id:
            raise ValueError(f"Need pass run_id of trained model in '--args.run_id'")
        self.run_id = args.run_id

    def save(self):
        with open(self.get_path("configs_path"), "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)


if __name__ == "__main__":
    sys.exit("Intended for import.")
