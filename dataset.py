"""Module for loading code dataset."""

import collections
import glob
import os
import pathlib
import random
import sys
from typing import Dict, List

import configs
import torch
from torch.utils import data
import transformers


class CodeDataset(data.Dataset):
    """PyTorch Dataset for Python source code using a stream-based approach."""

    def __init__(
        self,
        tokenizer: transformers.GPT2Tokenizer,
        split: str,
        cfg: configs.Config,
        seed: int = 49,
    ):
        """Initialise the dataset.

        Args:
            tokenizer: A GPT-2 tokenizer instance.
            split: 'train' or 'validation'.
            cfg: Configuration object.
            seed: Seed for reproducible category shuffling.
        """
        self.tokenizer = tokenizer
        self.max_length = cfg.max_length
        self.stride = cfg.stride
        self.split = split

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        files_to_process = self._get_files(cfg.data_dir, cfg.val_frac, seed)
        tokenized_files = self._tokenize_files(files_to_process)
        self.segments = self._prepare_segments(tokenized_files)

        if cfg.quick_test and self.segments:
            self.segments = self.segments[:cfg.test_samples]

    def _get_files(self, data_dir: str, val_frac: float, seed: int) -> List[str]:
        """Discover Python files and return file contents for the requested split."""
        source_files = sorted(glob.glob(f"{data_dir}/**/*.py", recursive=True),
                              key=lambda p: os.path.relpath(p, data_dir))
        if not len(source_files):
            raise ValueError(
                f"No .py files found under '{data_dir}'. "
                "Run `python a3/script_download_resources.py` to download the dataset.")

        by_category = collections.defaultdict(list)
        for f in source_files:
            category = pathlib.Path(f).relative_to(data_dir).parts[0]
            by_category[category].append(f)

        categories = sorted(by_category.keys())
        rng = random.Random(seed)
        rng.shuffle(categories)

        n_val_cats = max(1, int(len(categories) * val_frac))
        val_categories = set(categories[:n_val_cats])
        train_categories = set(categories[n_val_cats:])

        selected = train_categories if self.split == "train" else val_categories
        files_to_process = [f for cat in selected for f in by_category[cat]]
        contents = []
        for path in files_to_process:
            with open(path, "r", encoding="utf-8") as f:
                contents.append(f.read())
        return contents

    def _tokenize_files(self, file_contents: List[str]) -> List[List[int]]:
        """Tokenize each file and append EOS as a file-boundary marker."""
        tokenized = []
        for content in file_contents:
            token_ids = self.tokenizer.encode(content, add_special_tokens=False)
            token_ids.append(self.tokenizer.eos_token_id)
            tokenized.append(token_ids)
        return tokenized

    def _prepare_segments(self, tokenized_files: List[List[int]]) -> List[List[int]]:
        """Concatenate all files into one stream, then slide a window over it."""
        # Flatten all file tokens into a single stream
        stream: List[int] = []
        for tokens in tokenized_files:
            stream.extend(tokens)
        segments = []
        for start in range(0, len(stream) - self.max_length + 1, self.stride):
            segments.append(stream[start:start + self.max_length])
        return segments

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single fixed-length segment as a dict of tensors."""
        ids = torch.tensor(self.segments[idx], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": ids.clone(),
        }


if __name__ == "__main__":
    sys.exit("Intended for import")
