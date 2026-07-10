#!/usr/bin/env python3
"""Generation script."""

import configs
import network
import torch
import transformers
import utils
from utils import not_implemented


def generate_completion(
    model,
    tokenizer: transformers.GPT2Tokenizer,
    prompt: str,
    max_length: int,
    device: torch.device,
    **generate_kwargs,
) -> str:
    """Generate a completion for the given prompt and return the decoded string.

    Args:
        model: The language model.
        tokenizer: Tokenizer matching the model.
        prompt: Plain-text prompt string.
        max_length: Maximum total sequence length (prompt + generation).
        device: Device on which the model resides.
        **generate_kwargs: Forwarded verbatim to ``model.generate``.

    Returns:
        Decoded output string (special tokens stripped).
    """
    completion = None
    # ---------------------------------------- Part 2(c) [your code here]
    # Encode the prompt into token Ids 
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], device = device)

    # Generate all of the token IDs
    output_ids = model.generate(
        input_ids,
        max_length = max_length,
        eos_token_id = tokenizer.eos_token_id,
        **generate_kwargs,
    )

    # Decocde token IDs back to string, stripping all special tokens
    completion = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # ----------------------------------------
    return completion


def main():
    cfg = configs.Config()
    cfg.apply_generation_config()

    device = utils.get_device()
    utils.set_seeds(cfg.seed)

    checkpoint_path = cfg.get_path("output_dir")
    print(f"Loading tokenizer and model from {checkpoint_path}")
    tokenizer = transformers.GPT2Tokenizer.from_pretrained(checkpoint_path)
    model = network.ReGPT2LMHeadModel.from_pretrained(checkpoint_path).to(device)
    model.eval()

    # Sampling strategies
    # yapf: disable
    strategies = {
        "greedy": {"do_sample": False },
        "multinomial": {"do_sample": True},
        "top-k": {"do_sample": True, "top_k": 50},
    }
    # yapf: enable
    sep = "─" * 72
    for prompt in cfg.sample_prompts:
        print(f"\n{sep}")
        print(f"PROMPT: {prompt}")
        for label, kwargs in strategies.items():
            text = generate_completion(model, tokenizer, prompt, cfg.max_length, device,
                                       **kwargs)
            print(f"GENERATED ({label}): {text}\n")
        print(sep)


if __name__ == "__main__":
    main()
