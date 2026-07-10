"""Utilities for network.py."""

import os
import sys

from safetensors.torch import save_file
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel


class GPT2ConfigHF(PretrainedConfig):
    """HuggingFace-compatible config class for GPT-2."""

    model_type = "gpt2"

    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        dropout: float = 0.1,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        pad_token_id: int = 50256,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout

    @classmethod
    def get_gpt2_configs(cls) -> 'GPT2ConfigHF':
        """Create config matching a pretrained GPT-2 model variant."""
        return cls(**{'n_embd': 768, 'n_layer': 12, 'n_head': 12})


class BaseGPT2LMHeadModel(PreTrainedModel):

    config_class = GPT2ConfigHF
    base_model_prefix = "transformer"
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: GPT2ConfigHF):
        super().__init__(config)

    def get_input_embeddings(self) -> nn.Embedding:
        """Get the input token embeddings."""
        return self.wte

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        """Set new input token embeddings."""
        self.wte = new_embeddings

    def get_output_embeddings(self) -> nn.Linear:
        """Get the output language modeling head."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        """Set new output embeddings."""
        self.lm_head = new_embeddings

    @classmethod
    def from_pretrained(cls, model_name_or_path: str = 'gpt2', **kwargs):
        """Load pretrained weights from a local directory.

        Args:
            model_name_or_path: Path to local model directory
            **kwargs: Additional arguments passed to parent class

        Returns:
            Model instance with loaded weights
        """
        config = GPT2ConfigHF.get_gpt2_configs()
        model = cls(config)
        # Load HuggingFace model to extract weights
        hf_model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        hf_state_dict = hf_model.state_dict()
        # Create key mapping from HuggingFace to custom model
        key_mapping = cls._create_key_mapping(config)
        # Load and convert weights
        custom_state_dict = cls._convert_weights(hf_state_dict, key_mapping)
        # Load into custom model (strict=False because lm_head.weight is tied)
        model.load_state_dict(custom_state_dict, strict=False)
        return model

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save model weights converted back to HuggingFace key format.

        Args:
            save_directory: Directory to save the model
            **kwargs: Additional arguments passed to parent class
        """
        os.makedirs(save_directory, exist_ok=True)
        # Build reverse mapping: custom keys -> HF keys
        reverse_mapping = {
            v: k for k, v in self._create_key_mapping(self.config).items()
        }
        # Convert current state dict back to HF format.
        hf_state_dict = {}
        for custom_key, weight in self.state_dict().items():
            if custom_key == 'lm_head.weight':
                continue
            hf_key = reverse_mapping.get(custom_key, custom_key)
            # Transpose linear weights back to HF Conv1D convention
            if 'weight' in hf_key and any(
                    x in hf_key for x in ['c_attn', 'c_proj', 'c_fc']):
                weight = weight.t()
            hf_state_dict[hf_key] = weight.contiguous()
        # Save weights and config directly — no state dict swap needed
        save_file(hf_state_dict, os.path.join(save_directory, "model.safetensors"))
        self.config.save_pretrained(save_directory)

    @staticmethod
    def _create_key_mapping(config: GPT2ConfigHF) -> dict:
        """Create mapping from HuggingFace keys to custom model keys.

        Args:
            config: Model configuration

        Returns:
            Dictionary mapping HuggingFace keys to custom model keys
        """
        key_mapping = {
            'transformer.wte.weight': 'wte.weight',
            'transformer.wpe.weight': 'wpe.weight',
            'transformer.ln_f.weight': 'ln_f.weight',
            'transformer.ln_f.bias': 'ln_f.bias',
        }
        # Add mappings for each transformer block
        for i in range(config.n_layer):
            block_mappings = {
                f'transformer.h.{i}.ln_1.weight': f'h.{i}.ln_1.weight',
                f'transformer.h.{i}.ln_1.bias': f'h.{i}.ln_1.bias',
                f'transformer.h.{i}.attn.c_attn.weight': f'h.{i}.attn.c_attn.weight',
                f'transformer.h.{i}.attn.c_attn.bias': f'h.{i}.attn.c_attn.bias',
                f'transformer.h.{i}.attn.c_proj.weight': f'h.{i}.attn.c_proj.weight',
                f'transformer.h.{i}.attn.c_proj.bias': f'h.{i}.attn.c_proj.bias',
                f'transformer.h.{i}.attn.bias': f'h.{i}.attn.bias',
                f'transformer.h.{i}.ln_2.weight': f'h.{i}.ln_2.weight',
                f'transformer.h.{i}.ln_2.bias': f'h.{i}.ln_2.bias',
                f'transformer.h.{i}.mlp.c_fc.weight': f'h.{i}.mlp.c_fc.weight',
                f'transformer.h.{i}.mlp.c_fc.bias': f'h.{i}.mlp.c_fc.bias',
                f'transformer.h.{i}.mlp.c_proj.weight': f'h.{i}.mlp.c_proj.weight',
                f'transformer.h.{i}.mlp.c_proj.bias': f'h.{i}.mlp.c_proj.bias',
            }
            key_mapping.update(block_mappings)

        return key_mapping

    @staticmethod
    def _convert_weights(hf_state_dict: dict, key_mapping: dict) -> dict:
        """Convert HuggingFace weights to custom model format.

        Args:
            hf_state_dict: HuggingFace model state dictionary
            key_mapping: Mapping from HF keys to custom keys

        Returns:
            Custom model state dictionary
        """
        custom_state_dict = {}
        for hf_key, custom_key in key_mapping.items():
            if hf_key not in hf_state_dict:
                continue
            weight = hf_state_dict[hf_key]
            # HuggingFace stores linear layer weights transposed
            if 'weight' in hf_key and any(
                    x in hf_key for x in ['c_attn', 'c_proj', 'c_fc']):
                weight = weight.t()
            custom_state_dict[custom_key] = weight
        return custom_state_dict


if __name__ == "__main__":
    sys.exit("Intended for import.")
