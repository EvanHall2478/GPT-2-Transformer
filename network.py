import math
import sys

from einops import rearrange
from network_utils import BaseGPT2LMHeadModel
from network_utils import GPT2ConfigHF
import torch
import torch.nn as nn
from utils import not_implemented


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking.

    Attributes:
        head_dim: The dimension of a single attention head.
        c_attn: Linear projection for generating query, key, and value vectors.
        c_proj: Linear projection for the output of the attention heads.
        attn_dropout: Dropout applied to the attention probabilities.
        resid_dropout: Dropout applied to the final projected output.
        bias: Buffer containing the causal mask to prevent attending to future tokens.
    """

    def __init__(self, n_emb: int, n_head: int, n_pos: int, dropout: float):
        """Initialize the Multi-head Attention module.

        Args:
            n_emb: The total embedding dimension of the input.
            n_head: The number of parallel attention heads.
            n_pos: The maximum sequence length used to pre-compute the causal mask.
            dropout: The dropout probability.
        """
        super().__init__()
        if n_emb % n_head != 0:
            raise ValueError(f"Embedding dimension {n_emb} must be divisible by "
                             f"number of heads {n_head}")

        # Attention-related dimensions.
        self.n_head = n_head
        self.n_emb = n_emb
        self.head_dim = n_emb // n_head

        # Define linear projections for
        # (1) queries, keys, and values
        self.c_attn = nn.Linear(n_emb, 3 * n_emb)
        # (2) outputs.
        self.c_proj = nn.Linear(n_emb, n_emb)

        # Initialize dropout for attention and residual connections.
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Build a lower-triangular mask to enforce causal attention.
        mask = torch.tril(torch.ones(n_pos, n_pos))

        bias = None
        # ---------------------------------------- Part 1(b) [your code here]
        # Build the causal bias matrix, where the attention is allowed set 1, else -inf
        bias = torch.zeros_like(mask)
        bias = bias.masked_fill(mask == 0, float('-inf'))
        # ----------------------------------------

        bias = rearrange(bias, 'i j -> 1 1 i j')
        # Register the mask as a non-trainable buffer for device consistency.
        # Important: Use bias as an instance variable in the code (self.bias)
        self.register_buffer("bias", bias, persistent=False)

    def forward(self, x: torch.Tensor):
        """Apply multi-head self-attention with causal masking.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_emb)

        Returns:
            Output tensor of shape (batch_size, seq_len, n_emb)
        """
        B, T, C = x.size()
        # ---------------------------------------- Part 1(b) [your code here]
        # Project x into Q, K, V
        qkv = self.c_attn(x)    # (B, T, 3C)
        q, k, v = qkv.split(self.n_emb, dim=2)  # (B, T, C) each

        # Reshape for multi-head attention
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.n_head)     # (B, H, T, dk)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.n_head)     # (B, H, T, dk)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.n_head)     # (B, H, T, dk)

        # Compute A = softmax(QK^T / sqrt(d_k) + B) for each head
        # A_scores = QK^T / sqrt(d_k)
        # QK^T: (B, H, T, dk) * (B, H, dk, T) -> (B, H, T, T)
        A_scores = q @ rearrange(k, 'b h t d -> b h d t') / math.sqrt(self.head_dim) + self.bias[:, :, :T, :T]
        A_weights = torch.softmax(A_scores, dim=-1)
        A_weights = self.attn_dropout(A_weights)

        # Attend the values
        y = A_weights @ v

        # Merge heads and projects
        y = rearrange(y, 'b h t d -> b t (h d)')
        y = self.resid_dropout(self.c_proj(y))
        return y
        # ----------------------------------------


class MLP(nn.Module):
    """Feed-forward network with GELU activation.

    Attributes:
        c_fc: Linear layer that expands the embedding dimension (n_emb -> 4 * n_emb).
        c_proj: Linear layer that projects back to the embedding dimension (4 * n_emb -> n_emb).
        dropout: Dropout layer applied to the output of the projection.
        gelu: Gaussian Error Linear Unit activation function.
    """

    def __init__(self, n_emb: int, dropout: float):
        """Initialize the MLP.

        Args:
            n_emb: The embedding dimension.
            dropout: The dropout probability.
        """
        super().__init__()
        self.c_fc = nn.Linear(n_emb, 4 * n_emb)
        self.c_proj = nn.Linear(4 * n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transformation."""
        # ---------------------------------------- Part 1(c)(i) [your code here]
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        # ----------------------------------------
        return x


class Block(nn.Module):
    """Transformer block with pre-norm architecture.

    Attributes:
        ln_1: Layer normalization applied before self-attention.
        attn: Multi-head self-attention module.
        ln_2: Layer normalization applied before the feed-forward network.
        mlp: Feed-forward neural network (MLP).
    """

    def __init__(self, n_emb: int, n_head: int, n_pos: int, dropout: float):
        """Initialize the Block.

        Args:
            n_emb: The total embedding dimension of the input.
            n_head: The number of parallel attention heads.
            n_pos: The maximum sequence length used to pre-compute the causal mask.
            dropout: The dropout probability.
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_emb)
        self.attn = MultiHeadAttention(n_emb, n_head, n_pos, dropout)
        self.ln_2 = nn.LayerNorm(n_emb)
        self.mlp = MLP(n_emb, dropout)

    def forward(self, x: torch.Tensor):
        """Apply attention and feed-forward with residual connections.

        Args:
             x: Input tensor.
        """
        # ---------------------------------------- Part 1(d)(i) [your code here]
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        # ----------------------------------------
        return x


class ReGPT2LMHeadModel(BaseGPT2LMHeadModel):
    """Custom GPT-2 implementation.

    Attributes:
        wte: Embedding layer mapping tokens to dense vectors.
        wpe: Embedding layer encoding positional information.
        drop: Dropout applied to the combined token and position embeddings.
        h: Stack of Transformer blocks.
        ln_f: Final layer normalization applied after the last block.
        lm_head: Linear projection from embeddings to vocabulary logits.
    """

    def __init__(self, cfg: GPT2ConfigHF):
        """Initialize the model.

        Args:
            cfg: Configuration object containing model hyperparameters.
        """
        super().__init__(cfg)
        self.cfg = cfg

        # Token embeddings (map tokens to dense vectors)
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # Position embeddings (encode information about the order of the words)
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        # Transformer blocks
        args = [cfg.n_embd, cfg.n_head, cfg.n_positions, cfg.dropout]
        self.h = nn.ModuleList([Block(*args) for _ in range(cfg.n_layer)])

        # Final layer norm
        self.ln_f = nn.LayerNorm(cfg.n_embd)

        # Language modeling head (project embeddings to vocabulary logits)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weights tied with token embeddings (share parameters for efficiency)
        self.lm_head.weight = self.wte.weight

        # Initialize weights
        self.post_init()
        self.tie_weights()
        self.total_params = sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ) -> dict:
        """Forward pass for causal language modeling.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask (unused in basic implementation)
            labels: Labels for computing loss (same shape as input_ids)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with 'loss', 'logits'
        """
        B, T = input_ids.size()
        logits = None
        # ---------------------------------------- Part 1(e)(ii) [your code here]
        token_emb = self.wte(input_ids) 
        position = torch.arange(T, device = input_ids.device)
        position_emb = self.wpe(position)
        x = self.drop(token_emb + position_emb)

        # pass through transformer blocks
        for block in self.h:
            x = block(x)

        # final layer norm and project
        x = self.ln_f(x)
        logits = self.lm_head(x)
        # ----------------------------------------

        loss = None
        if labels is not None:
            # Shift for causal language modeling (predict next token)
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            shift_logits_flat = rearrange(shift_logits, 'b t v -> (b t) v')
            shift_labels_flat = rearrange(shift_labels, 'b t -> (b t)')
            loss = nn.functional.cross_entropy(shift_logits_flat, shift_labels_flat)

        result = {'loss': loss, 'logits': logits}
        return result

    def _top_k_filter(self, logits, top_k):
        top_k = min(top_k, logits.size(-1))
        remove_mask = None

        # Apply top-k filtering
        # Hints: Use `torch.topk`. Create a boolean mask (`remove_mask`) for filtering.
        # ---------------------------------------- Part 2(b)(iii) [your code here]
        # Get the kth largest value for each item within the batch
        threshold = torch.topk(logits, top_k, dim = -1).values[..., -1, None]

        remove_mask = logits < threshold
        # ----------------------------------------

        # Apply the mask to set removed tokens to negative infinity
        logits = torch.where(remove_mask, torch.full_like(logits, float('-inf')),
                             logits)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        top_k: int = 0,
        do_sample: bool = True,
        eos_token_id: int = None,
    ) -> torch.Tensor:
        """Generates text sequences autoregressively.

        Args:
            input_ids: Initial token indices of shape (batch_size, seq_len).
            max_length: The maximum total length of the sequence to generate.
            top_k: Limits sampling to the top k most probable tokens (0 to disable).
            do_sample: If True, uses sampling; otherwise, uses greedy decoding.
            eos_token_id: The token ID that signifies the end of a sequence.

        Returns:
            Generated token sequences of shape (batch_size, generated_length).
        """
        self.eval()
        B, T = input_ids.size()
        if not do_sample and top_k > 0:
            raise ValueError("top_k requires do_sample=True.")

        for _ in range(max_length - T):
            outputs = self.forward(input_ids)
            logits = outputs['logits'][:, -1, :]

            if top_k > 0:
                logits = self._top_k_filter(logits, top_k)

            probs = torch.softmax(logits, dim=-1)
            next_token = None
            if do_sample:
                # Sample the next token from the filtered probability distribution.
                # Covers pure sampling, top-k
                # ---------------------------------------- Part 2(b)(ii) [your code here]
                next_token = torch.multinomial(probs, num_samples = 1)
                # ----------------------------------------
            else:
                # For greedy decoding
                # ---------------------------------------- Part 2(b)(i) [your code here]
                next_token = torch.argmax(probs, dim = -1, keepdim = True)
                # ----------------------------------------

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop if EOS token is generated for all sequences in batch
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids


if __name__ == "__main__":
    sys.exit("Intended for import.")
