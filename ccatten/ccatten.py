import torch
import torch.nn as nn
import torch.nn.functional as F


def _gather_circular_weights(
    attn_weights: torch.Tensor,
    N: int,
    seq_len: int,
    num_heads: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Gather circularly shifted weights based on attention scores.

    Args:
        attn_weights (torch.Tensor): Attention weights with shape (B, h, N).
        N (int): Query sequence length.
        seq_len (int): Key/Value sequence length (M).
        num_heads (int): Number of attention heads.
        device (torch.device): Device for tensor creation.

    Returns:
        torch.Tensor: Circularly shifted weights with shape (B, h, N, M),
                      where roll_weights[b, h, i, j] = attn_weights[b, h, (j-i)%N].
    """
    B: int = attn_weights.shape[0]

    if attn_weights.dim() != 3 or attn_weights.shape[1] != num_heads or attn_weights.shape[2] != N:
        raise ValueError(
            f"Unexpected attn_weights shape: {attn_weights.shape}. Expected (B, h, N), where h is num_heads."
        )

    # Create indices with circular shifts: indices[i, j] = (j - i) % N, shape: (N, seq_len)
    col_indices = torch.arange(seq_len, device=device)  # Shape: (seq_len,)
    row_indices = torch.arange(N, device=device)          # Shape: (N,)
    indices = (col_indices.unsqueeze(0) - row_indices.unsqueeze(1)) % N  # (N, seq_len)

    # Expand indices to (B, num_heads, N, seq_len)
    indices = indices.view(1, 1, N, seq_len).expand(B, num_heads, N, seq_len)

    # Expand attn_weights to (B, num_heads, N, seq_len) and gather circularly shifted values
    expanded_attn = attn_weights.unsqueeze(3).expand(B, num_heads, N, seq_len)
    roll_weights = torch.gather(expanded_attn, dim=2, index=indices)

    roll_weights /= N  # Normalize weights by N

    return roll_weights


class CircularConvolutionalAttention(nn.Module):
    """
    Implementation of Circular-Convolutional Attention (CAT) using a gather-based version.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")

        self.num_heads: int = num_heads
        self.head_dim: int = dim // num_heads
        self.scale: float = self.head_dim ** -0.5

        # In CAT, a single projection is used for query/key (W_A)
        self.W_A = nn.Linear(dim, num_heads, bias=qkv_bias)  # Output: (B, N, h)
        self.W_V = nn.Linear(dim, dim, bias=qkv_bias)  # Value projection

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Compute attention scores and apply softmax
        z = self.W_A(x)               # (B, N, h)
        z = z.permute(0, 2, 1)          # (B, h, N)
        z_star = F.softmax(z, dim=-1)   # (B, h, N)
        z_star = self.attn_drop(z_star)

        # Gather circularly shifted weights (self-attention: seq_len == N)
        roll_weights = _gather_circular_weights(
            attn_weights=z_star,
            N=N,
            seq_len=N,
            num_heads=self.num_heads,
            device=x.device,
        )  # (B, h, N, N)

        # Project input to values
        V = self.W_V(x).reshape(B, N, self.num_heads, self.head_dim)
        V = V.permute(0, 2, 1, 3)  # (B, h, N, C/h)

        # Apply weights to values
        output = torch.matmul(roll_weights, V)  # (B, h, N, C/h)
        output = output.permute(0, 2, 1, 3).reshape(B, N, C)
        output = self.proj(output)
        output = self.proj_drop(output)
        return output


class AveragedKeyCircularConvolutionalAttention(nn.Module):
    """
    Implementation of the Averaged-Key variant for self- and cross-attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")

        self.num_heads: int = num_heads
        self.head_dim: int = dim // num_heads
        self.scale: float = self.head_dim ** -0.5

        # Separate query, key, and value projections
        self.W_Q = nn.Linear(dim, dim, bias=qkv_bias)
        self.W_K = nn.Linear(dim, dim, bias=qkv_bias)
        self.W_V = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).
            context (torch.Tensor): Optional context tensor for cross-attention.
                                    If None, self-attention is performed.
        """
        B, N, C = x.shape

        # Compute query projection and reshape
        Q = self.W_Q(x).reshape(B, N, self.num_heads, self.head_dim)
        Q = Q.permute(0, 2, 1, 3)  # (B, h, N, C/h)

        if context is not None:
            if context.dim() != 3:
                raise ValueError("Context must have 3 dimensions (B, M, C)")
            B_ctx, M, C_ctx = context.shape
            if B != B_ctx or C != C_ctx:
                raise ValueError("Batch size and feature dimension of context must match input x")
            K = self.W_K(context).reshape(B, M, self.num_heads, self.head_dim)
            K = K.permute(0, 2, 1, 3)  # (B, h, M, C/h)
            V = self.W_V(context).reshape(B, M, self.num_heads, self.head_dim)
            V = V.permute(0, 2, 1, 3)  # (B, h, M, C/h)
            seq_len: int = M
        else:
            K = self.W_K(x).reshape(B, N, self.num_heads, self.head_dim)
            K = K.permute(0, 2, 1, 3)  # (B, h, N, C/h)
            V = self.W_V(x).reshape(B, N, self.num_heads, self.head_dim)
            V = V.permute(0, 2, 1, 3)  # (B, h, N, C/h)
            seq_len = N

        # Average keys along the sequence dimension
        K_avg = K.mean(dim=2, keepdim=True)  # (B, h, 1, C/h)

        # Compute attention scores with averaged key
        z_circ = torch.matmul(Q, K_avg.transpose(-1, -2)) * self.scale  # (B, h, N, 1)
        z_circ = z_circ.squeeze(-1)  # (B, h, N)
        attn = F.softmax(z_circ, dim=-1)
        attn = self.attn_drop(attn)  # (B, h, N)

        # Gather circularly shifted weights
        roll_weights = _gather_circular_weights(
            attn_weights=attn,
            N=N,
            seq_len=seq_len,
            num_heads=self.num_heads,
            device=x.device,
        )  # (B, h, N, seq_len)

        output = torch.matmul(roll_weights, V)  # (B, h, N, C/h)
        output = output.permute(0, 2, 1, 3).reshape(B, N, C)
        output = self.proj(output)
        output = self.proj_drop(output)
        return output


class CausalCircularConvolutionalAttention(nn.Module):
    """
    Implementation of Circular-Convolutional Attention (CAT) for causal language modeling.
    Ensures that future tokens are masked.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")

        self.num_heads: int = num_heads
        self.head_dim: int = dim // num_heads
        self.scale: float = self.head_dim ** -0.5

        self.W_A = nn.Linear(dim, num_heads, bias=qkv_bias)
        self.W_V = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Compute attention scores and apply softmax along the sequence dimension
        z = self.W_A(x)                   # (B, N, h)
        z_star = F.softmax(z, dim=1)        # Softmax over the sequence dimension
        z_star = z_star.permute(0, 2, 1)      # (B, h, N)
        z_star = self.attn_drop(z_star)

        # Compute values projection and reshape to (B, h, N, C/h)
        V = self.W_V(x).reshape(B, N, self.num_heads, self.head_dim)
        V = V.permute(0, 2, 1, 3)

        # Gather circularly shifted weights (self-attention: seq_len == N)
        roll_weights = _gather_circular_weights(
            attn_weights=z_star,
            N=N,
            seq_len=N,
            num_heads=self.num_heads,
            device=x.device,
        )  # (B, h, N, N)

        # Create a causal mask to zero out future tokens
        causal_mask = torch.tril(torch.ones(N, N, device=x.device, dtype=torch.bool))
        causal_mask = causal_mask.view(1, 1, N, N).expand_as(roll_weights)
        masked_roll_weights = roll_weights.masked_fill(~causal_mask, 0.0)

        output = torch.matmul(masked_roll_weights, V)  # (B, h, N, C/h)
        output = output.permute(0, 2, 1, 3).reshape(B, N, C)
        output = self.proj(output)
        output = self.proj_drop(output)
        return output


# Demonstration of usage
if __name__ == "__main__":
    batch_size, seq_len, dim = 2, 32, 256
    x = torch.randn(batch_size, seq_len, dim)

    cat = CircularConvolutionalAttention(dim=dim)
    out_cat = cat(x)
    print(f"CAT output shape: {out_cat.shape}")

    avg_key = AveragedKeyCircularConvolutionalAttention(dim=dim)
    out_avg = avg_key(x)
    print(f"Averaged-Key output shape: {out_avg.shape}")

    context = torch.randn(batch_size, seq_len * 2, dim)
    out_cross = avg_key(x, context)
    print(f"Averaged-Key cross-attention output shape: {out_cross.shape}")

    causal_cat = CausalCircularConvolutionalAttention(dim=dim)
    out_causal = causal_cat(x)
    print(f"Causal CAT output shape: {out_causal.shape}")
