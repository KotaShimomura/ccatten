from typing import Optional

import torch
import torch.nn as nn

import ccatten as cat


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module used within transformer blocks.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ) -> None:
        """
        Initialize the MLP module.

        Args:
            in_features (int): Size of each input sample.
            hidden_features (int, optional): Size of the hidden layer. Defaults to in_features.
            out_features (int, optional): Size of each output sample. Defaults to in_features.
            drop (float, optional): Dropout probability. Defaults to 0.0.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1: nn.Linear = nn.Linear(in_features, hidden_features)
        self.act: nn.Module = nn.GELU()
        self.fc2: nn.Linear = nn.Linear(hidden_features, out_features)
        self.drop: nn.Dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying two linear layers, activation, and dropout.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block consisting of a circular attention layer and an MLP.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        """
        Initialize the Transformer block.

        Args:
            dim (int): Dimension of the input features.
            num_heads (int): Number of attention heads.
            mlp_ratio (float, optional): Ratio of mlp hidden dimension to embedding dimension. Defaults to 4.0.
            drop (float, optional): Dropout rate for MLP and output projection. Defaults to 0.0.
            attn_drop (float, optional): Dropout rate for attention weights. Defaults to 0.0.
        """
        super().__init__()
        self.norm1: nn.LayerNorm = nn.LayerNorm(dim)
        self.attn: CircularConvolutionalAttention = cat.CircularConvolutionalAttention(
            dim, num_heads=num_heads, qkv_bias=True, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2: nn.LayerNorm = nn.LayerNorm(dim)
        mlp_hidden_dim: int = int(dim * mlp_ratio)
        self.mlp: Mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, N, C) after applying attention and MLP with residual connections.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.

    This module splits an image into patches and projects each patch into an embedding vector using a convolutional layer.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Initialize the PatchEmbed module.

        Args:
            img_size (int, optional): Size of the input image (assumed square). Defaults to 224.
            patch_size (int, optional): Size of one patch. Defaults to 16.
            in_chans (int, optional): Number of input channels. Defaults to 3.
            embed_dim (int, optional): Dimension of the embedding. Defaults to 768.
        """
        super().__init__()
        self.img_size: int = img_size
        self.patch_size: int = patch_size
        self.grid_size: tuple[int, int] = (
            img_size // patch_size,
            img_size // patch_size,
        )
        self.num_patches: int = self.grid_size[0] * self.grid_size[1]
        self.proj: nn.Conv2d = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for patch embedding.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Patch embeddings of shape (B, num_patches, embed_dim).
        """
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model.

    This module implements a Vision Transformer architecture with patch embedding,
    class token, positional embeddings, Transformer blocks, and a final classification head.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ) -> None:
        """
        Initialize the Vision Transformer.

        Args:
            img_size (int, optional): Size of the input image. Defaults to 224.
            patch_size (int, optional): Size of each patch. Defaults to 16.
            in_chans (int, optional): Number of input channels. Defaults to 3.
            num_classes (int, optional): Number of output classes. Defaults to 1000.
            embed_dim (int, optional): Dimension of the embedding. Defaults to 768.
            depth (int, optional): Number of Transformer blocks. Defaults to 12.
            num_heads (int, optional): Number of attention heads in each block. Defaults to 12.
            mlp_ratio (float, optional): Ratio for hidden dimension in the MLP. Defaults to 4.0.
            drop_rate (float, optional): Dropout rate applied after embeddings and MLP. Defaults to 0.0.
            attn_drop_rate (float, optional): Dropout rate applied on attention weights. Defaults to 0.0.
        """
        super().__init__()
        self.num_classes: int = num_classes
        self.embed_dim: int = embed_dim

        # Patch embedding layer
        self.patch_embed: PatchEmbed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches: int = self.patch_embed.num_patches

        # Class token and positional embedding
        self.cls_token: torch.Tensor = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed: torch.Tensor = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.pos_drop: nn.Dropout = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks: nn.ModuleList = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                )
                for _ in range(depth)
            ]
        )
        self.norm: nn.LayerNorm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head: nn.Module = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize the weights of the model.
        """
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vision Transformer.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits for classification of shape (B, num_classes)
        """
        B: int = x.shape[0]
        # Patch embeddings: (B, num_patches, embed_dim)
        x = self.patch_embed(x)
        # Add class token: (B, num_patches+1, embed_dim)
        cls_tokens: torch.Tensor = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # Add positional embeddings and apply dropout
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # Apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # Extract the class token for classification
        cls_token_final: torch.Tensor = x[:, 0]
        x = self.head(cls_token_final)
        return x


def vit_tiny(**kwargs) -> VisionTransformer:
    """
    Constructs a Vision Transformer Tiny model.

    This configuration uses an embedding dimension of 192 and 3 attention heads.

    Returns:
        VisionTransformer: A ViT-Tiny model instance.
    """
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=192,  # Smaller embedding dimension for Tiny model
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        **kwargs,
    )


def vit_base(**kwargs) -> VisionTransformer:
    """
    Constructs a Vision Transformer Base model.

    This configuration uses an embedding dimension of 768 and 12 attention heads.

    Returns:
        VisionTransformer: A ViT-Base model instance.
    """
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,  # Larger embedding dimension for Base model
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        **kwargs,
    )


if __name__ == "__main__":
    x: torch.Tensor = torch.randn(2, 3, 224, 224)

    model_tiny: VisionTransformer = vit_tiny()
    logits_tiny: torch.Tensor = model_tiny(x)
    print("ViT Tiny logits shape:", logits_tiny.shape)

    model_base: VisionTransformer = vit_base()
    logits_base: torch.Tensor = model_base(x)
    print("ViT Base logits shape:", logits_base.shape)
