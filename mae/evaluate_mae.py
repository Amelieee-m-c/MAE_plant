"""
models_mae.py — MambaVision-MAE
================================
Channel / spatial trace for MambaVision-Tiny (dim=80):
  patch_embed : (N, 3, 224, 224) → (N, 80, 56, 56)   stride-4 CNN stem
  level 0     : (N, 80, 56, 56)  → (N, 160, 28, 28)  CNN + downsample
  level 1     : (N, 160, 28, 28) → (N, 320, 14, 14)  CNN + downsample
                                    196 tokens, 320-d  ← token grid

We stop at level 1. Level 2 outputs 640-d at 7×7 (too few tokens).

References
----------
[1] He et al., MAE, CVPR 2022.  https://arxiv.org/abs/2111.06377
[2] Hatamizadeh & Kautz, MambaVision, NeurIPS 2024.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from functools import partial
from typing import Tuple

from timm.models.vision_transformer import Block as ViTBlock
from util.pos_embed import get_2d_sincos_pos_embed
from models import mamba_vision
from models.registry import register_pip_model

# NOTE: mamba_vision is imported lazily inside __init__ to avoid circular
# import caused by models/__init__.py doing `from .mamba_vision import *`

# base_dim for each variant (the `dim` arg to MambaVision constructor)
# after level 0+1 (two downsample steps): output channels = base_dim * 4
_BASE_DIM = {
    "mamba_vision_T":     80,   # → 320
    "mamba_vision_T2":    80,   # → 320
    "mamba_vision_S":     96,   # → 384
    "mamba_vision_B":    128,   # → 512
    "mamba_vision_B_21k": 128,  # → 512
    "mamba_vision_L":    196,   # → 784
    "mamba_vision_L2":   196,   # → 784
}


# ---------------------------------------------------------------------------
# Positional embedding helper
# ---------------------------------------------------------------------------

def build_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Return (1, grid_size**2, embed_dim) fixed sin-cos positional embedding."""
    pos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
    return torch.from_numpy(pos).float().unsqueeze(0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MambaVisionMAE(nn.Module):
    """
    Masked Autoencoder with MambaVision CNN tokenizer.

    Tokenizer : patch_embed + levels[0, 1]  (pure CNN, no Mamba blocks)
    Encoder   : ViT blocks on visible tokens only  [1, Table 1c]
    Decoder   : ViT blocks on full token set (visible + mask tokens)
    Loss      : per-patch MSE on masked tokens only [1, §3]
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        encoder_embed_dim: int = 320,
        encoder_depth: int = 6,
        encoder_num_heads: int = 10,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss: bool = True,
        mamba_type: str = "mamba_vision_T",
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.in_chans = in_chans
        self.norm_pix_loss = norm_pix_loss
        self.grid_size = img_size // patch_size   # 14
        self.num_patches = self.grid_size ** 2    # 196

        # ── 1. MambaVision tokenizer ──────────────────────────────────────
        # Lazy import here to break the circular import chain:
        #   models_mae → models/__init__ → models/mamba_vision → models_mae
        import importlib
        _mv_module = importlib.import_module("models.mamba_vision")
        _build_fn = getattr(_mv_module, mamba_type)
        backbone = _build_fn(pretrained=False, num_classes=0)

        self.patch_embed = backbone.patch_embed
        self.cnn_stages = nn.Sequential(
            backbone.levels[0],   # 80ch 56×56 → 160ch 28×28
            backbone.levels[1],   # 160ch 28×28 → 320ch 14×14
        )

        if mamba_type not in _BASE_DIM:
            raise ValueError(
                f"Unknown mamba_type '{mamba_type}'. "
                f"Choose from: {list(_BASE_DIM.keys())}"
            )
        cnn_out_dim: int = _BASE_DIM[mamba_type] * 4

        # ── 2. Encoder input projection ───────────────────────────────────
        self.encoder_input_proj: nn.Module = (
            nn.Linear(cnn_out_dim, encoder_embed_dim, bias=True)
            if encoder_embed_dim != cnn_out_dim
            else nn.Identity()
        )

        # ── 3. Encoder positional embedding (sin-cos, frozen) ─────────────
        self.pos_embed = nn.Parameter(
            build_2d_sincos_pos_embed(encoder_embed_dim, self.grid_size),
            requires_grad=False,
        )

        # ── 4. Encoder Transformer blocks ─────────────────────────────────
        self.encoder_blocks = nn.ModuleList([
            ViTBlock(
                encoder_embed_dim, encoder_num_heads, mlp_ratio,
                qkv_bias=True, norm_layer=norm_layer,
            )
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # ── 5. Decoder ────────────────────────────────────────────────────
        self.decoder_embed = nn.Linear(
            encoder_embed_dim, decoder_embed_dim, bias=True
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            build_2d_sincos_pos_embed(decoder_embed_dim, self.grid_size),
            requires_grad=False,
        )
        self.decoder_blocks = nn.ModuleList([
            ViTBlock(
                decoder_embed_dim, decoder_num_heads, mlp_ratio,
                qkv_bias=True, norm_layer=norm_layer,
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size ** 2 * in_chans, bias=True
        )

        self._initialize_weights()

    # ── Weight initialisation ──────────────────────────────────────────────

    def _initialize_weights(self) -> None:
        self.pos_embed.data.copy_(
            build_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size)
        )
        self.decoder_pos_embed.data.copy_(
            build_2d_sincos_pos_embed(
                self.decoder_pos_embed.shape[-1], self.grid_size
            )
        )
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_module_weights)

    @staticmethod
    def _init_module_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    # ── Patchify / Unpatchify ──────────────────────────────────────────────

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) → (N, L, patch_size²·C)"""
        p = self.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], self.in_chans, h, p, w, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(imgs.shape[0], h * w, p ** 2 * self.in_chans)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(N, L, patch_size²·C) → (N, C, H, W)"""
        p = self.patch_size
        h = w = self.grid_size
        x = x.reshape(x.shape[0], h, w, p, p, self.in_chans)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], self.in_chans, h * p, w * p)

    # ── Random masking ─────────────────────────────────────────────────────

    def random_masking(
        self, x: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_visible = torch.gather(
            x, dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D),
        )

        mask = torch.ones(N, L, device=x.device)
        mask.scatter_(1, ids_keep, 0.0)
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_visible, mask, ids_restore

    # ── Encoder ────────────────────────────────────────────────────────────

    def forward_encoder(
        self, x: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.patch_embed(x)              # (N, 80, 56, 56)
        x = self.cnn_stages(x)              # (N, 320, 14, 14)
        x = x.flatten(2).transpose(1, 2)   # (N, 196, 320)
        x = self.encoder_input_proj(x)     # (N, 196, encoder_embed_dim)
        x = x + self.pos_embed
        x_visible, mask, ids_restore = self.random_masking(x, mask_ratio)
        for blk in self.encoder_blocks:
            x_visible = blk(x_visible)
        x_visible = self.encoder_norm(x_visible)
        return x_visible, mask, ids_restore

    # ── Decoder ────────────────────────────────────────────────────────────

    def forward_decoder(
        self, x: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        x = self.decoder_embed(x)
        num_mask = self.num_patches - x.shape[1]
        mask_tokens = self.mask_token.expand(x.shape[0], num_mask, -1)
        x_full = torch.cat([x, mask_tokens], dim=1)
        x_full = torch.gather(
            x_full, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[2]),
        )
        x_full = x_full + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x_full = blk(x_full)
        x_full = self.decoder_norm(x_full)
        pred = self.decoder_pred(x_full)
        return pred

    # ── Loss ───────────────────────────────────────────────────────────────

    def forward_loss(
        self,
        imgs: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var  = target.var( dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self, imgs: torch.Tensor, mask_ratio: float = 0.75
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def mae_mamba_vision_tiny(**kwargs) -> MambaVisionMAE:
    """Tiny: 320-d encoder, 512-d decoder."""
    return MambaVisionMAE(
        mamba_type="mamba_vision_T",
        encoder_embed_dim=320,
        encoder_depth=6,
        encoder_num_heads=10,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def mae_mamba_vision_small(**kwargs) -> MambaVisionMAE:
    """Small: 384-d encoder, 512-d decoder."""
    return MambaVisionMAE(
        mamba_type="mamba_vision_S",
        encoder_embed_dim=384,
        encoder_depth=8,
        encoder_num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def mae_mamba_vision_base(**kwargs) -> MambaVisionMAE:
    """Base: 512-d encoder, 512-d decoder."""
    return MambaVisionMAE(
        mamba_type="mamba_vision_B",
        encoder_embed_dim=512,
        encoder_depth=12,
        encoder_num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
