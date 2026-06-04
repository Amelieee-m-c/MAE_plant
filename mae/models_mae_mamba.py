# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block
from util.pos_embed import get_2d_sincos_pos_embed

import mamba_vision


# ===========================================================================
# MaskedAutoencoderViT（原版，僅修正 qk_scale）
# ===========================================================================

class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        from timm.models.vision_transformer import PatchEmbed
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(imgs.shape[0], h * w, p**2 * 3)

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        '''
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        '''
        x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        #x = self.norm(x)
        return self.norm(x), mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x[:, 1:, :]

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        return (loss * mask).sum() / mask.sum()

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


# ===========================================================================
# MaskedAutoencoderMamba
#
# 學長說的設計邏輯：
#   - MambaVision 的 CNN 下採樣本身就是在「切 patch」
#     patch_embed + level0 + level1 → [N, 320, 14, 14]（56倍下採樣）
#     patch_embed + level0 + level1 + level2 + level3 → [N, 640, 7, 7]
#     feature map 已經很小，不需要另外做 patch merge
#
#   - 完整 MambaVision 當 Encoder（看完整圖片）
#     Encoder 輸出: [N, 640, 7, 7] → flatten → [N, 49, 640]
#
#   - Masking 在 Encoder 輸出後才做（Decoder 用來重建原圖）
#     這樣 Encoder 永遠看到完整圖，window_partition 不會出問題
#
#   - 實驗目標：
#     實驗 A：用 MAE 預訓練過的 MambaVision Encoder → 分類 → ACC
#     實驗 B：原始 MambaVision（沒有 MAE）直接分類  → ACC
#     比較 A vs B，驗證 MAE 自監督預訓練對病灶辨識的效益
#
# Tensor 流程（mamba_vision_T）：
#   patch_embed              [N, 3,   224, 224] → [N, 80,  56, 56]
#   level 0 (CNN)            [N, 80,  56,  56]  → [N, 160, 28, 28]
#   level 1 (CNN)            [N, 160, 28,  28]  → [N, 320, 14, 14]
#   level 2 (Transformer)    [N, 320, 14,  14]  → [N, 640,  7,  7]
#   level 3 (Transformer)    [N, 640,  7,   7]  → [N, 640,  7,  7]
#   norm (BatchNorm2d)       [N, 640,  7,   7]
#   flatten                  [N, 640,  7,   7]  → [N, 49, 640]
#   ── Masking 在這裡 ──────────────────────────────────────────────
#   random_masking           [N, 49, 640]       → [N, 12, 640]  (75% masked)
#   decoder_embed            [N, 12, 640]       → [N, 12, 512]
#   unshuffle + mask_token   [N, 12, 512]       → [N, 49, 512]
#   decoder_pos_embed        [N, 49, 512]
#   Transformer blocks x8   [N, 49, 512]
#   decoder_pred             [N, 49, 512]       → [N, 49, 32^2 x 3=3072]
#   ── 重建原圖 7x7 個 patch，每個 patch 32x32 像素 ────────────────
# ===========================================================================

class MaskedAutoencoderMamba(nn.Module):
    """
    完整 MambaVision 作為 Encoder 的 MAE。

    Encoder 看完整圖片（不做 masking），輸出後再 masking，
    讓 Decoder 學習重建被遮蔽的 patch。

    預訓練完後，直接取 self.encoder 部分做下游分類任務。
    """

    def __init__(self,
                 mamba_type='mamba_vision_T',
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=16,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 norm_pix_loss=False,
                 **kwargs):
        super().__init__()

        # ------------------------------------------------------------------
        # Encoder：完整 MambaVision（包含全部 4 個 level）
        # ------------------------------------------------------------------
        self.encoder = getattr(mamba_vision, mamba_type)(num_classes=0, **kwargs)

        # 用 dummy forward 自動推算 encoder 輸出的 shape
        # mamba_vision_T: [N, 640, 7, 7]
        enc_dim, enc_grid = self._infer_encoder_output_shape()
        self.encoder_embed_dim = enc_dim    # 640
        self.encoder_grid_size = enc_grid   # 7
        self.num_patches = enc_grid ** 2    # 49

        # patch_size：原圖每個 patch 對應幾個像素
        # 224 / 7 = 32
        self.patch_size = 224 // enc_grid   # 32

        # Decoder 位置編碼 [1, 49, 512]（無 cls token）
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_embed_dim),
            requires_grad=False)

        # ------------------------------------------------------------------
        # Decoder：標準 Transformer（無 cls token）
        # ------------------------------------------------------------------
        self.decoder_embed = nn.Linear(self.encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio,
                  qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)

        # 重建每個 patch 的像素：32^2 × 3 = 3072
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, self.patch_size ** 2 * 3, bias=True)

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    @torch.no_grad()
    def _infer_encoder_output_shape(self):
        """用 dummy forward 推算完整 Encoder 輸出的 (channel, spatial_size)。"""
        dummy = torch.zeros(1, 3, 224, 224)
        x = self.encoder.patch_embed(dummy)
        for level in self.encoder.levels:
            x = level(x)
        # norm 是 BatchNorm2d，也要過
        x = self.encoder.norm(x)
        _, C, H, W = x.shape
        assert H == W, f"Encoder 輸出不是正方形：{H}x{W}"
        return C, H

    def initialize_weights(self):
        # Decoder 位置編碼：7x7 grid，無 cls token
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            self.encoder_grid_size,   # 7
            cls_token=False)
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: [N, 3, 224, 224]
        return: [N, 49, 3072]   （7x7 個 patch，每個 32x32x3）
        """
        p = self.patch_size  # 32
        h = w = imgs.shape[2] // p  # 7
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(imgs.shape[0], h * w, p**2 * 3)

    def random_masking(self, x, mask_ratio):
        """
        Encoder 輸出後才做 masking。
        x: [N, 49, 640]
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        """
        完整跑過 MambaVision，輸出後再做 masking。

        為什麼 Encoder 要看完整圖片？
        → MambaVisionLayer 內部做 window_partition，強制要求完整的 [N,C,H,W]
        → Masking 後的不規則 token 無法做 window partition
        → 所以 Encoder 永遠看完整圖，Masking 在輸出後做
        """
        # 完整 MambaVision forward
        x = self.encoder.patch_embed(x)       # [N, 80, 56, 56]
        for level in self.encoder.levels:
            x = level(x)                       # 最終 [N, 640, 7, 7]
        x = self.encoder.norm(x)              # BatchNorm2d

        # [N, C, H, W] → [N, H*W, C]
        N, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)      # [N, 49, 640]

        # Encoder 輸出後才 masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        # x: [N, 12, 640]（75% masked → 只剩 25% = 12 個）

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """
        無 cls token 版本的 Decoder。
        """
        x = self.decoder_embed(x)  # [N, 12, 512]

        # 補回被遮蔽的位置（用 mask_token 填充）
        num_masked = ids_restore.shape[1] - x.shape[1]  # 49 - 12 = 37
        mask_tokens = self.mask_token.repeat(x.shape[0], num_masked, 1)
        x_ = torch.cat([x, mask_tokens], dim=1)          # [N, 49, 512]

        # 還原原始 patch 順序
        x = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )  # [N, 49, 512]

        # 加位置編碼
        x = x + self.decoder_pos_embed

        # Transformer decode
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # 預測像素值
        x = self.decoder_pred(x)  # [N, 49, 3072]

        return x  # 無 cls token，直接返回

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, 224, 224]
        pred: [N, 49, 3072]
        mask: [N, 49]，0=保留，1=遮蔽
        只計算被遮蔽的 patch 的重建 loss
        """
        target = self.patchify(imgs)  # [N, 49, 3072]
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)          # [N, 49]
        return (loss * mask).sum() / mask.sum()

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


# ===========================================================================
# 用於下游分類的 Encoder（預訓練完後拿出來用）
# ===========================================================================

class MambaVisionForClassification(nn.Module):
    """
    MAE 預訓練完後，取出 Encoder 部分做分類。
    直接用 MambaVision 原本的 forward_features + 新的分類頭。

    實驗 A：載入 MAE 預訓練的 encoder 權重
    實驗 B：不載入，直接隨機初始化（或載入 ImageNet 預訓練）
    比較兩者在病灶分類的 ACC
    """
    def __init__(self, mamba_type='mamba_vision_T', num_classes=2, **kwargs):
        super().__init__()
        # 載入完整 MambaVision（head 先設為 Identity）
        self.backbone = getattr(mamba_vision, mamba_type)(num_classes=0, **kwargs)

        # 取得 backbone 輸出維度（avgpool 後是 1D）
        # mamba_vision_T: num_features = 80 * 2^3 = 640
        num_features = self.backbone.head.in_features \
            if hasattr(self.backbone.head, 'in_features') \
            else self._infer_feature_dim()

        # 新的分類頭
        self.head = nn.Linear(num_features, num_classes)

    def _infer_feature_dim(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = self.backbone.forward_features(dummy)
        return feat.shape[-1]

    def load_mae_encoder(self, mae_checkpoint_path):
        """
        從 MAE 預訓練的 checkpoint 載入 encoder 權重。
        使用方式：
            model = MambaVisionForClassification('mamba_vision_T', num_classes=2)
            model.load_mae_encoder('mae_pretrain.pth')
        """
        checkpoint = torch.load(mae_checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)

        # MAE checkpoint 裡 encoder 的 key 是 'encoder.*'
        encoder_state = {
            k.replace('encoder.', ''): v
            for k, v in state_dict.items()
            if k.startswith('encoder.')
        }
        missing, unexpected = self.backbone.load_state_dict(encoder_state, strict=False)
        print(f"載入 MAE encoder 權重")
        print(f"  Missing keys : {missing}")
        print(f"  Unexpected   : {unexpected}")

    def forward(self, x):
        # forward_features 內部：patch_embed → levels → norm → avgpool → flatten
        feat = self.backbone.forward_features(x)  # [N, 640]
        return self.head(feat)


# ===========================================================================
# 模型工廠函數
# ===========================================================================

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_mamba_vision_tiny(**kwargs):
    """
    完整 MambaVision-Tiny 作為 Encoder 的 MAE。

    Encoder 輸出：[N, 49, 640]（7x7 個 patch，每個 640 維）
    Decoder：8 層 Transformer，decoder_embed_dim=512
    patch_size：32（對應原圖 32x32 像素，7x7=49 個 patch）
    Masking：在 Encoder 輸出後才做
    """
    return MaskedAutoencoderMamba(
        mamba_type='mamba_vision_T',
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )


# 別名
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b