# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# --------------------------------------------------------
# 修改說明：
#   將原版 ViT backbone 換成 MambaVision-T。
#   - levels[0], levels[1]：CNN stages（4D 模式）
#   - levels[2], levels[3]：Mamba stages（3D token 模式）
#   - levels[2] 的 Downsample.token_reduction 負責 320 → 640
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block as ViTBlock
from util.pos_embed import get_2d_sincos_pos_embed
from models import mamba_vision

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)#qk_scale=None拿掉了
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            ViTBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)#拿掉了 qk_scale=None
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()
       

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


class MaskedAutoencoderMambaVision(nn.Module):
    """Masked Autoencoder with MambaVision-T backbone."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        encoder_dim=320,         # levels[1] 輸出 channel
        encoder_out_dim=640,     # levels[3] 輸出 channel（經過 token_reduction）
        num_patches=196,         # 14 × 14
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        pretrained=True,#我改成True
    ):
        super().__init__()

        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_patches = num_patches
        self.encoder_dim = encoder_dim
        self.encoder_out_dim = encoder_out_dim

        # ------------------------------------------------------------------
        # Backbone
        # ------------------------------------------------------------------
        self.backbone = mamba_vision.mamba_vision_T(pretrained=pretrained)
        self.backbone.head = nn.Identity()

        # ------------------------------------------------------------------
        # Encoder 輔助參數
        # pos_embed 含 cls token（shape: 1, 197, 320）
        # ------------------------------------------------------------------
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, encoder_dim),
            requires_grad=False,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))

        # encoder 最後的 norm 接 encoder_out_dim（640）
        self.norm = norm_layer(encoder_out_dim)

        # ------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------
        # encoder_out_dim (640) → decoder_embed_dim (512)
        self.decoder_embed = nn.Linear(encoder_out_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # decoder pos_embed 含 cls token（shape: 1, 197, 512）
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False,
        )

        self.decoder_blocks = nn.ModuleList([
            ViTBlock(
                decoder_embed_dim,
                decoder_num_heads,
                mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
            )
            for _ in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        # 每個 patch = 16×16×3 = 768
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size ** 2 * in_chans, bias=True
        )

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

        # =====================================================================
        # 正確的位置在這邊！【第一階段：MAE 預訓練】的 Tokenizer 凍結 
        # =====================================================================
        print("====== [MAE Pretrain] 正在鎖定 MambaVision 前段 CNN Tokenizer 權重 ======")
        
        # 1. 凍結最前面的 Patch Embedding 層
        for param in self.backbone.patch_embed.parameters():
            param.requires_grad = False
            
        # 2. 凍結 Stage 1 (levels[0])
        for param in self.backbone.levels[0].parameters():
            param.requires_grad = False
            
        # 3. 凍結 Stage 2 (levels[1])
        for param in self.backbone.levels[1].parameters():
            param.requires_grad = False
            
        print("====== [MAE Pretrain] 鎖定成功！僅訓練後段 Mamba Stages (levels[2, 3]) 與 Decoder ======")
        # =====================================================================

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def initialize_weights(self):
        # encoder pos_embed（含 cls token）
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.num_patches ** 0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        # decoder pos_embed（含 cls token）
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches ** 0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ------------------------------------------------------------------
    # Patchify / Unpatchify
    # ------------------------------------------------------------------
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        return: (N, L, patch_size**2 * 3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * 3)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * 3)
        return: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], 3, h * p, h * p)

    # ------------------------------------------------------------------
    # Random Masking
    # ------------------------------------------------------------------
    def random_masking(self, x, mask_ratio):
        """
        x: (N, L, D)
        return: x_masked, mask, ids_restore
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )

        mask = torch.ones(N, L, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------
    def forward_encoder(self, x, mask_ratio):
        backbone = self.backbone

        # Stage1, Stage2：CNN，4D feature map
        x = backbone.patch_embed(x)    # (B,  80, 56, 56)
        x = backbone.levels[0](x)      # (B, 160, 28, 28)
        x = backbone.levels[1](x)      # (B, 320, 14, 14)

        # 4D → 3D token
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)            # (B, 196, 320)

        # 加 pos_embed（不含 cls token 那格）
        x = x + self.pos_embed[:, 1:, :]            # (B, 196, 320)

        # Random masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)  # (B, 49, 320)

        # 加 cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)       # (B, 50, 320)

        # Stage3：Mamba 3D token mode
        # Downsample.token_reduction 在這裡把 320 → 640
        x = backbone.levels[2](x)                   # (B, 50, 640)

        # Stage4：Mamba 3D token mode，無 downsample
        x = backbone.levels[3](x)                   # (B, 50, 640)

        x = self.norm(x)                            # norm(640)
        return x, mask, ids_restore

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------
    def forward_decoder(self, x, ids_restore):
        # 640 → 512
        x = self.decoder_embed(x)                   # (B, 50, 512)

        # 補回 mask token
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # 去掉 cls
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )                                            # (B, 196, 512)
        x = torch.cat([x[:, :1, :], x_], dim=1)    # 補回 cls → (B, 197, 512)

        # 加 decoder pos_embed
        x = x + self.decoder_pos_embed

        # Transformer decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Prediction head
        x = self.decoder_pred(x)                    # (B, 197, 768)

        # 去掉 cls token
        x = x[:, 1:, :]                             # (B, 196, 768)
        return x

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)                     # (N, L)
        loss = (loss * mask).sum() / mask.sum()      # masked patches only
        return loss

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


# ------------------------------------------------------------------
# 建構函式
# ------------------------------------------------------------------

def mae_mamba_vision_T_dec512d8b(**kwargs):
    model = MaskedAutoencoderMambaVision(
        img_size=224,
        patch_size=16,
        in_chans=3,
        encoder_dim=320,
        encoder_out_dim=640,
        num_patches=196,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


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

mae_mamba_vision_T = mae_mamba_vision_T_dec512d8b

# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
