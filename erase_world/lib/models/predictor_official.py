from __future__ import annotations

import math
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, std=std)


def apply_masks(x: torch.Tensor, masks: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    """Gather patch tokens by index lists (official I-JEPA)."""
    if not isinstance(masks, list):
        masks = [masks]
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    return torch.cat(all_x, dim=0)


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)

    def get_1d(grid_axis: np.ndarray, dim: int) -> np.ndarray:
        omega = np.arange(dim // 2, dtype=np.float64)
        omega /= dim / 2.0
        omega = 1.0 / 10000**omega
        out = np.einsum("m,d->md", grid_axis.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    emb_h = get_1d(grid[0], embed_dim // 2)
    emb_w = get_1d(grid[1], embed_dim // 2)
    return np.concatenate([emb_h, emb_w], axis=1).astype(np.float32)


class _MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc2(self.drop(self.act(self.fc1(x))))
        return x


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class OfficialIJEPAPredictor(nn.Module):
    """Meta official VisionTransformerPredictor (ViT-H/14, pred_emb=384, depth=12)."""

    def __init__(
        self,
        num_patches: int = 256,
        embed_dim: int = 1280,
        predictor_embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        _trunc_normal_(self.mask_token, std=0.02)

        grid_size = int(num_patches**0.5)
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False
        )
        pos = _get_2d_sincos_pos_embed(predictor_embed_dim, grid_size)
        self.predictor_pos_embed.data.copy_(torch.from_numpy(pos).unsqueeze(0))

        self.predictor_blocks = nn.ModuleList(
            [_Block(predictor_embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        masks_x: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """x: context encoder tokens [B, N_ctxt, D]; masks_x/masks: [B, N] indices."""
        if masks_x.ndim == 1:
            masks_x = masks_x.unsqueeze(0)
        if masks.ndim == 1:
            masks = masks.unsqueeze(0)

        b = masks_x.shape[0]
        x = self.predictor_embed(x)
        x_pos = self.predictor_pos_embed.repeat(b, 1, 1)
        x = x + apply_masks(x_pos, masks_x)
        _, n_ctxt, _ = x.shape

        pos_embs = apply_masks(self.predictor_pos_embed.repeat(b, 1, 1), masks)
        pred_tokens = self.mask_token.expand(pos_embs.size(0), pos_embs.size(1), -1) + pos_embs
        x = torch.cat([x, pred_tokens], dim=1)

        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)
        x = x[:, n_ctxt:]
        return self.predictor_proj(x)


def build_official_predictor(
    embed_dim: int = 1280,
    grid_size: int = 16,
    pred_depth: int = 12,
    pred_emb_dim: int = 384,
    num_heads: int = 16,
) -> OfficialIJEPAPredictor:
    return OfficialIJEPAPredictor(
        num_patches=grid_size * grid_size,
        embed_dim=embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=num_heads,
    )


def _strip_module_prefix(state_dict: dict) -> OrderedDict:
    return OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())


def load_official_predictor_weights(predictor: OfficialIJEPAPredictor, checkpoint_path: str) -> bool:
    path = Path(checkpoint_path)
    if not path.is_file():
        return False
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    pred_sd = ckpt.get("predictor")
    if pred_sd is None:
        print(f"[Erase-World] no 'predictor' key in {path}")
        return False
    pred_sd = _strip_module_prefix(pred_sd)
    missing, unexpected = predictor.load_state_dict(pred_sd, strict=True)
    if missing or unexpected:
        print(f"[Erase-World] predictor load failed missing={missing} unexpected={unexpected}")
        return False
    print(f"[Erase-World] loaded official I-JEPA predictor from {path}")
    return True
