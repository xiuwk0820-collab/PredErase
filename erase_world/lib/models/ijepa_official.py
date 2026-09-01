from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F

from .ijepa_encoder import IJEPAEncoder, gather_tokens, load_ijepa_encoder

from .predictor_official import (
    build_official_predictor,
    load_official_predictor_weights,
)


IJEPA_MEAN = (0.5, 0.5, 0.5)
IJEPA_STD = (0.5, 0.5, 0.5)


def _strip_module_prefix(state_dict: dict) -> dict:
    return OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())


def load_official_encoder_from_checkpoint(
    checkpoint_path: str,
    img_size: int = 224,
    freeze: bool = True,
) -> tuple[IJEPAEncoder, int, int, int]:
    """Load encoder weights from official I-JEPA .pth.tar (paired with predictor)."""
    import timm

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    enc_sd = ckpt.get("target_encoder", ckpt.get("encoder", ckpt))
    enc_sd = _strip_module_prefix(enc_sd)

    if any("blocks.39" in k for k in enc_sd):
        model_name = "vit_gigantic_patch16_224"
    elif any("blocks.31" in k for k in enc_sd):
        model_name = "vit_huge_patch14_224"
    elif any("blocks.23" in k for k in enc_sd):
        model_name = "vit_large_patch14_224"
    else:
        model_name = "vit_base_patch16_224"

    backbone = timm.create_model(model_name, pretrained=False, num_classes=0, img_size=img_size)
    if "pos_embed" in enc_sd and enc_sd["pos_embed"].shape != backbone.pos_embed.shape:
        ckpt_pe = enc_sd["pos_embed"]
        model_pe = backbone.pos_embed
        if ckpt_pe.shape[1] + 1 == model_pe.shape[1] and ckpt_pe.shape[-1] == model_pe.shape[-1]:
            enc_sd = dict(enc_sd)
            enc_sd["pos_embed"] = torch.cat([model_pe[:, :1].clone(), ckpt_pe], dim=1)
            print("[Erase-World] prepended CLS to checkpoint pos_embed for timm encoder")

    missing, unexpected = backbone.load_state_dict(enc_sd, strict=False)
    if missing:
        print(f"[Erase-World] encoder missing keys: {len(missing)}")
    if unexpected:
        print(f"[Erase-World] encoder unexpected keys: {len(unexpected)}")

    patch_size = backbone.patch_embed.patch_size[0]
    hidden = backbone.embed_dim
    num_blocks = len(backbone.blocks)
    enc = IJEPAEncoder(backbone, hidden, patch_size, num_blocks, backend="timm")
    if freeze:
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)
    print(f"[Erase-World] loaded official I-JEPA encoder from {checkpoint_path}")
    return enc, hidden, patch_size, num_blocks


def mask_to_patch_indices(
    mask: torch.Tensor,
    grid_h: int,
    grid_w: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixel mask [1,H,W] or [H,W] -> visible / masked flat patch indices."""
    if mask.ndim == 3:
        mask = mask[0]
    m = mask.float().unsqueeze(0).unsqueeze(0)
    patch_mask = F.adaptive_max_pool2d(m, (grid_h, grid_w))[0, 0]
    flat = patch_mask.reshape(-1)
    masked_idx = (flat > 0.5).nonzero(as_tuple=False).squeeze(-1)
    visible_idx = (flat <= 0.5).nonzero(as_tuple=False).squeeze(-1)
    return visible_idx.to(device), masked_idx.to(device)


def preprocess_ijepa(image: torch.Tensor, img_size: int) -> torch.Tensor:
    """[B,3,H,W] in [0,1] -> normalized I-JEPA input."""
    x = F.interpolate(image, size=(img_size, img_size), mode="bilinear", align_corners=False)
    mean = torch.tensor(IJEPA_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IJEPA_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def build_visible_fill(image: torch.Tensor, mask: torch.Tensor, fill: float = 0.5) -> torch.Tensor:
    """Keep visible pixels; fill masked region with constant (I-JEPA context)."""
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    if mask.shape[1] == 1:
        mask = mask.expand(-1, 3, -1, -1)
    return image * (1.0 - mask) + fill * mask


class OfficialIJEPAModel(torch.nn.Module):
    """Frozen official I-JEPA encoder + predictor for representation guidance."""

    def __init__(
        self,
        model_name: str = "facebook/ijepa_vith14_1k",
        checkpoint_path: str | None = None,
        img_size: int = 224,
        device: str = "cuda",
    ):
        super().__init__()
        self.img_size = img_size
        self.device = torch.device(device)

        if checkpoint_path and Path(checkpoint_path).is_file():
            encoder, embed_dim, patch_size, num_blocks = load_official_encoder_from_checkpoint(
                checkpoint_path, img_size=img_size, freeze=True
            )
        else:
            encoder, embed_dim, patch_size, num_blocks = load_ijepa_encoder(
                model_name=model_name,
                checkpoint_path=None,
                img_size=img_size,
                freeze=True,
            )
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size

        heads = 16 if embed_dim == 1280 else (12 if embed_dim % 12 == 0 else 8)
        self.predictor = build_official_predictor(
            embed_dim=embed_dim,
            grid_size=self.grid_size,
            pred_depth=12,
            pred_emb_dim=384,
            num_heads=heads,
        )
        self.has_predictor = False
        if checkpoint_path:
            self.has_predictor = load_official_predictor_weights(self.predictor, checkpoint_path)

        grid_n = self.grid_size * self.grid_size
        self.pos_embed = torch.nn.Parameter(
            torch.zeros(1, grid_n, embed_dim), requires_grad=False
        )
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.to(self.device)
        self.eval()
        self._light_layer_ids = [max(0, min(num_blocks - 1, i)) for i in [2, 3, 4]]

    def set_shallow_light_layers(self, layer_ids: list[int]) -> None:
        n_blk = self.encoder.num_blocks
        self._light_layer_ids = [max(0, min(n_blk - 1, i)) for i in layer_ids]
        for p in self.parameters():
            p.requires_grad_(False)

    def encode_patches(self, image: torch.Tensor) -> torch.Tensor:
        x = preprocess_ijepa(image, self.img_size)
        return self.encoder.encode_last(x)

    @torch.no_grad()
    def encode_patches_frozen(self, image: torch.Tensor) -> torch.Tensor:
        return self.encode_patches(image)

    def predict_masked_tokens(
        self,
        patch_tokens: torch.Tensor,
        visible_idx: torch.Tensor,
        masked_idx: torch.Tensor,
    ) -> torch.Tensor:
        if not self.has_predictor:
            raise RuntimeError("predict_masked_tokens requires official predictor weights")
        ctxt = gather_tokens(patch_tokens, visible_idx.unsqueeze(0))
        return self.predictor(ctxt, visible_idx.unsqueeze(0), masked_idx.unsqueeze(0))

    @torch.no_grad()
    def precompute_target(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns e_target, visible_idx, masked_idx, e_vis_target."""
        vis_fill = build_visible_fill(image, mask)
        tokens = self.encode_patches_frozen(vis_fill)
        visible_idx, masked_idx = mask_to_patch_indices(
            mask, self.grid_size, self.grid_size, image.device
        )
        if masked_idx.numel() == 0:
            raise ValueError("Mask is empty — nothing to inpaint.")
        if visible_idx.numel() == 0:
            raise ValueError("Mask covers entire image — no visible context for I-JEPA.")

        if self.has_predictor:
            e_target = self.predict_masked_tokens(tokens, visible_idx, masked_idx)
        else:
            full_tokens = self.encode_patches_frozen(image)
            e_target = gather_tokens(full_tokens, masked_idx.unsqueeze(0))
        e_vis_target = gather_tokens(tokens, visible_idx.unsqueeze(0)).squeeze(0)
        return e_target.squeeze(0), visible_idx, masked_idx, e_vis_target

    @torch.no_grad()
    def precompute_counterfactual_causal(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Ori vs counterfactual patch repr (no geometry rules). Returns diff + targets."""
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        h, w = image.shape[-2], image.shape[-1]
        grid = self.grid_size
        if mask.shape[-2:] != (h, w):
            mask = F.interpolate(mask.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1)

        m = (mask > 0.5).float()
        i_vis = image * (1.0 - m.unsqueeze(1).expand_as(image))

        visible_idx, masked_idx = mask_to_patch_indices(m, grid, grid, image.device)

        tok_ori = self.encode_patches_frozen(image).squeeze(0)
        vis_fill = build_visible_fill(i_vis, m)
        tok_cf_in = self.encode_patches_frozen(vis_fill).squeeze(0)
        e_cf_struct = self.predict_masked_tokens(
            tok_cf_in.unsqueeze(0), visible_idx, masked_idx
        ).squeeze(0)

        tok_cf = tok_ori.clone()
        tok_cf[masked_idx] = e_cf_struct

        sh_ori = self.encode_shallow_light(image).squeeze(0)
        sh_cf = self.encode_shallow_light(vis_fill).squeeze(0)

        diff_s = (tok_ori - tok_cf).pow(2).sum(dim=-1).sqrt()
        diff_l = (sh_ori - sh_cf).pow(2).sum(dim=-1).sqrt()

        return {
            "diff_struct": diff_s,
            "diff_light": diff_l,
            "e_cf_struct": e_cf_struct,
            "e_cf_shallow_full": sh_cf,
            "visible_idx": visible_idx,
            "masked_idx": masked_idx,
            "i_vis": i_vis,
            "patch_grid": grid,
        }

    def mask_region_repr(
        self,
        image: torch.Tensor,
        visible_idx: torch.Tensor,
        masked_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable path for guidance: image -> E_curr on mask patches."""
        tokens = self.encode_patches(image)
        if self.has_predictor:
            pred = self.predict_masked_tokens(tokens, visible_idx, masked_idx)
            return pred.squeeze(0)
        return gather_tokens(tokens, masked_idx.unsqueeze(0)).squeeze(0)

    def masked_encoder_repr(self, image: torch.Tensor, masked_idx: torch.Tensor) -> torch.Tensor:
        """Direct encoder tokens on masked patches (D-JEPA g_i proxy at test time)."""
        tokens = self.encode_patches(image)
        return gather_tokens(tokens, masked_idx.unsqueeze(0)).squeeze(0)

    def visible_encoder_repr(self, image: torch.Tensor, visible_idx: torch.Tensor) -> torch.Tensor:
        tokens = self.encode_patches(image)
        return gather_tokens(tokens, visible_idx.unsqueeze(0)).squeeze(0)

    @property
    def shallow_block_idx(self) -> int:
        return max(1, int(self.encoder.num_blocks * getattr(self, "_light_ratio", 0.25)) - 1)

    def encode_shallow_light(self, image: torch.Tensor) -> torch.Tensor:
        """Shallow encoder tokens (layers 3–5), layer-norm — brightness / texture."""
        x = preprocess_ijepa(image, self.img_size)
        layer_ids = getattr(self, "_light_layer_ids", [2, 3, 4])
        feats = self.encoder.encode_layers(x, layer_ids)
        stacked = torch.stack([f.squeeze(0) for f in feats], dim=0).mean(dim=0)
        return F.layer_norm(stacked, (stacked.shape[-1],))

    def encode_deep_struct(self, image: torch.Tensor) -> torch.Tensor:
        """Deep encoder tokens (last 3 blocks), layer-norm — geometry / structure."""
        x = preprocess_ijepa(image, self.img_size)
        n_blk = self.encoder.num_blocks
        layer_ids = [max(0, n_blk - 3), max(0, n_blk - 2), n_blk - 1]
        feats = self.encoder.encode_layers(x, layer_ids)
        stacked = torch.stack([f.squeeze(0) for f in feats], dim=0).mean(dim=0)
        return F.layer_norm(stacked, (stacked.shape[-1],))

    def deep_masked_repr(self, image: torch.Tensor, masked_idx: torch.Tensor) -> torch.Tensor:
        tokens = self.encode_deep_struct(image)
        return gather_tokens(tokens.unsqueeze(0), masked_idx.unsqueeze(0)).squeeze(0)

    @torch.no_grad()
    def shallow_patch_darkness(
        self,
        image: torch.Tensor,
        reference_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-patch relative darkness from shallow tokens [grid, grid]."""
        if image.ndim == 3:
            image = image.unsqueeze(0)
        tokens = self.encode_shallow_light(image)
        grid = self.grid_size

        if reference_mask is not None:
            if reference_mask.ndim == 2:
                reference_mask = reference_mask.unsqueeze(0)
            _, vis_idx = mask_to_patch_indices(
                reference_mask.squeeze(0), grid, grid, image.device
            )
            ref = tokens[vis_idx].mean(0) if vis_idx.numel() > 0 else tokens.mean(0)
        else:
            ref = tokens.mean(0)

        ref_n = F.normalize(ref.unsqueeze(0), dim=-1)
        tok_n = F.normalize(tokens, dim=-1)
        token_dark = (1.0 - (tok_n * ref_n).sum(-1)).clamp(0.0, 1.0)

        gray = preprocess_ijepa(image, self.img_size).mean(dim=1, keepdim=True)
        patch_gray = F.adaptive_avg_pool2d(gray, (grid, grid))[0, 0].reshape(-1)
        g_min, g_max = patch_gray.min(), patch_gray.max()
        gray_dark = ((g_max - patch_gray) / (g_max - g_min + 1e-6)).clamp(0.0, 1.0)

        dark = (0.55 * token_dark + 0.45 * gray_dark).reshape(grid, grid)
        return dark

    def precompute_global_light(
        self,
        image: torch.Tensor,
        protect_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """E_light from native visible (excludes M_obj ∪ M_shadow — keeps tree/railing shadows)."""
        if protect_mask.ndim == 2:
            protect_mask = protect_mask.unsqueeze(0)
        vis_fill = build_visible_fill(image, protect_mask, fill=0.5)
        tokens = self.encode_shallow_light(vis_fill)
        visible_idx, _ = mask_to_patch_indices(
            protect_mask, self.grid_size, self.grid_size, image.device
        )
        # visible_idx = patches OUTSIDE protect (native scene)
        if visible_idx.numel() == 0:
            visible_idx = torch.arange(tokens.shape[0], device=tokens.device)
        e_vis = gather_tokens(tokens.unsqueeze(0), visible_idx.unsqueeze(0)).squeeze(0)
        e_light = e_vis.mean(dim=0)
        return e_light.detach(), visible_idx

    def shallow_region_mean(
        self, image: torch.Tensor, patch_idx: torch.Tensor
    ) -> torch.Tensor | None:
        if patch_idx is None or patch_idx.numel() == 0:
            return None
        tokens = self.encode_shallow_light(image)
        return gather_tokens(tokens.unsqueeze(0), patch_idx.unsqueeze(0)).mean(dim=1).squeeze(0)
