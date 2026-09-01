from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


def _strip_module_prefix(state_dict: dict) -> dict:
    return OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())


def default_encoder_layers(num_blocks: int) -> List[int]:
    """Quartile block indices for multi-scale readout."""
    if num_blocks <= 1:
        return [0]
    return [
        max(0, num_blocks // 4 - 1),
        max(0, num_blocks // 2 - 1),
        max(0, 3 * num_blocks // 4 - 1),
        num_blocks - 1,
    ]


def resolve_encoder_layers(layer_indices: Optional[List[int]], num_blocks: int) -> List[int]:
    """Map user block indices to valid [0, num_blocks-1], deduplicated and sorted."""
    if not layer_indices:
        return default_encoder_layers(num_blocks)
    out: List[int] = []
    for idx in layer_indices:
        i = min(max(int(idx), 0), num_blocks - 1)
        if not out or out[-1] != i:
            out.append(i)
    return out


def _patch_tokens(tokens: torch.Tensor, num_patches: int) -> torch.Tensor:
    if tokens.shape[1] > num_patches:
        tokens = tokens[:, 1 : num_patches + 1]
    return tokens


class IJEPAEncoder(nn.Module):
    """Frozen I-JEPA / ViT encoder with optional multi-block readout."""

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        patch_size: int,
        num_blocks: int,
        backend: str,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_blocks = num_blocks
        self.backend = backend

    def _num_patches(self, x: torch.Tensor) -> int:
        return (x.shape[-1] // self.patch_size) ** 2

    def encode_last(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_layers(x, [self.num_blocks - 1])[0]

    def encode_layers(self, x: torch.Tensor, layer_indices: List[int]) -> List[torch.Tensor]:
        if self.backend == "hf":
            return self._encode_layers_hf(x, layer_indices)
        return self._encode_layers_timm(x, layer_indices)

    def _encode_layers_hf(self, x: torch.Tensor, layer_indices: List[int]) -> List[torch.Tensor]:
        train_size = getattr(getattr(self.backbone, "config", None), "image_size", 224)
        if isinstance(train_size, (list, tuple)):
            train_size = train_size[0] if train_size else 224
        interpolate = x.shape[-1] != train_size or x.shape[-2] != train_size
        out = self.backbone(
            pixel_values=x,
            output_hidden_states=True,
            interpolate_pos_encoding=interpolate,
        )
        hidden_states = out.hidden_states
        num_patches = self._num_patches(x)
        feats: List[torch.Tensor] = []
        for block_idx in layer_indices:
            # hidden_states[0] = embeddings; hidden_states[i+1] = after block i
            tokens = hidden_states[block_idx + 1]
            feats.append(_patch_tokens(tokens, num_patches))
        return feats

    def _encode_layers_timm(self, x: torch.Tensor, layer_indices: List[int]) -> List[torch.Tensor]:
        vit = self.backbone
        num_patches = self._num_patches(x)
        want = set(layer_indices)

        x = vit.patch_embed(x)
        if hasattr(vit, "cls_token"):
            cls = vit.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
        if hasattr(vit, "pos_embed"):
            x = x + vit.pos_embed[:, : x.shape[1]]
        x = vit.pos_drop(x)

        feats: dict[int, torch.Tensor] = {}
        for i, blk in enumerate(vit.blocks):
            x = blk(x)
            if i in want:
                feats[i] = _patch_tokens(x, num_patches)
        return [feats[i] for i in layer_indices]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_last(x)

    def unfreeze_last_blocks(self, n: int) -> List[nn.Parameter]:
        """Unfreeze the last n transformer blocks; return trainable params."""
        if n <= 0:
            return []
        params: List[nn.Parameter] = []
        if self.backend == "hf":
            vit = self.backbone
            blocks = None
            if hasattr(vit, "layers"):
                blocks = vit.layers
            elif hasattr(vit, "encoder") and hasattr(vit.encoder, "layer"):
                blocks = vit.encoder.layer
            elif hasattr(vit, "vision_model") and hasattr(vit.vision_model, "encoder"):
                blocks = vit.vision_model.encoder.layers
            if blocks is None:
                print("[Erase-World] warn: could not locate HF encoder blocks for unfreeze")
                return []
            for blk in list(blocks)[-n:]:
                for p in blk.parameters():
                    p.requires_grad = True
                    params.append(p)
        else:
            for blk in self.backbone.blocks[-n:]:
                for p in blk.parameters():
                    p.requires_grad = True
                    params.append(p)
        if params:
            print(f"[Erase-World] unfroze last {n} encoder block(s), {len(params)} tensors")
        return params


def load_ijepa_encoder(
    model_name: str = "facebook/ijepa_vith14_1k",
    checkpoint_path: Optional[str] = None,
    img_size: int = 224,
    freeze: bool = True,
) -> Tuple[nn.Module, int, int, int]:
    """
    Returns (encoder, embed_dim, patch_size, num_blocks).
    Tries: local checkpoint -> HuggingFace -> timm ViT fallback.
    """
    if checkpoint_path and Path(checkpoint_path).is_file():
        return _load_official_checkpoint(checkpoint_path, img_size, freeze)

    try:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        print(f"[Erase-World] loaded I-JEPA encoder: {model_name} ({model.__class__.__name__})")
        patch_size = getattr(getattr(model, "config", None), "patch_size", 14)
        hidden = getattr(getattr(model, "config", None), "hidden_size", 1280)
        num_blocks = getattr(getattr(model, "config", None), "num_hidden_layers", 32)

        enc = IJEPAEncoder(model, hidden, patch_size, num_blocks, backend="hf")
        if freeze:
            enc.eval()
            for p in enc.parameters():
                p.requires_grad = False
        return enc, hidden, patch_size, num_blocks
    except Exception as exc:
        print(f"[Erase-World] HF load failed ({exc}); falling back to timm vit_base_patch16_224")

    import timm

    backbone = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    patch_size = backbone.patch_embed.patch_size[0]
    hidden = backbone.embed_dim
    num_blocks = len(backbone.blocks)

    enc = IJEPAEncoder(backbone, hidden, patch_size, num_blocks, backend="timm")
    if freeze:
        enc.eval()
        for p in enc.parameters():
            p.requires_grad = False
    return enc, hidden, patch_size, num_blocks


def _load_official_checkpoint(path: str, img_size: int, freeze: bool) -> Tuple[nn.Module, int, int, int]:
    """Load facebookresearch/ijepa .pth.tar (encoder + optional predictor)."""
    import timm

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
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

    backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
    backbone.load_state_dict(enc_sd, strict=False)
    patch_size = backbone.patch_embed.patch_size[0]
    hidden = backbone.embed_dim
    num_blocks = len(backbone.blocks)

    enc = IJEPAEncoder(backbone, hidden, patch_size, num_blocks, backend="timm")
    if freeze:
        enc.eval()
        for p in enc.parameters():
            p.requires_grad = False
    return enc, hidden, patch_size, num_blocks


def gather_tokens(tokens: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """tokens [B,N,D], idx [B,K] -> [B,K,D]."""
    b, _, d = tokens.shape
    idx_exp = idx.unsqueeze(-1).expand(b, idx.shape[1], d)
    return torch.gather(tokens, 1, idx_exp)
