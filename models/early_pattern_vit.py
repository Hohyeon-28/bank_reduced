import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .early_vit_partial_routing import VisionTransformer, Block


DEFAULT_6_PATTERNS = {
    "early6": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    "mid6": [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    "late6": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    "alternating6": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    "edge6": [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    "trio6": [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    "spread6": [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1],
    "center_tail6": [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
}

DEFAULT_8_PATTERNS = {
    "early8": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    "mid8": [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    "late8": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    "alternating8": [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
    "full12": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
}

DEFAULT_PATTERN_BANKS = {
    "patterns_6": DEFAULT_6_PATTERNS,
    "patterns_8": DEFAULT_8_PATTERNS,
    "patterns_mixed": {**DEFAULT_6_PATTERNS, **DEFAULT_8_PATTERNS},
    "bank2": {k: DEFAULT_6_PATTERNS[k] for k in ("early6", "late6")},
    "bank4": {k: DEFAULT_6_PATTERNS[k] for k in ("early6", "mid6", "late6", "alternating6")},
    "bank8": DEFAULT_6_PATTERNS,
}


def _simple_yaml_patterns(text: str) -> Dict[str, List[int]]:
    patterns = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        name, value = line.split(":", 1)
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            patterns[name.strip()] = [int(v.strip()) for v in value[1:-1].split(",") if v.strip()]
    return patterns


def load_pattern_bank(pattern_bank: Optional[Union[str, Dict[str, Sequence[int]]]]) -> Tuple[List[str], torch.Tensor]:
    if pattern_bank is None:
        patterns = DEFAULT_PATTERN_BANKS["bank4"]
    elif isinstance(pattern_bank, dict):
        patterns = {k: list(v) for k, v in pattern_bank.items()}
    else:
        path = Path(pattern_bank)
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if path.suffix.lower() == ".json":
                patterns = json.loads(text)
            else:
                try:
                    import yaml
                    patterns = yaml.safe_load(text)
                except Exception:
                    patterns = _simple_yaml_patterns(text)
        else:
            key = str(pattern_bank).replace(".yml", "").replace(".yaml", "")
            patterns = DEFAULT_PATTERN_BANKS[key]

    names = list(patterns.keys())
    tensor = torch.tensor([patterns[name] for name in names], dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError("Pattern bank must be a name -> 1D pattern mapping.")
    if not torch.all((tensor == 0) | (tensor == 1)):
        raise ValueError("Patterns must contain only 0/1 values.")
    return names, tensor


def pattern_metadata(names: Sequence[str], patterns: torch.Tensor) -> List[Dict[str, object]]:
    metas = []
    depth = patterns.shape[1]
    for name, row in zip(names, patterns):
        active = [int(i) for i, v in enumerate(row.tolist()) if v > 0.5]
        metas.append({
            "pattern_name": name,
            "active_mlp_count": len(active),
            "active_mlp_ratio": len(active) / depth,
            "active_layer_indices": active,
        })
    return metas


class PatternMLPBlock(Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_active_mlp_count = 0

    def forward(self, x: torch.Tensor, pattern_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))

        active_idx = (pattern_mask > 0.5).nonzero(as_tuple=True)[0]
        self.last_active_mlp_count = int(active_idx.numel())
        if active_idx.numel() == 0:
            return x

        x_active = x.index_select(0, active_idx).contiguous()
        mlp_residual = self.drop_path2(self.ls2(self.mlp(self.norm2(x_active))))
        active_gate = pattern_mask.index_select(0, active_idx).to(dtype=x.dtype)
        active_gate = active_gate.view(-1, *([1] * (x.ndim - 1)))
        x_active = x_active + active_gate * mlp_residual

        x_out = x.clone()
        x_out.index_copy_(0, active_idx, x_active)
        return x_out


class PatternMLPViT(VisionTransformer):
    def __init__(
            self,
            pattern_bank: Optional[Union[str, Dict[str, Sequence[int]]]] = None,
            pattern_mode: str = "router",
            fixed_pattern: Optional[str] = None,
            router_tau: float = 1.0,
            router_init: str = "zero_logits",
            router_probe_blocks: int = 0,
            router_random_start_prob: float = 0.0,
            router_random_mid_prob: float = 0.0,
            router_random_end_prob: float = 0.0,
            **kwargs):
        # timm.create_model forwards registry/build metadata that this local
        # VisionTransformer implementation does not accept directly.
        for key in (
                "pretrained_cfg", "pretrained_cfg_overlay", "cache_dir",
                "features_only", "out_indices", "scriptable", "exportable",
                "no_jit", "bn_momentum", "bn_eps", "drop_connect_rate"):
            kwargs.pop(key, None)
        super().__init__(block_fn=PatternMLPBlock, **kwargs)
        names, patterns = load_pattern_bank(pattern_bank)
        if patterns.shape[1] != len(self.blocks):
            raise ValueError(f"Pattern length ({patterns.shape[1]}) must match depth ({len(self.blocks)}).")
        if pattern_mode not in ("fixed", "sampled_supernet", "sampled_uniform", "sampled_quota", "router"):
            raise ValueError("pattern_mode must be one of fixed, sampled_supernet, sampled_uniform, sampled_quota, router.")
        if router_probe_blocks < 0 or router_probe_blocks >= len(self.blocks):
            raise ValueError("router_probe_blocks must be in [0, depth - 1].")

        self.pattern_names = names
        self.register_buffer("patterns", patterns)
        self.num_patterns = len(names)
        self.pattern_metadata = pattern_metadata(names, patterns)
        self.pattern_mode = pattern_mode
        self.fixed_pattern = fixed_pattern or names[0]
        self.router_tau = router_tau
        self.router_probe_blocks = router_probe_blocks
        self.router_random_start_prob = float(router_random_start_prob)
        self.router_random_mid_prob = float(router_random_mid_prob)
        self.router_random_end_prob = float(router_random_end_prob)
        self.training_epoch = 0
        self.training_epochs = 1
        self.pattern_router = nn.Linear(self.embed_dim, self.num_patterns)
        self._sample_cursor = 0
        self._last_routing: Optional[Dict[str, torch.Tensor]] = None
        self._init_pattern_router(router_init)

    def _init_pattern_router(self, router_init: str) -> None:
        if router_init == "default":
            return
        if router_init != "zero_logits":
            raise ValueError("router_init must be default or zero_logits.")
        nn.init.zeros_(self.pattern_router.weight)
        nn.init.zeros_(self.pattern_router.bias)

    def _pattern_index(self, name: str) -> int:
        if name not in self.pattern_names:
            raise ValueError(f"Unknown pattern {name}. Available: {self.pattern_names}")
        return self.pattern_names.index(name)

    def set_training_progress(self, epoch: int, num_epochs: int) -> None:
        self.training_epoch = int(epoch)
        self.training_epochs = max(int(num_epochs), 1)

    def _quota_indices(self, bsz: int, device: torch.device) -> torch.Tensor:
        repeats = (bsz + self.num_patterns - 1) // self.num_patterns
        idx = torch.arange(self.num_patterns, device=device, dtype=torch.long).repeat(repeats)[:bsz]
        return idx[torch.randperm(bsz, device=device)]

    def _router_random_prob(self) -> float:
        if self.training_epochs <= 1:
            frac = 1.0
        else:
            frac = self.training_epoch / max(self.training_epochs - 1, 1)
        if frac < 0.3:
            return self.router_random_start_prob
        if frac < 0.7:
            return self.router_random_mid_prob
        return self.router_random_end_prob

    def _select_patterns(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz = x.shape[0]
        device = x.device
        if self.pattern_mode == "fixed":
            idx = torch.full((bsz,), self._pattern_index(self.fixed_pattern), device=device, dtype=torch.long)
            selection_probs = F.one_hot(idx, num_classes=self.num_patterns).to(dtype=x.dtype)
            soft_probs = selection_probs
            logits = selection_probs
        elif self.pattern_mode == "sampled_supernet":
            if self.training:
                idx_val = self._sample_cursor % self.num_patterns
                self._sample_cursor += 1
            else:
                idx_val = self._pattern_index(self.fixed_pattern)
            idx = torch.full((bsz,), idx_val, device=device, dtype=torch.long)
            selection_probs = F.one_hot(idx, num_classes=self.num_patterns).to(dtype=x.dtype)
            soft_probs = selection_probs
            logits = selection_probs
        elif self.pattern_mode == "sampled_uniform":
            if self.training:
                idx = torch.randint(self.num_patterns, (bsz,), device=device, dtype=torch.long)
            else:
                idx = torch.full((bsz,), self._pattern_index(self.fixed_pattern), device=device, dtype=torch.long)
            selection_probs = F.one_hot(idx, num_classes=self.num_patterns).to(dtype=x.dtype)
            soft_probs = selection_probs
            logits = selection_probs
        elif self.pattern_mode == "sampled_quota":
            if self.training:
                idx = self._quota_indices(bsz, device)
            else:
                idx = torch.full((bsz,), self._pattern_index(self.fixed_pattern), device=device, dtype=torch.long)
            selection_probs = F.one_hot(idx, num_classes=self.num_patterns).to(dtype=x.dtype)
            soft_probs = selection_probs
            logits = selection_probs
        else:
            logits = self.pattern_router(x.mean(dim=1))
            soft_probs = logits.softmax(dim=-1)
            random_prob = self._router_random_prob() if self.training else 0.0
            use_random = self.training and random_prob > 0 and random.random() < random_prob
            if use_random:
                idx = self._quota_indices(bsz, device)
                selection_probs = F.one_hot(idx, num_classes=self.num_patterns).to(dtype=x.dtype)
            elif self.training:
                selection_probs = F.gumbel_softmax(logits, tau=self.router_tau, hard=True, dim=-1)
                idx = selection_probs.argmax(dim=-1)
            else:
                idx = logits.argmax(dim=-1)
                selection_probs = F.one_hot(idx, num_classes=self.num_patterns).to(dtype=x.dtype)

        masks = selection_probs @ self.patterns.to(device=device, dtype=selection_probs.dtype)
        if self.router_probe_blocks:
            masks = masks.clone()
            masks[:, :self.router_probe_blocks] = 1.0

        active_counts = masks.sum(dim=1)
        soft_expected_counts = (soft_probs @ self.patterns.sum(dim=1).to(device=device, dtype=soft_probs.dtype)).view(-1)
        if self.router_probe_blocks:
            suffix = self.patterns[:, self.router_probe_blocks:].sum(dim=1).to(device=device, dtype=soft_probs.dtype)
            soft_expected_counts = self.router_probe_blocks + (soft_probs @ suffix).view(-1)

        return masks, {
            "pattern_idx": idx.detach(),
            "pattern_probs": soft_probs,
            "pattern_selection_probs": selection_probs.detach(),
            "pattern_logits": logits,
            "pattern_masks": masks.detach(),
            "active_mlp_count": active_counts.detach(),
            "expected_active_mlp": soft_expected_counts,
        }

    def forward_features(self, x: torch.Tensor, return_routing: bool = False):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        masks = None
        routing = None
        for i, blk in enumerate(self.blocks):
            if i == self.router_probe_blocks:
                masks, routing = self._select_patterns(x)
            if masks is None:
                full_mask = x.new_ones(x.shape[0])
                x = blk(x, pattern_mask=full_mask)
            else:
                x = blk(x, pattern_mask=masks[:, i])

        x = self.norm(x)
        if routing is None:
            masks, routing = self._select_patterns(x)
        self._last_routing = routing
        if return_routing:
            return x, routing
        return x

    def forward(self, x: torch.Tensor, return_routing: bool = False):
        if return_routing:
            x, routing = self.forward_features(x, return_routing=True)
            return self.forward_head(x, pre_logits=False), self.routing_info_to_python(routing)
        x = self.forward_features(x)
        return self.forward_head(x, pre_logits=False)

    def get_budget_loss(self, target_active_mlp: Optional[float], weight: float = 1.0) -> torch.Tensor:
        if not target_active_mlp or weight == 0 or self._last_routing is None:
            return next(self.parameters()).new_tensor(0.0)
        expected = self._last_routing["expected_active_mlp"].mean()
        return weight * (expected - float(target_active_mlp)).pow(2)

    def routing_info_to_python(self, routing: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, object]:
        routing = routing or self._last_routing
        if routing is None:
            return {}
        idx = routing["pattern_idx"].detach().cpu().tolist()
        names = [self.pattern_names[int(i)] for i in idx]
        probs = routing["pattern_probs"].detach().cpu()
        entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1)
        return {
            "pattern_idx": idx,
            "pattern_name": names,
            "pattern_probs": probs.tolist(),
            "pattern_masks": routing["pattern_masks"].detach().cpu().tolist(),
            "active_mlp_count": routing["active_mlp_count"].detach().cpu().tolist(),
            "expected_active_mlp": routing["expected_active_mlp"].detach().cpu().tolist(),
            "entropy": entropy.tolist(),
            "metadata": self.pattern_metadata,
        }


EarlyPatternViT = PatternMLPViT


def write_pattern_bank_metadata(pattern_bank: Union[str, Dict[str, Sequence[int]]], output_csv: str) -> None:
    names, patterns = load_pattern_bank(pattern_bank)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pattern_name", "active_mlp_count", "active_mlp_ratio", "active_layer_indices"])
        writer.writeheader()
        for row in pattern_metadata(names, patterns):
            writer.writerow(row)


from timm.models._registry import register_model


@register_model
def pattern_router_vit_small(pretrained: bool = False, **kwargs):
    model_args = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        class_token=False,
        global_pool="avg",
        pattern_mode="router",
        pattern_bank="bank4",
    )
    return PatternMLPViT(**dict(model_args, **kwargs))


@register_model
def router_vit_small_patch16_224_depth12_pattern_router(pretrained: bool = False, **kwargs):
    model_args = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        class_token=False,
        global_pool="avg",
        pattern_mode="router",
        pattern_bank="bank4",
    )
    return PatternMLPViT(**dict(model_args, **kwargs))


@register_model
def pattern_mlp_vit_small_patch16_224_depth12(pretrained: bool = False, **kwargs):
    model_args = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        class_token=False,
        global_pool="avg",
    )
    return PatternMLPViT(**dict(model_args, **kwargs))
