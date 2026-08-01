from __future__ import annotations

from typing import Dict, Tuple

import torch


SOLUTION_FEATURE_KEYS: Tuple[str, ...] = (
    "objective",
    "log_prob",
    "obj_z",
    "rank",
    "best_cost_gap",
    "best_cost_gap_normalized",
    "regret",
    "advantage",
    "sequence_length",
    "length_normalized_log_prob",
    "entropy",
    "length_normalized_entropy",
    "per_step_log_prob",
)


INSTANCE_FEATURE_KEYS: Tuple[str, ...] = (
    "instance_num_rollouts",
    "instance_obj_mean",
    "instance_obj_std",
    "instance_obj_median",
    "instance_obj_mad",
    "instance_obj_min",
    "instance_obj_max",
    "instance_obj_range",
    "instance_log_prob_mean",
    "instance_log_prob_std",
    "instance_log_prob_min",
    "instance_log_prob_max",
    "instance_log_prob_range",
    "instance_best_cost_gap_mean",
    "instance_best_cost_gap_std",
    "instance_best_cost_gap_max",
    "instance_regret_mean",
    "instance_regret_std",
    "instance_regret_max",
)


def zscore(x: torch.Tensor, *, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True, unbiased=False)
    return (x - mean) / (std + eps)


def rank01(x: torch.Tensor, *, dim: int = 1) -> torch.Tensor:
    if x.numel() == 0:
        return x

    k = int(x.shape[dim])
    if k <= 1:
        return torch.zeros_like(x, dtype=torch.float32)

    order = x.argsort(dim=dim)
    ranks = order.argsort(dim=dim).to(dtype=torch.float32)
    return ranks / float(k - 1)


def robust_scale_mad(x: torch.Tensor, *, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    median = x.median(dim=dim, keepdim=True).values
    mad = (x - median).abs().median(dim=dim, keepdim=True).values
    scale = mad * 1.4826
    return scale.clamp_min(eps)


def _range(x: torch.Tensor, *, dim: int = 1) -> torch.Tensor:
    return x.amax(dim=dim) - x.amin(dim=dim)


def _instance_summary_stats(
    x: torch.Tensor,
    *,
    prefix: str,
    dim: int = 1,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    median = x.median(dim=dim).values
    mad = (x - median.unsqueeze(dim)).abs().median(dim=dim).values
    return {
        f"{prefix}_mean": x.mean(dim=dim),
        f"{prefix}_std": x.std(dim=dim, unbiased=False).clamp_min(eps),
        f"{prefix}_median": median,
        f"{prefix}_mad": mad.clamp_min(eps),
        f"{prefix}_min": x.amin(dim=dim),
        f"{prefix}_max": x.amax(dim=dim),
        f"{prefix}_range": _range(x, dim=dim).clamp_min(eps),
    }


def regret(
    objective: torch.Tensor,
    *,
    dim: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:

    best = objective.min(dim=dim, keepdim=True).values
    scale = robust_scale_mad(objective, dim=dim, eps=eps)
    return (objective - best) / scale


def compute_co_features(
    objective: torch.Tensor,
    *,
    dim: int = 1,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    best = objective.min(dim=dim, keepdim=True).values
    best_cost_gap = objective - best
    best_cost_gap_normalized = best_cost_gap / robust_scale_mad(objective, dim=dim, eps=eps)
    return {
        "obj_z": zscore(objective, dim=dim, eps=eps),
        "rank": rank01(objective, dim=dim),
        "best_cost_gap": best_cost_gap,
        "best_cost_gap_normalized": best_cost_gap_normalized,
        "regret": best_cost_gap_normalized,
    }


def compute_instance_features(
    objective: torch.Tensor,
    log_prob: torch.Tensor,
    *,
    co_features: Dict[str, torch.Tensor] | None = None,
    dim: int = 1,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    feats = co_features if isinstance(co_features, dict) else compute_co_features(objective, dim=dim, eps=eps)
    regret_feat = feats["regret"]
    best_cost_gap_feat = feats["best_cost_gap_normalized"]
    out: Dict[str, torch.Tensor] = {
        "instance_num_rollouts": torch.full(
            (int(objective.shape[0]),),
            float(objective.shape[dim]),
            dtype=objective.dtype,
            device=objective.device,
        )
    }
    out.update(_instance_summary_stats(objective, prefix="instance_obj", dim=dim, eps=eps))
    log_prob_stats = _instance_summary_stats(log_prob, prefix="instance_log_prob", dim=dim, eps=eps)
    for key in (
        "instance_log_prob_mean",
        "instance_log_prob_std",
        "instance_log_prob_min",
        "instance_log_prob_max",
        "instance_log_prob_range",
    ):
        out[key] = log_prob_stats[key]
    out["instance_regret_mean"] = regret_feat.mean(dim=dim)
    out["instance_regret_std"] = regret_feat.std(dim=dim, unbiased=False).clamp_min(eps)
    out["instance_regret_max"] = regret_feat.amax(dim=dim)
    out["instance_best_cost_gap_mean"] = best_cost_gap_feat.mean(dim=dim)
    out["instance_best_cost_gap_std"] = best_cost_gap_feat.std(dim=dim, unbiased=False).clamp_min(eps)
    out["instance_best_cost_gap_max"] = best_cost_gap_feat.amax(dim=dim)
    return out


def gather_pairwise_deltas(
    features: Dict[str, torch.Tensor],
    *,
    b_idx: torch.Tensor,
    winner_idx: torch.Tensor,
    loser_idx: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for base_key in ("obj_z", "rank", "regret"):
        if base_key not in features:
            continue
        feat = features[base_key]
        w = feat[b_idx, winner_idx]
        l = feat[b_idx, loser_idx]
        out[f"delta_{'z' if base_key == 'obj_z' else base_key}"] = l - w
    return out


def build_model_output(
    *,
    objective: torch.Tensor,
    log_prob: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

    feats = compute_co_features(objective, dim=1, eps=eps)
    instance_feats = compute_instance_features(objective, log_prob, co_features=feats, dim=1, eps=eps)
    model_output: Dict[str, torch.Tensor] = {
        "objective": objective,
        "log_prob": log_prob,
        **feats,
        **instance_feats,
    }
    return model_output, {**feats, **instance_feats}

