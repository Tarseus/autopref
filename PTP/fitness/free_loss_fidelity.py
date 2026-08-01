from __future__ import annotations
import gc
from contextlib import nullcontext
from dataclasses import dataclass, field
import os
import traceback
from typing import Any, Dict, List, Mapping, Protocol, Sequence, Tuple
import logging
import torch
from torch.optim import Adam
from .co_features import build_model_output, gather_pairwise_deltas
from .ptp_high_fidelity import HighFidelityConfig, _set_seed, resolve_pomo_size, get_hf_epoch_plan, get_total_hf_train_steps
from ptp_discovery.free_loss_compiler import CompiledFreeLoss
logger = logging.getLogger(__name__)
_PRECISION_OVERRIDE_WARNED: set[tuple[str, str, str, str]] = set()
def aggregate_pairwise_objective(pair_loss: torch.Tensor, *, instance_idx: torch.Tensor, weight: torch.Tensor, num_instances: int, eps: float=1e-08) -> torch.Tensor:
    if not isinstance(pair_loss, torch.Tensor):
        raise TypeError(f'pair loss must be a tensor, got {type(pair_loss)}')
    pair_loss = pair_loss.reshape(-1)
    instance_idx = instance_idx.reshape(-1).to(device=pair_loss.device, dtype=torch.long)
    weight = weight.reshape(-1).to(device=pair_loss.device, dtype=pair_loss.dtype)
    if pair_loss.numel() != instance_idx.numel() or pair_loss.numel() != weight.numel():
        raise ValueError('pair loss, instance_idx, and weight must have the same number of elements')
    if int(num_instances) <= 0:
        raise ValueError('num_instances must be positive')
    if not torch.isfinite(pair_loss).all() or not torch.isfinite(weight).all():
        raise ValueError('pair loss and weight must be finite')
    if (weight < 0).any():
        raise ValueError('pair weights must be non-negative')
    numerator = pair_loss.new_zeros(int(num_instances))
    denominator = pair_loss.new_zeros(int(num_instances))
    numerator.scatter_add_(0, instance_idx, pair_loss * weight)
    denominator.scatter_add_(0, instance_idx, weight)
    per_instance = numerator / denominator.clamp_min(float(eps))
    return per_instance.mean()

def _cfg_like_get(cfg_like: Mapping[str, Any] | Any, key: str, default: Any=None) -> Any:
    if isinstance(cfg_like, Mapping):
        return cfg_like.get(key, default)
    return getattr(cfg_like, key, default)

def _tensor_debug_summary(value: torch.Tensor) -> Dict[str, Any]:
    summary: Dict[str, Any] = {'type': 'tensor', 'shape': list(value.shape), 'dtype': str(value.dtype), 'device': str(value.device), 'numel': int(value.numel())}
    if value.numel() <= 0:
        return summary
    if value.is_cuda:
        summary['stats_skipped'] = 'cuda_tensor'
        return summary
    try:
        if value.is_floating_point() or value.is_complex():
            finite_mask = torch.isfinite(value)
            finite_all = bool(finite_mask.all().item())
            summary['isfinite'] = finite_all
            if finite_all:
                summary['min'] = float(value.amin().item())
                summary['max'] = float(value.amax().item())
            else:
                summary['finite_ratio'] = float(finite_mask.float().mean().item())
        elif value.dtype == torch.bool:
            summary['true_ratio'] = float(value.float().mean().item())
        else:
            summary['min'] = float(value.amin().item())
            summary['max'] = float(value.amax().item())
    except Exception as exc:
        summary['summary_error'] = f'{type(exc).__name__}: {exc}'
    return summary

def _mapping_debug_summary(value: Any, *, depth: int=0, max_depth: int=2, max_items: int=8) -> Dict[str, Any]:
    summary: Dict[str, Any] = {'type': type(value).__name__}
    batch_size = getattr(value, 'batch_size', None)
    if batch_size is not None:
        try:
            summary['batch_size'] = list(batch_size)
        except Exception:
            summary['batch_size'] = str(batch_size)
    try:
        keys = [str(k) for k in list(value.keys())]
    except Exception:
        return summary
    summary['keys'] = keys[:max_items]
    if len(keys) > max_items:
        summary['truncated_keys'] = int(len(keys) - max_items)
    if depth >= max_depth:
        return summary
    items: Dict[str, Any] = {}
    for key in keys[:max_items]:
        try:
            item = value.get(key)
        except Exception:
            try:
                item = value[key]
            except Exception as exc:
                items[str(key)] = {'summary_error': f'{type(exc).__name__}: {exc}'}
                continue
        items[str(key)] = _debug_value_summary(item, depth=depth + 1, max_depth=max_depth, max_items=max_items)
    summary['items'] = items
    return summary

def _object_debug_summary(value: Any, *, depth: int=0, max_depth: int=2, max_items: int=8) -> Dict[str, Any]:
    summary: Dict[str, Any] = {'type': type(value).__name__}
    field_names = [name for name in ('node_embeddings', 'graph_context', 'glimpse_key', 'glimpse_val', 'logit_key') if hasattr(value, name)]
    if not field_names:
        return summary
    summary['fields'] = field_names
    if depth >= max_depth:
        return summary
    items: Dict[str, Any] = {}
    for name in field_names[:max_items]:
        try:
            items[name] = _debug_value_summary(getattr(value, name), depth=depth + 1, max_depth=max_depth, max_items=max_items)
        except Exception as exc:
            items[name] = {'summary_error': f'{type(exc).__name__}: {exc}'}
    summary['items'] = items
    return summary

def _debug_value_summary(value: Any, *, depth: int=0, max_depth: int=2, max_items: int=8) -> Any:
    if isinstance(value, torch.Tensor):
        return _tensor_debug_summary(value)
    if isinstance(value, Mapping):
        return _mapping_debug_summary(value, depth=depth, max_depth=max_depth, max_items=max_items)
    if hasattr(value, 'keys') and callable(getattr(value, 'keys', None)):
        return _mapping_debug_summary(value, depth=depth, max_depth=max_depth, max_items=max_items)
    if isinstance(value, (list, tuple)):
        items = [_debug_value_summary(item, depth=depth + 1, max_depth=max_depth, max_items=max_items) for item in value[:max_items]]
        summary: Dict[str, Any] = {'type': type(value).__name__, 'len': len(value), 'items': items}
        if len(value) > max_items:
            summary['truncated_items'] = int(len(value) - max_items)
        return summary
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return _object_debug_summary(value, depth=depth, max_depth=max_depth, max_items=max_items)

def _should_aggressive_cuda_cleanup(cfg_like: Mapping[str, Any] | Any) -> bool:
    override = _cfg_like_get(cfg_like, 'aggressive_cuda_cleanup', None)
    if override is not None:
        return bool(override)
    env_name = str(_cfg_like_get(cfg_like, 'env_name', None) or _cfg_like_get(cfg_like, 'problem', 'tsp')).strip().lower()
    generator_params = _cfg_like_get(cfg_like, 'generator_params', {}) or {}
    try:
        ffsp_jobs = int((generator_params or {}).get('num_job', _cfg_like_get(cfg_like, 'train_problem_size', 0)) or 0)
    except Exception:
        ffsp_jobs = 0
    return env_name == 'ffsp' and ffsp_jobs >= 100

def _aggressive_cuda_cleanup_mode(cfg_like: Mapping[str, Any] | Any) -> str:
    if not _should_aggressive_cuda_cleanup(cfg_like):
        return 'off'
    mode = str(_cfg_like_get(cfg_like, 'aggressive_cuda_cleanup_mode', 'step') or 'step').strip().lower()
    if mode not in {'step', 'epoch', 'phase', 'off'}:
        return 'step'
    return mode

def _should_run_aggressive_cleanup(cfg_like: Mapping[str, Any] | Any, *, when: str) -> bool:
    mode = _aggressive_cuda_cleanup_mode(cfg_like)
    when_norm = str(when or '').strip().lower()
    if mode == 'off':
        return False
    if mode == 'step':
        return when_norm in {'step', 'epoch', 'phase'}
    if mode == 'epoch':
        return when_norm in {'epoch', 'phase'}
    if mode == 'phase':
        return when_norm == 'phase'
    return False

def _empty_cuda_cache_for_device(device: torch.device, *, collect_garbage: bool=False, synchronize: bool=False) -> None:
    if device.type != 'cuda' or not torch.cuda.is_available():
        return
    if collect_garbage:
        try:
            gc.collect()
        except Exception:
            pass
    try:
        with torch.cuda.device(device):
            if synchronize:
                torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
    except Exception:
        pass

def _maybe_aggressive_cuda_cleanup(device: torch.device, cfg_like: Mapping[str, Any] | Any, *, collect_garbage: bool=False) -> None:
    if device.type != 'cuda' or not _should_aggressive_cuda_cleanup(cfg_like):
        return
    _empty_cuda_cache_for_device(device, collect_garbage=collect_garbage, synchronize=True)

def _normalize_precision_mode(value: str | None) -> str:
    mode = str(value or '32-true').strip().lower()
    if mode in {'16', '16-mixed', 'fp16', 'fp16-mixed'}:
        return '16-mixed'
    if mode in {'bf16', 'bf16-mixed'}:
        return 'bf16-mixed'
    return '32-true'

def _effective_precision_mode(cfg_like: Mapping[str, Any] | Any) -> str:
    requested = _normalize_precision_mode(_cfg_like_get(cfg_like, 'precision', '32-true'))
    env_name = str(_cfg_like_get(cfg_like, 'env_name', _cfg_like_get(cfg_like, 'problem', '')) or '').strip().lower()
    policy_name = str(_cfg_like_get(cfg_like, 'policy_name', '') or '').strip().lower()
    if requested == '16-mixed' and env_name == 'ffsp' and (policy_name == 'matnet'):
        key = (env_name, policy_name, requested, '32-true')
        if key not in _PRECISION_OVERRIDE_WARNED:
            logger.warning('Overriding precision from %s to 32-true for env=%s policy=%s in RL4CO HF rollout to avoid FFSP MatNet CUDA invalid-argument failures.', requested, env_name, policy_name)
            _PRECISION_OVERRIDE_WARNED.add(key)
        return '32-true'
    return requested

def _autocast_context(device: torch.device, precision: str):
    mode = _normalize_precision_mode(precision)
    if device.type != 'cuda':
        return nullcontext()
    if mode == '16-mixed':
        return torch.autocast(device_type='cuda', dtype=torch.float16)
    if mode == 'bf16-mixed':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return nullcontext()

def _make_grad_scaler(device: torch.device, precision: str):
    enabled = device.type == 'cuda' and _normalize_precision_mode(precision) == '16-mixed'
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except Exception:
        try:
            return torch.cuda.amp.GradScaler(enabled=enabled)
        except Exception:
            return None

def _infer_pairwise_reference_tensor(batch: Mapping[str, Any]) -> torch.Tensor | None:
    for key in ('log_prob_w', 'log_prob_l', 'weight', 'cost_a', 'cost_b', 'cost_gap', 'delta_z', 'delta_rank', 'delta_regret'):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            return value
    return None

def prepare_pairwise_loss_batch(full_batch: Mapping[str, Any], expects: Sequence[str] | None=None) -> Dict[str, torch.Tensor]:
    batch_full: Dict[str, torch.Tensor] = {str(key): value for key, value in full_batch.items() if isinstance(value, torch.Tensor)}
    ref = _infer_pairwise_reference_tensor(batch_full)
    if 'cost_gap' not in batch_full and isinstance(batch_full.get('cost_a'), torch.Tensor) and isinstance(batch_full.get('cost_b'), torch.Tensor):
        batch_full['cost_gap'] = batch_full['cost_b'] - batch_full['cost_a']
    if 'advantage_w' not in batch_full and isinstance(batch_full.get('cost_a'), torch.Tensor) and ('advantage_gap' not in batch_full):
        batch_full['advantage_w'] = -batch_full['cost_a']
    if 'advantage_l' not in batch_full and isinstance(batch_full.get('cost_b'), torch.Tensor) and ('advantage_gap' not in batch_full):
        batch_full['advantage_l'] = -batch_full['cost_b']
    if 'advantage_gap' not in batch_full:
        if isinstance(batch_full.get('advantage_w'), torch.Tensor) and isinstance(batch_full.get('advantage_l'), torch.Tensor):
            batch_full['advantage_gap'] = batch_full['advantage_l'] - batch_full['advantage_w']
        elif isinstance(batch_full.get('cost_gap'), torch.Tensor):
            batch_full['advantage_gap'] = -batch_full['cost_gap']
        elif isinstance(batch_full.get('cost_a'), torch.Tensor) and isinstance(batch_full.get('cost_b'), torch.Tensor):
            batch_full['advantage_gap'] = batch_full['cost_a'] - batch_full['cost_b']
        elif isinstance(ref, torch.Tensor):
            batch_full['advantage_gap'] = torch.zeros_like(ref)
    if 'weight' not in batch_full and isinstance(ref, torch.Tensor):
        batch_full['weight'] = torch.ones_like(ref)
    requested = [str(key).strip() for key in expects or [] if str(key).strip()]
    if not requested:
        return dict(batch_full)
    out: Dict[str, torch.Tensor] = {key: batch_full[key] for key in requested if key in batch_full}
    if isinstance(ref, torch.Tensor):
        for key in requested:
            if key in out:
                continue
            if key in {'advantage_w', 'advantage_l', 'advantage_gap'}:
                out[key] = torch.zeros_like(ref)
            elif key == 'weight':
                out[key] = torch.ones_like(ref)
    return out

def evaluate_pairwise_loss(compiled: CompiledFreeLoss, *, full_batch: Mapping[str, Any], model_output: Mapping[str, torch.Tensor] | None=None, extra: Mapping[str, Any] | None=None, num_instances: int | None=None) -> torch.Tensor:
    expects = [str(key) for key in compiled.ir.implementation_hint.expects or []]
    forbidden = {'weight', 'instance_idx'}.intersection(expects)
    if forbidden:
        raise ValueError(f'pairwise loss must not read framework fields: {sorted(forbidden)}')
    batch = prepare_pairwise_loss_batch(full_batch, expects)
    ref = batch.get('log_prob_w')
    if not isinstance(ref, torch.Tensor):
        raise ValueError('pairwise loss requires log_prob_w')
    pair_loss = compiled.loss_fn(batch=batch, model_output=dict(model_output or {}), extra=dict(extra or {}))
    if not isinstance(pair_loss, torch.Tensor):
        raise TypeError(f'pairwise loss returned non-tensor: {type(pair_loss)}')
    if tuple(pair_loss.shape) != tuple(ref.shape):
        raise ValueError(f'pairwise loss must return one value per pair: expected {tuple(ref.shape)}, got {tuple(pair_loss.shape)}')
    instance_idx = full_batch.get('instance_idx')
    if not isinstance(instance_idx, torch.Tensor):
        instance_idx = torch.zeros_like(ref, dtype=torch.long)
    weight = full_batch.get('weight')
    if not isinstance(weight, torch.Tensor):
        weight = torch.ones_like(ref)
    if num_instances is None:
        num_instances = int(instance_idx.max().item()) + 1 if instance_idx.numel() else 1
    return aggregate_pairwise_objective(pair_loss, instance_idx=instance_idx, weight=weight, num_instances=int(num_instances))

@dataclass
class PrefBatch:
    mode: str
    pair_idx: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    weight: torch.Tensor | None = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def num_examples(self) -> int:
        return int(self.pair_idx[0].numel())

    def to_pairwise_loss_batch(self, feature_cache: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        b_idx, winner_idx, loser_idx = self.pair_idx
        objective = feature_cache['objective']
        log_prob = feature_cache['log_prob']
        cost_a_tensor = objective[b_idx, winner_idx]
        cost_b_tensor = objective[b_idx, loser_idx]
        logp_w_tensor = log_prob[b_idx, winner_idx]
        logp_l_tensor = log_prob[b_idx, loser_idx]
        features: Dict[str, torch.Tensor] = {}
        for key in ('obj_z', 'rank', 'regret'):
            value = feature_cache.get(key)
            if isinstance(value, torch.Tensor):
                features[key] = value
        pairwise_deltas = gather_pairwise_deltas(features, b_idx=b_idx, winner_idx=winner_idx, loser_idx=loser_idx)
        weight = self.weight
        if weight is None:
            weight = torch.ones_like(logp_w_tensor)
        batch = {'instance_idx': b_idx, 'cost_a': cost_a_tensor, 'cost_b': cost_b_tensor, 'cost_gap': cost_b_tensor - cost_a_tensor, 'log_prob_w': logp_w_tensor, 'log_prob_l': logp_l_tensor, **pairwise_deltas, 'weight': weight}
        return batch

class PrefBuilder(Protocol):

    def build(self, feature_cache: Mapping[str, torch.Tensor], *, meta: Mapping[str, Any] | None=None) -> PrefBatch:
        ...

def extract_feature_cache(objective: torch.Tensor, log_prob: torch.Tensor, *, extra: Mapping[str, torch.Tensor] | None=None) -> Dict[str, torch.Tensor]:
    model_output, _ = build_model_output(objective=objective, log_prob=log_prob)
    cache: Dict[str, torch.Tensor] = dict(model_output)
    if extra:
        for k, v in extra.items():
            if isinstance(v, torch.Tensor):
                cache[str(k)] = v
    advantage = cache.get('advantage')
    if not isinstance(advantage, torch.Tensor):
        advantage = objective.mean(dim=1, keepdim=True) - objective
    sequence_length = cache.get('sequence_length', cache.get('seq_len'))
    if not isinstance(sequence_length, torch.Tensor):
        sequence_length = torch.ones_like(log_prob)
    sequence_length = sequence_length.to(device=log_prob.device, dtype=log_prob.dtype).clamp_min(1.0)
    per_step_log_prob = cache.get('per_step_log_prob', cache.get('log_likelihood_step'))
    if not isinstance(per_step_log_prob, torch.Tensor):
        per_step_log_prob = log_prob.unsqueeze(-1)
    entropy = cache.get('entropy')
    if not isinstance(entropy, torch.Tensor):
        entropy = torch.zeros_like(log_prob)
    while entropy.ndim > log_prob.ndim:
        entropy = entropy.sum(dim=-1)
    cache['advantage'] = advantage
    cache['sequence_length'] = sequence_length
    cache['length_normalized_log_prob'] = log_prob / sequence_length
    cache['entropy'] = entropy
    cache['length_normalized_entropy'] = entropy / sequence_length
    cache['per_step_log_prob'] = per_step_log_prob
    return cache

class _AllPairsPrefBuilder:

    def build(self, feature_cache: Mapping[str, torch.Tensor], *, meta: Mapping[str, Any] | None=None) -> PrefBatch:
        objective = feature_cache['objective']
        (b_idx, winner_idx, loser_idx), _ = _build_preference_pairs(objective)
        out_meta: Dict[str, Any] = {'builder': 'all_pairs'}
        if meta:
            out_meta.update(dict(meta))
        return PrefBatch(mode='pairwise', pair_idx=(b_idx, winner_idx, loser_idx), weight=None, meta=out_meta)

class AverageMeter:

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int=1) -> None:
        value = float(val)
        self.val = value
        self.sum += value * n
        self.count += int(n)
        if self.count > 0:
            self.avg = self.sum / self.count

@dataclass
class FreeLossFidelityConfig:
    hf: HighFidelityConfig
    init_checkpoint_path: str | None = None
    scratch_hf_epochs: int = 0
    warmstart_hf_epochs: int = 0

def _extract_state_dict_from_checkpoint(payload: object) -> Mapping[str, torch.Tensor] | None:
    if isinstance(payload, dict):
        sd = payload.get('state_dict')
        if isinstance(sd, dict):
            return sd
        sd = payload.get('model_state_dict')
        if isinstance(sd, dict):
            return sd
    return None

def _load_policy_weights_from_checkpoint(policy, ckpt_path: str) -> None:
    if not ckpt_path:
        return
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'init_checkpoint_path does not exist: {ckpt_path}')
    ckpt: object
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    except Exception as exc:
        logger.warning('weights_only checkpoint load failed (%s); retrying weights_only=False for %s', type(exc).__name__, os.path.abspath(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = _extract_state_dict_from_checkpoint(ckpt)
    if state_dict is None:
        raise ValueError(f'Unsupported checkpoint format (missing state_dict): {ckpt_path}')
    target_sd = policy.state_dict()
    candidates: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    prefixes = ['policy.', 'model.policy.', 'model.', 'net.', 'module.', '']
    for prefix in prefixes:
        if prefix:
            sliced = {k[len(prefix):]: v for k, v in state_dict.items() if isinstance(k, str) and k.startswith(prefix)}
        else:
            sliced = {k: v for k, v in state_dict.items() if isinstance(k, str)}
        if not sliced:
            continue
        candidates.append((prefix, sliced))

    def _score(sd: Mapping[str, torch.Tensor]) -> int:
        score = 0
        for k, v in sd.items():
            if k not in target_sd:
                continue
            tv = target_sd[k]
            if isinstance(v, torch.Tensor) and isinstance(tv, torch.Tensor) and (tuple(v.shape) == tuple(tv.shape)):
                score += 1
        return score
    best_prefix = None
    best_sd: Dict[str, torch.Tensor] | None = None
    best_score = -1
    for prefix, cand_sd in candidates:
        s = _score(cand_sd)
        if s > best_score:
            best_score = s
            best_prefix = prefix
            best_sd = cand_sd
    if best_sd is None or best_score <= 0:
        raise ValueError(f'Could not match checkpoint weights to policy state_dict (ckpt={ckpt_path}). state_dict_keys={len(state_dict)} policy_keys={len(target_sd)}.')
    missing, unexpected = policy.load_state_dict(best_sd, strict=False)
    logger.info('Loaded init checkpoint into policy: path=%s prefix=%s matched=%d missing=%d unexpected=%d', os.path.abspath(ckpt_path), str(best_prefix), int(best_score), int(len(missing)), int(len(unexpected)))

def _build_preference_pairs(objective: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], int]:
    mask = objective[:, :, None] < objective[:, None, :]
    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)
    pair_count = int(b_idx.numel())
    return ((b_idx, winner_idx, loser_idx), pair_count)

def _rl4co_env_name(cfg: HighFidelityConfig) -> str:
    env_name = getattr(cfg, 'env_name', '') or getattr(cfg, 'problem', 'tsp')
    return str(env_name).strip().lower()

def _rl4co_size_key(env_name: str) -> str | None:
    return {'tsp': 'num_loc', 'cvrp': 'num_loc', 'jssp': 'num_jobs', 'fjsp': 'num_jobs', 'ffsp': 'num_job'}.get(env_name)

def _rl4co_policy_name(cfg: HighFidelityConfig, env_name: str) -> str:
    policy_name = str(getattr(cfg, 'policy_name', '') or '').strip().lower()
    if policy_name:
        return policy_name
    return {'tsp': 'pomo', 'cvrp': 'pomo', 'jssp': 'l2d', 'fjsp': 'l2d', 'ffsp': 'matnet'}.get(env_name, 'pomo')

def _rl4co_rollout_strategy(cfg: HighFidelityConfig, policy_name: str) -> str:
    strategy = str(getattr(cfg, 'rollout_strategy', 'auto') or 'auto').strip().lower()
    if strategy and strategy != 'auto':
        return strategy
    if policy_name in {'pomo', 'matnet'}:
        return 'policy_multistart'
    return 'batchify_sampling'

def _rl4co_set_multistart_decode(policy) -> None:
    for phase in ('train', 'val', 'test'):
        attr = f'{phase}_decode_type'
        val = getattr(policy, attr, None)
        if val is None:
            continue
        if 'multistart' in str(val):
            continue
        setattr(policy, attr, f'multistart_{val}')

def _rl4co_build_env(cfg: HighFidelityConfig, problem_size: int):
    from rl4co.envs import CVRPEnv, FJSPEnv, JSSPEnv, TSPEnv
    from rl4co.envs.scheduling.ffsp.env import FFSPEnv
    env_name = _rl4co_env_name(cfg)
    env_kwargs = dict(getattr(cfg, 'env_kwargs', {}) or {})
    generator_params = dict(getattr(cfg, 'generator_params', {}) or {})
    size_key = _rl4co_size_key(env_name)
    if size_key is not None:
        generator_params[size_key] = int(problem_size)
    env_map = {'tsp': TSPEnv, 'cvrp': CVRPEnv, 'jssp': JSSPEnv, 'fjsp': FJSPEnv, 'ffsp': FFSPEnv}
    if env_name not in env_map:
        raise ValueError(f'Unsupported env_name for RL4CO backend: {env_name}')
    env = env_map[env_name](generator_params=generator_params, **env_kwargs)
    return env

def _rl4co_build_policy(cfg: HighFidelityConfig, env):
    from rl4co.models.zoo.am import AttentionModelPolicy
    from rl4co.models.zoo.l2d.policy import L2DPolicy
    from rl4co.models.zoo.matnet.model import select_matnet_policy
    env_name = _rl4co_env_name(cfg)
    policy_name = _rl4co_policy_name(cfg, env_name)
    policy_kwargs = dict(getattr(cfg, 'policy_kwargs', {}) or {})
    if policy_name == 'pomo':
        policy_defaults = {'num_encoder_layers': 6, 'normalization': 'instance', 'use_graph_context': False}
        policy_defaults.update(policy_kwargs)
        policy = AttentionModelPolicy(env_name=env.name, **policy_defaults)
    elif policy_name == 'am':
        policy_defaults = {'num_encoder_layers': 6, 'normalization': 'instance', 'use_graph_context': False}
        policy_defaults.update(policy_kwargs)
        policy = AttentionModelPolicy(env_name=env.name, **policy_defaults)
    elif policy_name == 'l2d':
        policy_kwargs.setdefault('test_decode_type', 'greedy')
        policy = L2DPolicy(env_name=env.name, **policy_kwargs)
    elif policy_name == 'matnet':
        policy = select_matnet_policy(env=env, **policy_kwargs)
    else:
        raise ValueError(f'Unsupported policy_name for RL4CO backend: {policy_name}')
    rollout_strategy = _rl4co_rollout_strategy(cfg, policy_name)
    if rollout_strategy == 'policy_multistart':
        _rl4co_set_multistart_decode(policy)
    return (policy, rollout_strategy)

def _rl4co_objective_from_reward(reward: torch.Tensor) -> torch.Tensor:
    return -reward

def _rl4co_rollout_full(env, policy, batch_size: int, num_rollouts: int, *, phase: str, rollout_strategy: str, device: torch.device, precision: str='32-true', cfg_like: Mapping[str, Any] | Any | None=None, return_actions: bool=False, return_entropy: bool=False, return_step_logp: bool=False) -> Dict[str, torch.Tensor | None]:
    from rl4co.utils.ops import batchify, unbatchify
    gen = getattr(env, 'generator', None)
    if gen is None:
        raise RuntimeError('RL4CO env has no generator')
    if hasattr(gen, 'set_split') and callable(getattr(gen, 'set_split')):
        try:
            gen.set_split(str(phase))
        except Exception:
            pass
    batch = gen(batch_size)
    batch = batch.to(device)
    td = env.reset(batch)
    with _autocast_context(device, precision):
        try:
            if rollout_strategy == 'policy_multistart':
                out = policy(td, env, phase=phase, num_starts=num_rollouts, return_actions=return_actions, return_entropy=return_entropy, return_sum_log_likelihood=not return_step_logp)
                reward = unbatchify(out['reward'], num_rollouts)
            else:
                td_rep = batchify(td, num_rollouts) if num_rollouts > 1 else td
                out = policy(td_rep, env, phase=phase, return_actions=return_actions, return_entropy=return_entropy, return_sum_log_likelihood=not return_step_logp)
                reward = unbatchify(out['reward'], num_rollouts)
        except Exception:
            raise
    raw_log_likelihood = out['log_likelihood']
    log_likelihood_step = None
    if return_step_logp:
        log_likelihood_candidate = unbatchify(raw_log_likelihood, num_rollouts)
        if log_likelihood_candidate.ndim > reward.ndim:
            log_likelihood_step = log_likelihood_candidate
            log_likelihood = log_likelihood_step.sum(dim=-1)
        else:
            log_likelihood = log_likelihood_candidate
            log_likelihood_step = log_likelihood.unsqueeze(-1)
    else:
        log_likelihood = unbatchify(raw_log_likelihood, num_rollouts)
    actions = None
    if return_actions and isinstance(out.get('actions'), torch.Tensor):
        actions = unbatchify(out['actions'], num_rollouts)
    entropy = None
    if return_entropy and isinstance(out.get('entropy'), torch.Tensor):
        entropy = unbatchify(out['entropy'], num_rollouts)
    seq_len = None
    if isinstance(actions, torch.Tensor):
        seq_len = torch.full_like(log_likelihood, float(actions.shape[-1]))
    elif isinstance(log_likelihood_step, torch.Tensor):
        seq_len = torch.full_like(log_likelihood, float(log_likelihood_step.shape[-1]))
    return {'reward': reward, 'log_likelihood': log_likelihood, 'log_likelihood_step': log_likelihood_step, 'entropy': entropy, 'actions': actions, 'seq_len': seq_len}

def _rl4co_rollout(env, policy, batch_size: int, num_rollouts: int, *, phase: str, rollout_strategy: str, device: torch.device, precision: str='32-true', cfg_like: Mapping[str, Any] | Any | None=None) -> Tuple[torch.Tensor, torch.Tensor]:
    out = _rl4co_rollout_full(env, policy, batch_size, num_rollouts, phase=phase, rollout_strategy=rollout_strategy, device=device, precision=precision, cfg_like=cfg_like, return_actions=False, return_entropy=False, return_step_logp=False)
    return (out['reward'], out['log_likelihood'])

def run_rl4co_rollout_smoke_test(cfg: HighFidelityConfig, *, init_checkpoint_path: str | None=None, phase: str='train', device: str | torch.device | None=None, batch_size: int=1, num_rollouts: int=1) -> Dict[str, Any]:
    target_device = device if isinstance(device, torch.device) else torch.device(str(device or cfg.device))
    effective_precision = _effective_precision_mode(cfg)
    effective_batch_size = max(1, min(int(batch_size), int(getattr(cfg, 'train_batch_size', 1) or 1)))
    max_rollouts = resolve_pomo_size(getattr(cfg, 'pomo_size', None), int(cfg.train_problem_size))
    effective_num_rollouts = max(1, min(int(num_rollouts), int(max_rollouts)))
    env_name = str(getattr(cfg, 'env_name', '') or '').strip().lower()
    policy_name = str(getattr(cfg, 'policy_name', '') or '').strip().lower()
    policy_kwargs = dict(getattr(cfg, 'policy_kwargs', {}) or {})
    result: Dict[str, Any] = {'ok': False, 'phase': str(phase), 'device': str(target_device), 'precision': str(effective_precision), 'batch_size': int(effective_batch_size), 'num_rollouts': int(effective_num_rollouts), 'init_checkpoint_path': str(init_checkpoint_path) if init_checkpoint_path else None}
    env = None
    policy = None
    model = None
    try:
        _set_seed(int(cfg.seed))
        env = _rl4co_build_env(cfg, cfg.train_problem_size)
        env = env.to(target_device)
        policy, rollout_strategy = _rl4co_build_policy(cfg, env)
        if init_checkpoint_path:
            _load_policy_weights_from_checkpoint(policy, str(init_checkpoint_path))
        policy = policy.to(target_device)
        policy.eval()
        with torch.no_grad():
            rollout = _rl4co_rollout_full(env, policy, effective_batch_size, effective_num_rollouts, phase=str(phase), rollout_strategy=rollout_strategy, device=target_device, precision=effective_precision, cfg_like=cfg, return_actions=False, return_entropy=False, return_step_logp=False)
        reward = rollout['reward']
        log_likelihood = rollout['log_likelihood']
        result['ok'] = True
        result['rollout_strategy'] = str(rollout_strategy)
        result['reward'] = _debug_value_summary(reward, max_depth=1, max_items=6)
        result['log_likelihood'] = _debug_value_summary(log_likelihood, max_depth=1, max_items=6)
        return result
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        result['error_traceback'] = traceback.format_exc()
        try:
            result['rollout_strategy'] = str(_rl4co_rollout_strategy(cfg, str(getattr(cfg, 'policy_name', '') or '')))
        except Exception:
            result['rollout_strategy'] = None
        return result
    finally:
        try:
            env = None
            policy = None
            model = None
            if target_device.type == 'cuda':
                _empty_cuda_cache_for_device(target_device, collect_garbage=_should_run_aggressive_cleanup(cfg, when='phase'), synchronize=True)
        except Exception:
            pass

def _train_one_batch_with_free_loss_rl4co(env, policy, optimizer: Adam, compiled_loss: CompiledFreeLoss, hf_cfg: HighFidelityConfig, rollout_strategy: str, device: torch.device, scaler=None, *, pref_builder: PrefBuilder | None=None) -> Tuple[float, float, int]:
    batch_size = hf_cfg.train_batch_size
    num_rollouts = resolve_pomo_size(hf_cfg.pomo_size, hf_cfg.train_problem_size)
    aggressive_cleanup = _should_aggressive_cuda_cleanup(hf_cfg)
    policy.train()
    rollout = _rl4co_rollout_full(env, policy, batch_size, num_rollouts, phase='train', rollout_strategy=rollout_strategy, device=device, precision=_effective_precision_mode(hf_cfg), cfg_like=hf_cfg, return_actions=True, return_entropy=True, return_step_logp=True)
    reward = rollout['reward'].float()
    log_likelihood = rollout['log_likelihood'].float()
    objective = _rl4co_objective_from_reward(reward)
    log_prob = log_likelihood
    feature_cache = extract_feature_cache(objective, log_prob, extra={
        'advantage': reward - reward.mean(dim=1, keepdim=True),
        'sequence_length': rollout['seq_len'],
        'entropy': rollout['entropy'],
        'per_step_log_prob': rollout['log_likelihood_step'],
    })
    builder = pref_builder or _AllPairsPrefBuilder()
    pref = builder.build(feature_cache, meta={'stage': 'train', 'rollout_strategy': str(rollout_strategy), 'problem': str(hf_cfg.problem), 'problem_size': int(hf_cfg.train_problem_size)})
    pair_count = pref.num_examples()
    loss = evaluate_pairwise_loss(compiled_loss, full_batch=pref.to_pairwise_loss_batch(feature_cache), model_output=feature_cache, extra={'alpha': hf_cfg.alpha}, num_instances=int(objective.shape[0]))
    max_reward, _ = reward.max(dim=1)
    score_mean = _rl4co_objective_from_reward(max_reward).float().mean()
    optimizer.zero_grad(set_to_none=True)
    if not torch.isfinite(loss).all():
        raise RuntimeError('Non-finite loss encountered during mini-train')
    if scaler is not None and bool(getattr(scaler, 'is_enabled', lambda: False)()):
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    score_item = float(score_mean.item())
    loss_item = float(loss.item())
    del rollout, reward, log_likelihood, objective, log_prob, feature_cache, max_reward, score_mean, loss
    try:
        del pref
    except Exception:
        pass
    if _should_run_aggressive_cleanup(hf_cfg, when='step'):
        _maybe_aggressive_cuda_cleanup(device, hf_cfg)
    return (score_item, loss_item, pair_count)

@torch.no_grad()
def _evaluate_rl4co_model(*, policy, cfg: HighFidelityConfig, problem_size: int, device: torch.device, num_episodes: int, batch_size: int, rollout_strategy: str) -> float:
    env = _rl4co_build_env(cfg, problem_size)
    env = env.to(device)
    policy.eval()
    aggressive_cleanup = _should_aggressive_cuda_cleanup(cfg)
    num_rollouts = resolve_pomo_size(cfg.pomo_size, problem_size)
    score_meter = AverageMeter()
    episodes_done = 0
    while episodes_done < num_episodes:
        remaining = num_episodes - episodes_done
        current_batch = min(batch_size, remaining)
        reward, _ = _rl4co_rollout(env, policy, current_batch, num_rollouts, phase='test', rollout_strategy=rollout_strategy, device=device, precision=_effective_precision_mode(cfg), cfg_like=cfg)
        max_reward, _ = reward.max(dim=1)
        score = _rl4co_objective_from_reward(max_reward).float().mean().item()
        score_meter.update(score, n=current_batch)
        episodes_done += current_batch
        del max_reward, reward
        if _should_run_aggressive_cleanup(cfg, when='step'):
            _maybe_aggressive_cuda_cleanup(device, cfg)
    env = None
    if _should_run_aggressive_cleanup(cfg, when='phase'):
        _maybe_aggressive_cuda_cleanup(device, cfg, collect_garbage=aggressive_cleanup)
    return float(score_meter.avg)

def _evaluate_free_loss_candidate_rl4co(compiled_loss: CompiledFreeLoss, cfg: FreeLossFidelityConfig, *, pref_builder: PrefBuilder | None=None) -> Dict[str, Any]:
    _set_seed(cfg.hf.seed)
    device_name = cfg.hf.device
    if device_name == 'cuda' and (not torch.cuda.is_available()):
        device_name = 'cpu'
    device = torch.device(device_name)
    env = _rl4co_build_env(cfg.hf, cfg.hf.train_problem_size).to(device)
    policy, rollout_strategy = _rl4co_build_policy(cfg.hf, env)
    if cfg.init_checkpoint_path:
        _load_policy_weights_from_checkpoint(policy, cfg.init_checkpoint_path)
    policy = policy.to(device)
    scaler = _make_grad_scaler(device, _effective_precision_mode(cfg.hf))
    optimizer = Adam(policy.parameters(), lr=float(cfg.hf.learning_rate), weight_decay=float(cfg.hf.weight_decay))
    steps_per_epoch, configured_epochs = get_hf_epoch_plan(cfg.hf)
    if steps_per_epoch > 0 and configured_epochs > 0:
        if cfg.init_checkpoint_path and cfg.warmstart_hf_epochs > 0:
            epochs = int(cfg.warmstart_hf_epochs)
        elif not cfg.init_checkpoint_path and cfg.scratch_hf_epochs > 0:
            epochs = int(cfg.scratch_hf_epochs)
        else:
            epochs = int(configured_epochs)
        total_steps = int(steps_per_epoch) * int(epochs)
    else:
        epochs = 0
        total_steps = get_total_hf_train_steps(cfg.hf)
    score_meter = AverageMeter()
    loss_meter = AverageMeter()
    pair_count = 0
    epoch_objectives: List[float] = []
    log_interval = max(int(total_steps) // 10, 1)
    logger.info('RL4CO free-loss training: steps=%d epochs=%d train_problem_size=%d rollouts=%d batch_size=%d device=%s env=%s', int(total_steps), int(epochs), int(cfg.hf.train_problem_size), resolve_pomo_size(cfg.hf.pomo_size, cfg.hf.train_problem_size), int(cfg.hf.train_batch_size), str(device), _rl4co_env_name(cfg.hf))
    try:
        for step in range(int(total_steps)):
            score, loss, batch_pairs = _train_one_batch_with_free_loss_rl4co(env=env, policy=policy, optimizer=optimizer, compiled_loss=compiled_loss, hf_cfg=cfg.hf, rollout_strategy=rollout_strategy, device=device, scaler=scaler, pref_builder=pref_builder)
            score_meter.update(score)
            loss_meter.update(loss)
            pair_count += int(batch_pairs)
            if (step + 1) % log_interval == 0 or step == 0:
                logger.info('RL4CO free-loss step %d/%d: score=%.6f loss=%.6f pairs=%d', step + 1, int(total_steps), float(score), float(loss), int(batch_pairs))
            if epochs > 0 and (step + 1) % int(steps_per_epoch) == 0:
                value = _evaluate_rl4co_model(policy=policy, cfg=cfg.hf, problem_size=cfg.hf.train_problem_size, device=device, num_episodes=cfg.hf.num_validation_episodes, batch_size=cfg.hf.validation_batch_size, rollout_strategy=rollout_strategy)
                epoch_objectives.append(float(value))
                if torch.cuda.is_available():
                    _empty_cuda_cache_for_device(device, synchronize=True)
        main_objective = float(_evaluate_rl4co_model(policy=policy, cfg=cfg.hf, problem_size=cfg.hf.train_problem_size, device=device, num_episodes=cfg.hf.num_validation_episodes, batch_size=cfg.hf.validation_batch_size, rollout_strategy=rollout_strategy))
        size_objectives: Dict[int, float] = {int(cfg.hf.train_problem_size): main_objective}
        for size in cfg.hf.valid_problem_sizes:
            size_value = int(size)
            if size_value not in size_objectives:
                size_objectives[size_value] = float(_evaluate_rl4co_model(policy=policy, cfg=cfg.hf, problem_size=size_value, device=device, num_episodes=cfg.hf.num_validation_episodes, batch_size=cfg.hf.validation_batch_size, rollout_strategy=rollout_strategy))
        return {'validation_objective': main_objective, 'size_objectives': size_objectives, 'epoch_objectives': epoch_objectives, 'train_score_mean': float(score_meter.avg), 'train_loss_mean': float(loss_meter.avg), 'pair_count': int(pair_count), 'total_train_steps': int(total_steps)}
    finally:
        env = None
        policy = None
        if torch.cuda.is_available():
            _empty_cuda_cache_for_device(device, collect_garbage=True, synchronize=True)

def evaluate_free_loss_candidate(compiled_loss: CompiledFreeLoss, cfg: FreeLossFidelityConfig, *, pref_builder: PrefBuilder | None=None) -> Dict[str, Any]:
    return _evaluate_free_loss_candidate_rl4co(compiled_loss, cfg, pref_builder=pref_builder)
