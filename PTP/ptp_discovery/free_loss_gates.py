from __future__ import annotations
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Set, Tuple
import torch
from .free_loss_compiler import CompiledFreeLoss
from .free_loss_ir import FreeLossIR
from fitness.co_features import build_model_output
from fitness.free_loss_fidelity import PrefBatch, evaluate_pairwise_loss, prepare_pairwise_loss_batch

_WEIGHT_CV_THRESHOLD = 0.1

@dataclass
class StaticGateResult:
    ok: bool
    reason: str = ''
    trace: Dict[str, Any] | None = None

@dataclass
class PreferenceBuilderGateResult:
    ok: bool
    reason: str = ''
    pair_count: int | None = None
    coverage: float | None = None
    max_pairs_per_instance: int | None = None
    weight_min: float | None = None
    weight_max: float | None = None
    weight_cv_min: float | None = None
    semantic_pass_rate: float | None = None
    trace: Dict[str, Any] | None = None

@dataclass
class JointPreferenceGateResult:
    ok: bool
    reason: str = ''
    grad_w_pass_rate: float | None = None
    grad_l_pass_rate: float | None = None
    swap_ok: bool | None = None
    effective_grad_ratio: float | None = None
    trace: Dict[str, Any] | None = None

def _tensor_schema(x: Any) -> Dict[str, Any]:
    if not isinstance(x, torch.Tensor):
        return {'type': type(x).__name__}
    out: Dict[str, Any] = {'shape': list(x.shape), 'dtype': str(x.dtype), 'device': str(x.device)}
    if x.numel() <= 0:
        return out
    try:
        if x.is_floating_point() or x.is_complex():
            out['isfinite'] = bool(torch.isfinite(x).all().item())
    except Exception:
        pass
    try:
        sample = x.detach().reshape(-1)
        if sample.numel() > 4096:
            sample = sample[:4096]
        if sample.dtype == torch.bool:
            sample_f = sample.to(dtype=torch.float32)
        elif sample.is_complex():
            sample_f = sample.abs().to(dtype=torch.float32)
        else:
            sample_f = sample.to(dtype=torch.float32)
        out['min'] = float(sample_f.min().item())
        out['max'] = float(sample_f.max().item())
        out['mean'] = float(sample_f.mean().item())
    except Exception:
        pass
    return out

def _batch_schema(batch: Mapping[str, Any] | None, *, max_keys: int=32) -> Dict[str, Any]:
    if not isinstance(batch, Mapping):
        return {}
    out: Dict[str, Any] = {}
    for key in sorted((str(k) for k in batch.keys()))[:max(int(max_keys), 0)]:
        try:
            out[key] = _tensor_schema(batch[key])
        except Exception as exc:
            out[key] = {'type': 'schema_error', 'message': str(exc)}
    return out

def _joint_gate_error_trace(*, compiled: CompiledFreeLoss, failure_kind: str, exc: Exception, variant: str, full_batch: Mapping[str, Any] | None, batch: Mapping[str, Any] | None, min_pass_rate: float, swap_tolerance: float, grad_eps: float, min_effective_grad_ratio: float, swap_check_mode: str, swap_test_margin: float) -> Dict[str, Any]:
    return {'failed_gate': 'JointPreference', 'failure_kind': str(failure_kind), 'message': str(exc), 'exception_type': type(exc).__name__, 'traceback': traceback.format_exc(), 'expects': list(compiled.ir.implementation_hint.expects or []), 'batch_keys_full': sorted(list(full_batch.keys())) if isinstance(full_batch, Mapping) else None, 'batch_keys_filtered': sorted(list(batch.keys())) if isinstance(batch, Mapping) else [], 'batch_schema': _batch_schema(batch, max_keys=32), 'gate_thresholds': {'min_pass_rate': float(min_pass_rate), 'swap_tolerance': float(swap_tolerance), 'grad_eps': float(grad_eps), 'min_effective_grad_ratio': float(min_effective_grad_ratio), 'swap_check_mode': str(swap_check_mode), 'swap_test_margin': float(swap_test_margin)}, 'variant': str(variant)}

def run_preference_builder_gates(pref_batch: PrefBatch, *, feature_cache: Mapping[str, torch.Tensor]) -> PreferenceBuilderGateResult:
    objective = feature_cache.get('objective')
    if not isinstance(objective, torch.Tensor) or objective.ndim != 2:
        return PreferenceBuilderGateResult(ok=False, reason='invalid_objective')
    if pref_batch.pair_idx is None:
        return PreferenceBuilderGateResult(ok=False, reason='missing_pair_idx')
    b_idx, winner_idx, loser_idx = pref_batch.pair_idx
    expected = (objective[:, :, None] < objective[:, None, :]).nonzero(as_tuple=True)
    if any((actual.shape != target.shape or not torch.equal(actual, target) for actual, target in zip((b_idx, winner_idx, loser_idx), expected))):
        return PreferenceBuilderGateResult(ok=False, reason='pair_topology_changed')
    pair_count = int(b_idx.numel())
    explicit_weight = pref_batch.weight is not None
    weight = pref_batch.weight
    if weight is None:
        weight = torch.ones(pair_count, dtype=torch.float32, device=b_idx.device)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 1 or int(weight.numel()) != pair_count:
        return PreferenceBuilderGateResult(ok=False, reason='invalid_weight', pair_count=pair_count)
    if not torch.isfinite(weight).all().item():
        return PreferenceBuilderGateResult(ok=False, reason='weight_not_finite', pair_count=pair_count)
    if (weight < 0.0).any().item():
        return PreferenceBuilderGateResult(ok=False, reason='weight_negative', pair_count=pair_count)
    totals = torch.zeros(int(objective.shape[0]), dtype=weight.dtype, device=weight.device)
    totals.scatter_add_(0, b_idx.to(device=weight.device), weight)
    if (totals <= 0.0).any().item():
        return PreferenceBuilderGateResult(ok=False, reason='instance_without_positive_weight', pair_count=pair_count, weight_min=float(weight.min().item()) if pair_count else None, weight_max=float(weight.max().item()) if pair_count else None)
    weight_cv_min = None
    if explicit_weight:
        cvs = []
        for instance_idx in range(int(objective.shape[0])):
            instance_weight = weight[b_idx.to(device=weight.device) == instance_idx]
            mean_abs = instance_weight.abs().mean().clamp_min(1e-8)
            cvs.append(instance_weight.std(unbiased=False) / mean_abs)
        weight_cv_min = float(torch.stack(cvs).min().item())
        if weight_cv_min <= _WEIGHT_CV_THRESHOLD:
            return PreferenceBuilderGateResult(ok=False, reason='instance_weight_cv_too_low', pair_count=pair_count, weight_min=float(weight.min().item()), weight_max=float(weight.max().item()), weight_cv_min=weight_cv_min, trace={'threshold': _WEIGHT_CV_THRESHOLD, 'observed_min': weight_cv_min})
    counts = torch.bincount(b_idx, minlength=int(objective.shape[0]))
    coverage = float((counts > 0).to(dtype=torch.float32).mean().item())
    max_pairs = int(counts.max().item())
    return PreferenceBuilderGateResult(ok=True, reason='ok', pair_count=pair_count, coverage=coverage, max_pairs_per_instance=max_pairs, weight_min=float(weight.min().item()), weight_max=float(weight.max().item()), weight_cv_min=weight_cv_min)

def run_joint_preference_gates(compiled: CompiledFreeLoss, *, pref_batch: PrefBatch, feature_cache: Mapping[str, torch.Tensor], min_pass_rate: float=0.8, swap_tolerance: float=0.001, swap_check_mode: str='data', swap_test_margin: float=1.0, grad_eps: float=1e-08, min_effective_grad_ratio: float=0.1, numeric_stress_enabled: bool=False, numeric_stress_margin: float=120.0, numeric_stress_aux_scale: float=32.0, variant: str='visible') -> JointPreferenceGateResult:
    variant = str(variant or 'visible').strip().lower()
    a = 2.0
    b = 1.0
    swap_check_mode = str(swap_check_mode or 'data').strip().lower()
    stress_enabled = bool(numeric_stress_enabled)
    try:
        stress_margin = abs(float(numeric_stress_margin))
    except (TypeError, ValueError):
        stress_margin = 120.0
    if stress_margin < 1e-06:
        stress_margin = 120.0
    try:
        stress_aux_scale = abs(float(numeric_stress_aux_scale))
    except (TypeError, ValueError):
        stress_aux_scale = 32.0
    if stress_aux_scale < 1.0:
        stress_aux_scale = 1.0
    mode = str(getattr(compiled.ir.implementation_hint, 'mode', 'pairwise') or 'pairwise').strip().lower()
    if mode != 'pairwise':
        return JointPreferenceGateResult(ok=True, reason='skipped_non_pairwise', trace={'failed_gate': None, 'variant': variant, 'mode': mode})
    full_batch: Dict[str, torch.Tensor] | None = None
    batch: Dict[str, torch.Tensor] | None = None
    try:
        full_batch = pref_batch.to_pairwise_loss_batch(feature_cache)
    except Exception as exc:
        return JointPreferenceGateResult(ok=False, reason=f'pref_batch_to_loss_batch_error: {exc}', trace=_joint_gate_error_trace(compiled=compiled, failure_kind='pref_batch_to_loss_batch_error', exc=exc, variant=variant, full_batch=full_batch, batch=batch, min_pass_rate=min_pass_rate, swap_tolerance=swap_tolerance, grad_eps=grad_eps, min_effective_grad_ratio=min_effective_grad_ratio, swap_check_mode=swap_check_mode, swap_test_margin=swap_test_margin))
    expects = [str(x) for x in compiled.ir.implementation_hint.expects or []]
    batch = prepare_pairwise_loss_batch(full_batch, expects)
    log_prob_w0 = batch.get('log_prob_w')
    log_prob_l0 = batch.get('log_prob_l')
    if not isinstance(log_prob_w0, torch.Tensor) or not isinstance(log_prob_l0, torch.Tensor):
        return JointPreferenceGateResult(ok=False, reason='missing_log_prob_tensors', trace={'failed_gate': 'JointPreference', 'failure_kind': 'missing_log_prob_tensors', 'variant': variant})
    log_prob_w = log_prob_w0.detach().clone().requires_grad_(True)
    log_prob_l = log_prob_l0.detach().clone().requires_grad_(True)
    batch = dict(batch)
    batch['log_prob_w'] = log_prob_w
    batch['log_prob_l'] = log_prob_l
    objective = feature_cache.get('objective')
    num_instances = int(objective.shape[0]) if isinstance(objective, torch.Tensor) else None

    def _joint_loss(current_batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        aggregate_batch = dict(full_batch or {})
        aggregate_batch.update(current_batch)
        return evaluate_pairwise_loss(compiled, full_batch=aggregate_batch, model_output={}, extra={'alpha': 1.0}, num_instances=num_instances)
    try:
        loss = _joint_loss(batch)
    except Exception as exc:
        return JointPreferenceGateResult(ok=False, reason=f'forward_error: {exc}', trace=_joint_gate_error_trace(compiled=compiled, failure_kind='forward_error', exc=exc, variant=variant, full_batch=full_batch, batch=batch, min_pass_rate=min_pass_rate, swap_tolerance=swap_tolerance, grad_eps=grad_eps, min_effective_grad_ratio=min_effective_grad_ratio, swap_check_mode=swap_check_mode, swap_test_margin=swap_test_margin))
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        return JointPreferenceGateResult(ok=False, reason='loss_not_scalar_tensor', trace={'failed_gate': 'JointPreference', 'failure_kind': 'loss_not_scalar_tensor', 'observed_type': str(type(loss)), 'observed_shape': None if not isinstance(loss, torch.Tensor) else tuple(loss.shape), 'variant': variant})
    if not torch.isfinite(loss).all().item():
        return JointPreferenceGateResult(ok=False, reason='loss_not_finite', trace={'failed_gate': 'JointPreference', 'failure_kind': 'loss_not_finite', 'variant': variant})
    try:
        loss.backward()
    except Exception as exc:
        return JointPreferenceGateResult(ok=False, reason=f'backward_error: {exc}', trace=_joint_gate_error_trace(compiled=compiled, failure_kind='backward_error', exc=exc, variant=variant, full_batch=full_batch, batch=batch, min_pass_rate=min_pass_rate, swap_tolerance=swap_tolerance, grad_eps=grad_eps, min_effective_grad_ratio=min_effective_grad_ratio, swap_check_mode=swap_check_mode, swap_test_margin=swap_test_margin))
    grad_w = log_prob_w.grad
    grad_l = log_prob_l.grad
    if grad_w is None or grad_l is None:
        return JointPreferenceGateResult(ok=False, reason='missing_grads', trace={'failed_gate': 'JointPreference', 'failure_kind': 'missing_grads', 'variant': variant})
    if not torch.isfinite(grad_w).all().item() or not torch.isfinite(grad_l).all().item():
        return JointPreferenceGateResult(ok=False, reason='grad_not_finite', trace={'failed_gate': 'JointPreference', 'failure_kind': 'grad_not_finite', 'variant': variant})
    if stress_enabled:
        stress_batch = dict(batch)
        lpw_ref = batch['log_prob_w'].detach()
        lpl_ref = batch['log_prob_l'].detach()
        mid = 0.5 * (lpw_ref + lpl_ref)
        half_margin = float(stress_margin) * 0.5
        stress_lpw = (mid - half_margin).detach().clone().requires_grad_(True)
        stress_lpl = (mid + half_margin).detach().clone().requires_grad_(True)
        stress_batch['log_prob_w'] = stress_lpw
        stress_batch['log_prob_l'] = stress_lpl
        for key in ('weight', 'delta_regret', 'delta_z', 'cost_gap', 'advantage_gap', 'advantage_w', 'advantage_l'):
            value = stress_batch.get(key)
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                stress_batch[key] = value.detach() * float(stress_aux_scale)
        try:
            stress_loss = _joint_loss(stress_batch)
        except Exception as exc:
            tr = _joint_gate_error_trace(compiled=compiled, failure_kind='numeric_stress_forward_error', exc=exc, variant=variant, full_batch=full_batch, batch=stress_batch, min_pass_rate=min_pass_rate, swap_tolerance=swap_tolerance, grad_eps=grad_eps, min_effective_grad_ratio=min_effective_grad_ratio, swap_check_mode=swap_check_mode, swap_test_margin=swap_test_margin)
            tr['numeric_stress'] = {'enabled': True, 'margin': float(stress_margin), 'aux_scale': float(stress_aux_scale)}
            return JointPreferenceGateResult(ok=False, reason=f'numeric_stress_forward_error: {exc}', trace=tr)
        if not isinstance(stress_loss, torch.Tensor) or stress_loss.numel() != 1:
            return JointPreferenceGateResult(ok=False, reason='numeric_stress_loss_not_scalar_tensor', trace={'failed_gate': 'JointPreference', 'failure_kind': 'numeric_stress_loss_not_scalar_tensor', 'observed_type': str(type(stress_loss)), 'observed_shape': None if not isinstance(stress_loss, torch.Tensor) else tuple(stress_loss.shape), 'variant': variant, 'numeric_stress': {'enabled': True, 'margin': float(stress_margin), 'aux_scale': float(stress_aux_scale)}})
        if not torch.isfinite(stress_loss).all().item():
            return JointPreferenceGateResult(ok=False, reason='numeric_stress_loss_not_finite', trace={'failed_gate': 'JointPreference', 'failure_kind': 'numeric_stress_loss_not_finite', 'variant': variant, 'numeric_stress': {'enabled': True, 'margin': float(stress_margin), 'aux_scale': float(stress_aux_scale), 'loss': float(stress_loss.detach().item())}})
        try:
            stress_grad_w, stress_grad_l = torch.autograd.grad(stress_loss, [stress_lpw, stress_lpl], allow_unused=True)
        except Exception as exc:
            tr = _joint_gate_error_trace(compiled=compiled, failure_kind='numeric_stress_backward_error', exc=exc, variant=variant, full_batch=full_batch, batch=stress_batch, min_pass_rate=min_pass_rate, swap_tolerance=swap_tolerance, grad_eps=grad_eps, min_effective_grad_ratio=min_effective_grad_ratio, swap_check_mode=swap_check_mode, swap_test_margin=swap_test_margin)
            tr['numeric_stress'] = {'enabled': True, 'margin': float(stress_margin), 'aux_scale': float(stress_aux_scale)}
            return JointPreferenceGateResult(ok=False, reason=f'numeric_stress_backward_error: {exc}', trace=tr)
        if stress_grad_w is None or stress_grad_l is None:
            return JointPreferenceGateResult(ok=False, reason='numeric_stress_missing_grads', trace={'failed_gate': 'JointPreference', 'failure_kind': 'numeric_stress_missing_grads', 'variant': variant, 'numeric_stress': {'enabled': True, 'margin': float(stress_margin), 'aux_scale': float(stress_aux_scale)}})
        if not torch.isfinite(stress_grad_w).all().item() or not torch.isfinite(stress_grad_l).all().item():
            return JointPreferenceGateResult(ok=False, reason='numeric_stress_grad_not_finite', trace={'failed_gate': 'JointPreference', 'failure_kind': 'numeric_stress_grad_not_finite', 'variant': variant, 'numeric_stress': {'enabled': True, 'margin': float(stress_margin), 'aux_scale': float(stress_aux_scale)}})
    weight = (full_batch or {}).get('weight')
    if not isinstance(weight, torch.Tensor):
        active = torch.ones_like(grad_w, dtype=torch.bool)
    else:
        active = weight.reshape(-1).to(device=grad_w.device) > 0.0
    if int(active.numel()) != int(grad_w.numel()):
        return JointPreferenceGateResult(ok=False, reason='active_weight_shape_mismatch', trace={'failed_gate': 'JointPreference', 'failure_kind': 'active_weight_shape_mismatch', 'weight_count': int(active.numel()), 'gradient_count': int(grad_w.numel()), 'variant': variant})
    active_pair_count = int(active.sum().item())
    if active_pair_count <= 0:
        return JointPreferenceGateResult(ok=False, reason='no_positive_weight_pairs', trace={'failed_gate': 'JointPreference', 'failure_kind': 'no_positive_weight_pairs', 'variant': variant})
    w_pass = float((grad_w[active] < 0.0).to(dtype=torch.float32).mean().item())
    l_pass = float((grad_l[active] > 0.0).to(dtype=torch.float32).mean().item())
    effective = (grad_w[active].abs() > float(grad_eps)) | (grad_l[active].abs() > float(grad_eps))
    effective_ratio = float(effective.to(dtype=torch.float32).mean().item())
    active_pair_ratio = float(active_pair_count) / float(max(int(active.numel()), 1))

    def _swap_signals(in_batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(in_batch)
        out['log_prob_w'], out['log_prob_l'] = (out['log_prob_l'].detach(), out['log_prob_w'].detach())
        if 'cost_a' in out and 'cost_b' in out:
            out['cost_a'], out['cost_b'] = (out['cost_b'], out['cost_a'])
        for key in list(out.keys()):
            if key.startswith('delta_') and isinstance(out[key], torch.Tensor):
                out[key] = -out[key]
        return out
    swap_ref_loss: float | None = None
    swap_ok: bool | None = None
    loss_swap_val: float | None = None
    try:
        if swap_check_mode == 'none':
            swap_ok = None
        elif swap_check_mode == 'synthetic':
            lpw_ref = batch['log_prob_w'].detach()
            lpl_ref = batch['log_prob_l'].detach()
            mid = 0.5 * (lpw_ref + lpl_ref)
            mag = (lpw_ref - lpl_ref).abs() + float(swap_test_margin)
            swap_test_batch = dict(batch)
            swap_test_batch['log_prob_w'] = (mid + 0.5 * mag).detach()
            swap_test_batch['log_prob_l'] = (mid - 0.5 * mag).detach()
            loss_test = _joint_loss(swap_test_batch)
            if isinstance(loss_test, torch.Tensor) and loss_test.numel() == 1 and torch.isfinite(loss_test).all().item():
                swap_ref_loss = float(loss_test.item())
                swap_batch = _swap_signals(swap_test_batch)
                loss_swap = _joint_loss(swap_batch)
                if isinstance(loss_swap, torch.Tensor) and loss_swap.numel() == 1 and torch.isfinite(loss_swap).all().item():
                    loss_swap_val = float(loss_swap.item())
                    swap_ok = loss_swap_val >= float(swap_ref_loss) + float(swap_tolerance)
        else:
            swap_ref_loss = float(loss.item())
            swap_batch = _swap_signals(batch)
            loss_swap = _joint_loss(swap_batch)
            if isinstance(loss_swap, torch.Tensor) and loss_swap.numel() == 1 and torch.isfinite(loss_swap).all().item():
                loss_swap_val = float(loss_swap.item())
                swap_ok = loss_swap_val >= float(swap_ref_loss) + float(swap_tolerance)
    except Exception:
        swap_ok = False
    ok = w_pass >= float(min_pass_rate) and l_pass >= float(min_pass_rate) and (effective_ratio >= float(min_effective_grad_ratio)) and (True if swap_ok is None else bool(swap_ok))
    where_failed: list[str] = []
    if w_pass < float(min_pass_rate):
        where_failed.append('log_prob_w_direction')
    if l_pass < float(min_pass_rate):
        where_failed.append('log_prob_l_direction')
    if effective_ratio < float(min_effective_grad_ratio):
        where_failed.append('saturation')
    if swap_ok is False:
        where_failed.append('swap')
    return JointPreferenceGateResult(ok=ok, reason='ok' if ok else 'joint_preference_violation', grad_w_pass_rate=w_pass, grad_l_pass_rate=l_pass, swap_ok=swap_ok, effective_grad_ratio=effective_ratio, trace={'failed_gate': None if ok else 'JointPreference', 'failure_kind': None if ok else 'joint_preference_violation', 'observed': {'grad_w_pass_rate': w_pass, 'grad_l_pass_rate': l_pass, 'effective_grad_ratio': effective_ratio, 'loss': float(loss.item()), 'loss_swap': loss_swap_val, 'swap_ok': swap_ok, 'swap_check_mode': swap_check_mode, 'swap_ref_loss': swap_ref_loss, 'active_pair_count': active_pair_count, 'active_pair_ratio': active_pair_ratio}, 'threshold': {'min_pass_rate': float(min_pass_rate), 'swap_tolerance': float(swap_tolerance), 'swap_test_margin': float(swap_test_margin), 'min_effective_grad_ratio': float(min_effective_grad_ratio), 'grad_eps': float(grad_eps)}, 'where_failed': where_failed, 'variant': variant})
_PAIRWISE_SUPPORTED_KEYS: Set[str] = {'log_prob_w', 'log_prob_l', 'cost_a', 'cost_b', 'cost_gap', 'delta_z', 'delta_rank', 'delta_regret', 'advantage_w', 'advantage_l', 'advantage_gap'}

_PAIRWISE_REQUIRED_LOGPROB: Set[str] = {'log_prob_w', 'log_prob_l'}
_PAIRWISE_REQUIRED_OBJECTIVE_SIGNAL: Set[str] = {'delta_z', 'delta_rank', 'delta_regret', 'cost_a', 'cost_b', 'cost_gap', 'advantage_w', 'advantage_l', 'advantage_gap'}

def supported_keys_for_mode(mode: str='pairwise') -> Sequence[str]:
    return sorted(_PAIRWISE_SUPPORTED_KEYS)

def run_static_gates(ir: FreeLossIR, *, operator_whitelist: Sequence[str]) -> StaticGateResult:
    if not ir.name or not ir.intuition.strip() or (not ir.pseudocode) or (not ir.operators_used):
        return StaticGateResult(ok=False, reason='incomplete_loss_ir')
    returns = ir.implementation_hint.returns.strip().lower()
    if returns not in {'per_pair', 'pairwise', 'one_per_pair'}:
        return StaticGateResult(ok=False, reason="implementation_hint.returns must be 'per_pair'.")
    expects = {str(value) for value in ir.implementation_hint.expects or []}
    if not expects:
        return StaticGateResult(ok=False, reason='implementation_hint.expects must be non-empty.')
    missing = _PAIRWISE_REQUIRED_LOGPROB - expects
    if missing:
        return StaticGateResult(ok=False, reason=f'implementation_hint.expects missing required keys: {sorted(missing)}')
    if not expects.intersection(_PAIRWISE_REQUIRED_OBJECTIVE_SIGNAL):
        return StaticGateResult(ok=False, reason='implementation_hint.expects requires an objective signal.')
    extra = expects - _PAIRWISE_SUPPORTED_KEYS
    if extra:
        return StaticGateResult(ok=False, reason=f'implementation_hint.expects contains unsupported keys: {sorted(extra)}')
    if not ir.code.strip():
        return StaticGateResult(ok=False, reason='loss code is required.')
    for key, value in ir.hyperparams.items():
        if isinstance(value, (int, float)) and (not torch.isfinite(torch.tensor(float(value)))):
            return StaticGateResult(ok=False, reason=f'hyperparameter {key} is non-finite.')
    return StaticGateResult(ok=True)

@dataclass
class AffineInvarianceGateResult:
    ok: bool
    reason: str = ''
    abs_delta: float | None = None
    rel_delta: float | None = None
    trace: Dict[str, Any] | None = None

def _loss_value(compiled: CompiledFreeLoss, *, batch: Mapping[str, Any], model_output: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return evaluate_pairwise_loss(compiled, full_batch=batch, model_output=model_output, extra={})

def _delta_metrics(a: torch.Tensor, b: torch.Tensor, eps: float=1e-08) -> Tuple[float, float]:
    abs_delta = float((a - b).abs().item())
    denom = float((a.abs() + b.abs() + eps).item())
    rel_delta = abs_delta / denom
    return (abs_delta, rel_delta)

def run_affine_invariance_gate(compiled: CompiledFreeLoss, *, max_abs_delta: float=0.05, pairwise_batch_size: int=64, variant: str='visible') -> AffineInvarianceGateResult:
    generator = torch.Generator().manual_seed(5082 if str(variant).lower() == 'visible' else 6017)
    scale = 2.0
    shift = 1.0
    try:
        log_prob_l = torch.rand(pairwise_batch_size, generator=generator) * -8.0
        log_prob_w = log_prob_l + torch.empty(pairwise_batch_size).uniform_(-3.0, 3.0, generator=generator)
        cost_a = torch.rand(pairwise_batch_size, generator=generator)
        gap = torch.rand(pairwise_batch_size, generator=generator)
        cost_b = cost_a + gap
        batch_raw: Dict[str, Any] = {'log_prob_w': log_prob_w, 'log_prob_l': log_prob_l, 'cost_a': cost_a, 'cost_b': cost_b, 'delta_z': gap * 2.0, 'delta_rank': (gap > 0.5).float(), 'delta_regret': gap / (gap.median() + 1e-06), 'weight': torch.ones(pairwise_batch_size)}
        batch_affine = dict(batch_raw)
        batch_affine['cost_a'] = scale * cost_a + shift
        batch_affine['cost_b'] = scale * cost_b + shift
        loss_raw = _loss_value(compiled, batch=batch_raw, model_output={})
        loss_affine = _loss_value(compiled, batch=batch_affine, model_output={})
        if loss_raw.numel() != 1 or loss_affine.numel() != 1:
            return AffineInvarianceGateResult(ok=False, reason='loss_not_scalar')
        if not torch.isfinite(loss_raw).all() or not torch.isfinite(loss_affine).all():
            return AffineInvarianceGateResult(ok=False, reason='non_finite_loss')
        abs_delta, rel_delta = _delta_metrics(loss_raw, loss_affine)
        ok = abs_delta <= float(max_abs_delta)
        return AffineInvarianceGateResult(ok=ok, reason='ok' if ok else 'affine_invariance_violation', abs_delta=abs_delta, rel_delta=rel_delta, trace={'failed_gate': None if ok else 'AffineInvariance', 'observed': {'abs_delta': abs_delta, 'rel_delta': rel_delta}, 'threshold': {'max_abs_delta': float(max_abs_delta)}, 'variant': str(variant)})
    except Exception as exc:
        return AffineInvarianceGateResult(ok=False, reason=f'error: {exc}', trace={'failed_gate': 'AffineInvariance', 'failure_kind': 'error', 'message': str(exc)})
