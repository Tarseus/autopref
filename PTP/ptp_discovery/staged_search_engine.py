from __future__ import annotations
import ast
import collections
import gc
import io
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
import time
import traceback
import tokenize
from dataclasses import asdict
from hashlib import sha1
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import torch
import yaml
from fitness.free_loss_fidelity import FreeLossFidelityConfig, PrefBatch, _empty_cuda_cache_for_device, extract_feature_cache, evaluate_free_loss_candidate, run_rl4co_rollout_smoke_test
from fitness.ptp_high_fidelity import HighFidelityConfig, _set_seed
from ptp_discovery.free_loss_compiler import CompiledFreeLoss, CompileError, compile_free_loss
from ptp_discovery.free_loss_gates import StaticGateResult, run_affine_invariance_gate, run_joint_preference_gates, run_preference_builder_gates, run_static_gates
from ptp_discovery.free_loss_ir import FreeLossIR, FreeLossImplementationHint, ir_from_json as free_loss_ir_from_json
from ptp_discovery.pref_builder_compiler import CompiledPreferenceBuilder, PreferenceBuilderCompileError, compile_preference_builder
from ptp_discovery.pref_builder_ir import PreferenceBuilderIR, PreferenceBuilderImplementationHint, ir_from_json as pref_builder_ir_from_json
import ptp_discovery.free_loss_llm_ops as loss_llm_ops
import ptp_discovery.pref_builder_llm_ops as builder_llm_ops
LOGGER = logging.getLogger('ptp_discovery.two_stage_engine')

class _CompiledBuilderAdapter:

    def __init__(self, compiled: CompiledPreferenceBuilder) -> None:
        self._compiled = compiled

    def build(self, feature_cache: Mapping[str, torch.Tensor], *, meta: Mapping[str, Any] | None=None) -> PrefBatch:
        return self._compiled.build_fn(feature_cache, dict(meta or {}))

def _normalize_device_alias(device_str: str) -> str:
    ds = str(device_str or '').strip()
    if ds.lower() == 'gpu':
        return 'cuda'
    return ds

def _repo_root_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def _high_fidelity_cleanup_eval_device(device_str: str | None) -> None:
    try:
        gc.collect()
    except Exception:
        pass
    if not torch.cuda.is_available():
        return
    try:
        dev = torch.device(str(device_str or 'cuda'))
    except Exception:
        return
    _empty_cuda_cache_for_device(dev, collect_garbage=True, synchronize=True)

def _abs_from_repo_root(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_repo_root_dir(), path))
_FILE_SHA1_CACHE: Dict[str, str] = {}
_BASELINE_MINI_EVAL_CACHE: Dict[str, Dict[str, Any]] = {}

def _file_sha1_cached(path: str, *, chunk_size: int=8 * 1024 * 1024) -> str:
    p = _abs_from_repo_root(str(path))
    cached = _FILE_SHA1_CACHE.get(p)
    if cached is not None:
        return str(cached)
    h = sha1()
    with open(p, 'rb') as f:
        while True:
            b = f.read(int(chunk_size))
            if not b:
                break
            h.update(b)
    out = h.hexdigest()
    _FILE_SHA1_CACHE[p] = out
    return str(out)

def _load_baseline_mini_eval(path: str) -> Dict[str, Any]:
    p = _abs_from_repo_root(str(path))
    cached = _BASELINE_MINI_EVAL_CACHE.get(p)
    if isinstance(cached, dict):
        return cached
    with open(p, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f'Invalid baseline mini-eval JSON (expected dict): {path}')
    _BASELINE_MINI_EVAL_CACHE[p] = dict(payload)
    return dict(payload)

def _resolve_training_seed(cfg_yaml: Mapping[str, Any], *, default: int=1234) -> int:
    raw = cfg_yaml.get('scratch_init_seed', None)
    if raw is not None:
        try:
            return int(raw)
        except Exception:
            pass
    raw = cfg_yaml.get('seed', default)
    try:
        return int(raw)
    except Exception:
        return int(default)

def _alpha_from_cfg(cfg_yaml: Mapping[str, Any], *, key: str='alpha') -> float:
    env_name = str(cfg_yaml.get('env_name') or cfg_yaml.get('problem') or 'tsp').strip().lower()
    default_alpha = 0.03 if env_name == 'cvrp' else 0.05
    return float(cfg_yaml.get(key, default_alpha) or default_alpha)

def _high_fidelity_slug(value: Any) -> str:
    raw = str(value or '').strip().lower()
    slug = re.sub('[^a-z0-9]+', '_', raw).strip('_')
    return slug or 'item'

def _high_fidelity_init_specs_from_baseline_cfg(cfg_yaml: Mapping[str, Any]) -> List[Tuple[str, str | None]]:
    baseline_cfg = cfg_yaml.get('baseline', {}) or {}
    if not isinstance(baseline_cfg, Mapping):
        baseline_cfg = {}
    include_scratch = bool(baseline_cfg.get('include_scratch', True))
    ckpts = baseline_cfg.get('checkpoints') or []
    if not isinstance(ckpts, Sequence) or isinstance(ckpts, (str, bytes)):
        ckpts = []
    init_specs: List[Tuple[str, str | None]] = []
    if include_scratch:
        init_specs.append(('scratch', None))
    used_names = {str(name) for name, _ in init_specs}
    for idx, ckpt in enumerate(ckpts):
        if ckpt is None or not str(ckpt).strip():
            continue
        ckpt_s = str(ckpt)
        epoch = _infer_baseline_epoch_from_path(ckpt_s)
        if epoch is not None:
            init_name = f'ckpt_{int(epoch)}'
        else:
            stem = os.path.splitext(os.path.basename(ckpt_s))[0]
            init_name = f'ckpt_{_high_fidelity_slug(stem)}'
        if init_name in used_names:
            stem = os.path.splitext(os.path.basename(ckpt_s))[0]
            init_name = f'ckpt_{int(idx):03d}_{_high_fidelity_slug(stem)}'
        suffix = 2
        base_name = str(init_name)
        while init_name in used_names:
            init_name = f'{base_name}_{int(suffix)}'
            suffix += 1
        used_names.add(str(init_name))
        init_specs.append((str(init_name), ckpt_s))
    return init_specs

def _build_high_fidelity_eval_signature(cfg_yaml: Mapping[str, Any]) -> Dict[str, Any]:
    init_specs = _high_fidelity_init_specs_from_baseline_cfg(cfg_yaml)
    if not init_specs:
        raise ValueError('high_fidelity requires at least one initialization source')
    generator_params = dict(cfg_yaml.get('generator_params', {}) or {})
    return {'protocol': 'high_fidelity_minitrain_v1', 'env_name': str(cfg_yaml.get('env_name') or cfg_yaml.get('problem') or 'tsp'), 'policy_name': str(cfg_yaml.get('policy_name') or ''), 'policy_kwargs': dict(cfg_yaml.get('policy_kwargs', {}) or {}), 'generator_params': generator_params, 'train_problem_size': int(cfg_yaml.get('train_problem_size', 20) or 20), 'valid_problem_sizes': [int(v) for v in cfg_yaml.get('valid_problem_sizes', [20])], 'train_batch_size': int(cfg_yaml.get('train_batch_size', 64) or 64), 'validation_batch_size': int(cfg_yaml.get('validation_batch_size', 64) or 64), 'num_validation_episodes': int(cfg_yaml.get('num_validation_episodes', 128) or 128), 'f1_steps': int(cfg_yaml.get('f1_steps', 32) or 32), 'hf_epochs': int(cfg_yaml.get('hf_epochs', 0) or 0), 'hf_instances_per_epoch': int(cfg_yaml.get('hf_instances_per_epoch', 0) or 0), 'learning_rate': float(cfg_yaml.get('learning_rate', 0.0003) or 0.0003), 'weight_decay': float(cfg_yaml.get('weight_decay', 1e-06) or 1e-06), 'seed': int(_resolve_training_seed(cfg_yaml)), 'checkpoints': [{'name': name, 'path': path, 'sha1': _file_sha1_cached(path)} for name, path in init_specs if path is not None]}

def _extract_high_fidelity_size_objectives(fitness: Mapping[str, Any], *, valid_sizes: Sequence[int]) -> Tuple[Dict[int, float], float]:
    size_objectives_raw = fitness.get('size_objectives', {})
    size_objectives: Dict[int, float] = {}
    if isinstance(size_objectives_raw, Mapping):
        for k, v in size_objectives_raw.items():
            try:
                size_objectives[int(k)] = float(v)
            except Exception:
                continue
    by_size: Dict[int, float] = {}
    for sz in valid_sizes:
        if int(sz) not in size_objectives:
            raise RuntimeError(f'Missing size_objectives[{int(sz)}] while generating high_fidelity baseline cache')
        by_size[int(sz)] = float(size_objectives[int(sz)])
    agg = float(sum((by_size[int(sz)] for sz in valid_sizes)) / max(len(valid_sizes), 1))
    return (by_size, float(agg))

def _evaluate_high_fidelity_reference_baseline(*, cfg_yaml: Mapping[str, Any], hf_cfg: HighFidelityConfig, operator_whitelist: Sequence[str], init_ckpt: str | None) -> Dict[str, Any]:
    init_ckpt_abs = _abs_from_repo_root(str(init_ckpt)) if init_ckpt else None
    compiled_builder = compile_preference_builder(_ref_builder_ir(), operator_whitelist=list(operator_whitelist))
    ref_loss_ir = _ref_loss_ir()
    static_ref = run_static_gates(ref_loss_ir, operator_whitelist=list(operator_whitelist))
    if not static_ref.ok:
        raise RuntimeError(f'Reference loss failed static gates: {static_ref.reason}')
    compiled_loss = compile_free_loss(ref_loss_ir, operator_whitelist=list(operator_whitelist))
    adapter = _CompiledBuilderAdapter(compiled_builder)
    free_cfg = FreeLossFidelityConfig(hf=hf_cfg, init_checkpoint_path=init_ckpt_abs, scratch_hf_epochs=int(cfg_yaml.get('scratch_hf_epochs', 0) or 0), warmstart_hf_epochs=int(cfg_yaml.get('warmstart_hf_epochs', 0) or 0))
    return evaluate_free_loss_candidate(compiled_loss, free_cfg, pref_builder=adapter)

def _high_fidelity_fidelity_key(cfg_yaml: Mapping[str, Any]) -> str:
    hf_epochs = int(cfg_yaml.get('hf_epochs', 0) or 0)
    hf_instances = int(cfg_yaml.get('hf_instances_per_epoch', 0) or 0)
    if hf_epochs > 0 and hf_instances > 0:
        return f'epoch{int(hf_epochs)}_inst{int(hf_instances)}'
    K = int(cfg_yaml.get('f1_steps', 32) or 32)
    return f'K{int(K)}'

def _default_high_fidelity_baseline_mini_eval_path(cfg_yaml: Mapping[str, Any]) -> str:
    env_name = str(cfg_yaml.get('env_name') or cfg_yaml.get('problem') or 'tsp').strip().lower()
    train_problem_size = int(cfg_yaml.get('train_problem_size', 20) or 20)
    fidelity = _high_fidelity_fidelity_key(cfg_yaml)
    return os.path.join('baseline', 'mini_eval', f'baseline_minitrain_{env_name}{int(train_problem_size)}_{str(fidelity)}.json').replace('\\', '/')

def _high_fidelity_baseline_cfg_dict(cfg_yaml: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(cfg_yaml, dict):
        baseline_cfg = cfg_yaml.get('baseline', {}) or {}
        return dict(baseline_cfg) if isinstance(baseline_cfg, Mapping) else {}
    baseline_cfg = cfg_yaml.get('baseline', {}) or {}
    if not isinstance(baseline_cfg, dict):
        baseline_cfg = {}
        cfg_yaml['baseline'] = baseline_cfg
    return baseline_cfg

def _record_high_fidelity_baseline_mini_eval_path(cfg_yaml: Mapping[str, Any], path: str) -> None:
    baseline_cfg = _high_fidelity_baseline_cfg_dict(cfg_yaml)
    fidelity = _high_fidelity_fidelity_key(cfg_yaml)
    raw_map = baseline_cfg.get('mini_eval_paths')
    if not isinstance(raw_map, dict):
        raw_map = {}
        baseline_cfg['mini_eval_paths'] = raw_map
    raw_map[str(fidelity)] = str(path)
    if str(fidelity).startswith('K'):
        try:
            raw_map[str(int(str(fidelity)[1:]))] = str(path)
        except Exception:
            pass
    baseline_cfg['mini_eval_path'] = str(path)

def _resolve_high_fidelity_baseline_mini_eval_path(cfg_yaml: Mapping[str, Any], baseline_cfg: Mapping[str, Any]) -> str | None:
    raw = baseline_cfg.get('mini_eval_paths', None)
    if raw is None:
        raw = baseline_cfg.get('mini_eval_path', None)
    if isinstance(raw, Mapping):
        key = _high_fidelity_fidelity_key(cfg_yaml)
        for cand in (key, str(key).replace('K', ''), str(key).lower(), str(key).upper()):
            if cand in raw and raw.get(cand):
                return str(raw.get(cand))
        if key.startswith('K'):
            try:
                k_int = int(key[1:])
            except Exception:
                k_int = None
            if k_int is not None:
                for cand in (k_int, str(k_int)):
                    if cand in raw and raw.get(cand):
                        return str(raw.get(cand))
        for cand in ('default', 'DEFAULT', '_default_', '*'):
            if cand in raw and raw.get(cand):
                return str(raw.get(cand))
        return _default_high_fidelity_baseline_mini_eval_path(cfg_yaml)
    if isinstance(raw, str) and raw.strip():
        return str(raw)
    return _default_high_fidelity_baseline_mini_eval_path(cfg_yaml)

@torch.no_grad()
def _high_fidelity_pre_minitrain_eval(*, cfg_yaml: Mapping[str, Any], init_checkpoint: str | None, train_problem_size: int, valid_problem_sizes: Sequence[int], num_validation_episodes: int, train_batch_size: int, scratch_init_seed: int) -> Tuple[Dict[int, float], float]:
    from fitness.free_loss_fidelity import _evaluate_rl4co_model, _load_policy_weights_from_checkpoint, _rl4co_build_env, _rl4co_build_policy
    generator_params = dict(cfg_yaml.get('generator_params', {}) or {})
    hf_cfg = HighFidelityConfig(problem=str(cfg_yaml.get('problem', 'tsp')), env_name=str(cfg_yaml.get('env_name') or cfg_yaml.get('problem', 'tsp')), env_kwargs=dict(cfg_yaml.get('env_kwargs', {}) or {}), generator_params=generator_params, policy_name=str(cfg_yaml.get('policy_name', '') or ''), policy_kwargs=dict(cfg_yaml.get('policy_kwargs', {}) or {}), rollout_strategy=str(cfg_yaml.get('rollout_strategy', 'auto') or 'auto'), hf_steps=1, hf_epochs=0, hf_instances_per_epoch=0, train_problem_size=int(train_problem_size), valid_problem_sizes=tuple((int(x) for x in valid_problem_sizes)), train_batch_size=int(train_batch_size), pomo_size=int(cfg_yaml.get('pomo_size')) if cfg_yaml.get('pomo_size', None) is not None else None, learning_rate=float(cfg_yaml.get('learning_rate', 0.0003) or 0.0003), weight_decay=float(cfg_yaml.get('weight_decay', 1e-06) or 1e-06), alpha=_alpha_from_cfg(cfg_yaml), device=str(cfg_yaml.get('device', 'cuda') or 'cuda'), seed=int(scratch_init_seed), num_validation_episodes=int(num_validation_episodes), validation_batch_size=int(cfg_yaml.get('validation_batch_size', 64) or 64))
    _set_seed(int(hf_cfg.seed))
    device_str = str(hf_cfg.device)
    if device_str == 'cuda' and (not torch.cuda.is_available()):
        device_str = 'cpu'
    device = torch.device(device_str)
    env = None
    policy = None
    try:
        env = _rl4co_build_env(hf_cfg, int(train_problem_size)).to(device)
        policy, rollout_strategy = _rl4co_build_policy(hf_cfg, env)
        if init_checkpoint:
            _load_policy_weights_from_checkpoint(policy, _abs_from_repo_root(str(init_checkpoint)))
        policy = policy.to(device)
        policy.eval()
        by_size: Dict[int, float] = {}
        for sz in valid_problem_sizes:
            obj = _evaluate_rl4co_model(policy=policy, cfg=hf_cfg, problem_size=int(sz), device=device, num_episodes=int(num_validation_episodes), batch_size=int(hf_cfg.validation_batch_size), rollout_strategy=str(rollout_strategy))
            by_size[int(sz)] = float(obj)
        aggregated = float(sum((by_size[int(sz)] for sz in valid_problem_sizes)) / max(len(valid_problem_sizes), 1))
        return (by_size, float(aggregated))
    finally:
        policy = None
        env = None
        _high_fidelity_cleanup_eval_device(str(device))

def _ensure_high_fidelity_baseline_mini_eval(*, cfg_yaml: Mapping[str, Any], operator_whitelist: Sequence[str], device_str: str) -> Dict[str, Any]:
    baseline_cfg = _high_fidelity_baseline_cfg_dict(cfg_yaml)
    mini_eval_path = _resolve_high_fidelity_baseline_mini_eval_path(cfg_yaml, baseline_cfg)
    if not mini_eval_path:
        raise ValueError('Failed to resolve high_fidelity baseline mini-eval path')
    _record_high_fidelity_baseline_mini_eval_path(cfg_yaml, str(mini_eval_path))
    expected_sig = _build_high_fidelity_eval_signature(cfg_yaml)
    init_specs = _high_fidelity_init_specs_from_baseline_cfg(cfg_yaml)
    if not init_specs:
        raise ValueError('high_fidelity baseline requires at least one init source (scratch and/or baseline.checkpoints)')
    expected_init_names = [str(init_name) for init_name, _ in init_specs]
    existing = None
    if os.path.isfile(_abs_from_repo_root(str(mini_eval_path))):
        try:
            existing = _load_baseline_mini_eval(str(mini_eval_path))
        except Exception:
            existing = None
    existing_per_init: Dict[str, Any] = {}
    if isinstance(existing, Mapping) and existing.get('eval_signature') == expected_sig and isinstance(existing.get('per_init'), Mapping):
        existing_per_init = {str(init_name): dict(init_payload) for init_name, init_payload in dict(existing.get('per_init') or {}).items() if str(init_name) in expected_init_names and isinstance(init_payload, Mapping)}
        if all((init_name in existing_per_init for init_name in expected_init_names)):
            return {'path': str(mini_eval_path), 'cached': True, 'regenerated': False, 'eval_signature': expected_sig}
    scratch_init_seed = int(_resolve_training_seed(cfg_yaml))
    train_problem_size = int(cfg_yaml.get('train_problem_size', 20) or 20)
    valid_problem_sizes = [int(v) for v in cfg_yaml.get('valid_problem_sizes', [train_problem_size])]
    valid_problem_sizes = list(dict.fromkeys(valid_problem_sizes))
    num_validation_episodes = int(cfg_yaml.get('num_validation_episodes', 128) or 128)
    train_batch_size = int(cfg_yaml.get('train_batch_size', 64) or 64)
    K = int(cfg_yaml.get('f1_steps', 32) or 32)
    generator_params = dict(cfg_yaml.get('generator_params', {}) or {})
    cfg_hf = dict(cfg_yaml)
    cfg_hf['f1_steps'] = int(K)
    if not (int(cfg_yaml.get('hf_epochs', 0) or 0) > 0 and int(cfg_yaml.get('hf_instances_per_epoch', 0) or 0) > 0):
        cfg_hf['hf_epochs'] = 0
        cfg_hf['hf_instances_per_epoch'] = 0
    hf_cfg = _build_hf_cfg(cfg_hf, seed=int(scratch_init_seed), device_str=str(device_str))
    per_init: Dict[str, Any] = dict(existing_per_init)

    def _write_baseline_payload(*, complete: bool) -> None:
        payload: Dict[str, Any] = {'schema_version': 1, 'created_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'config_path': None, 'eval_signature': expected_sig, 'per_init': dict(per_init), 'reference': {'builder_ir': asdict(_ref_builder_ir()), 'loss_ir': asdict(_ref_loss_ir())}, 'complete': bool(complete)}
        resolved_path = _abs_from_repo_root(str(mini_eval_path))
        _atomic_write_json(resolved_path, payload)
        _BASELINE_MINI_EVAL_CACHE[resolved_path] = dict(payload)
    for init_name, init_ckpt in init_specs:
        if str(init_name) in per_init:
            continue
        try:
            pre_by_size, pre_agg = _high_fidelity_pre_minitrain_eval(cfg_yaml=cfg_yaml, init_checkpoint=init_ckpt, train_problem_size=int(train_problem_size), valid_problem_sizes=list(valid_problem_sizes), num_validation_episodes=int(num_validation_episodes), train_batch_size=int(train_batch_size), scratch_init_seed=int(scratch_init_seed))
            fitness = _evaluate_high_fidelity_reference_baseline(cfg_yaml=cfg_yaml, hf_cfg=hf_cfg, operator_whitelist=operator_whitelist, init_ckpt=init_ckpt)
            by_size, agg = _extract_high_fidelity_size_objectives(fitness, valid_sizes=valid_problem_sizes)
            per_init[str(init_name)] = {'pre_val_objective_by_size': {str(int(k)): float(v) for k, v in pre_by_size.items()}, 'pre_val_reward_by_size': {str(int(k)): float(-float(v)) for k, v in pre_by_size.items()}, 'pre_aggregated_objective': float(pre_agg), 'pre_aggregated_reward': float(-float(pre_agg)), 'val_objective_by_size': {str(int(k)): float(v) for k, v in by_size.items()}, 'val_reward_by_size': {str(int(k)): float(-float(v)) for k, v in by_size.items()}, 'aggregated_objective': float(agg), 'aggregated_reward': float(-float(agg)), 'delta_objective_post_minus_pre': float(float(agg) - float(pre_agg)), 'delta_reward_post_minus_pre': float(float(pre_agg) - float(agg)), 'init_checkpoint': str(init_ckpt) if init_ckpt else None}
            _write_baseline_payload(complete=all((name in per_init for name in expected_init_names)))
        finally:
            _high_fidelity_cleanup_eval_device(str(device_str))
    return {'path': str(mini_eval_path), 'cached': False, 'regenerated': True, 'eval_signature': expected_sig}

def _infer_baseline_epoch_from_path(path: str) -> int | None:
    name = os.path.basename(str(path))
    m = re.search('(?:^|[._-])epoch_(\\d+)(?:\\D|$)', name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None

def _timestamp_dir(root: str) -> str:
    ts = time.strftime('%Y%m%d-%H%M%S')
    path = os.path.join(root, ts)
    os.makedirs(path, exist_ok=True)
    return path

def _append_jsonl(path: str, records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        return
    with open(path, 'a', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(dict(rec), ensure_ascii=False) + '\n')

def _atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f'{path}.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(dict(payload), f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def _load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def _load_stage1_loss_entry(loss_path: str) -> Dict[str, Any]:
    payload = _load_json(loss_path)
    if not isinstance(payload, Mapping) or not isinstance(payload.get('ir'), Mapping):
        raise ValueError(f'Invalid Stage 1 loss artifact: {loss_path}')
    ir = free_loss_ir_from_json(dict(payload['ir']))
    signature = _sig_free_loss(ir)
    source_id = str(payload.get('id') or signature[:8])
    mechanism_family = _canonical_mechanism_family(payload.get('mechanism_family')) or _mechanism_family_candidates(ir)[0]
    return {'generation': -1, 'index': 0, 'id': f'fstage1_{signature[:8]}', 'signature': signature, 'mechanism_family': mechanism_family, 'origin': 'STAGE_HANDOFF', 'origin_base': source_id, 'op_type': 'STAGE_HANDOFF', 'parents': [source_id], 'attempt': 0, 'prompt_sha1': None, 'prompt_path': None, 'llm_seed': None, 'history': [], 'ir': asdict(ir), 'static_ok': True, 'static_reason': 'stage_handoff', 'static_trace': {}, 'compile_ok': True, 'compile_reason': 'stage_handoff', 'fitness': 0.0}

def _stage0_sandbox_script_path() -> str:
    return os.path.join(_repo_root_dir(), 'PTP', 'ptp_discovery', 'run_stage0_sandbox_gate.py')

def _run_stage0_sandbox_gate(*, run_dir: str, generation: int, pair_index: int, g_id: str, f_id: str, g_ir: PreferenceBuilderIR, f_ir: FreeLossIR, operator_whitelist: Sequence[str], cfg_yaml: Mapping[str, Any]) -> Dict[str, Any]:
    script_path = _stage0_sandbox_script_path()
    if not os.path.isfile(script_path):
        return {'ok': False, 'failure_kind': 'sandbox_script_missing', 'reason': f'Missing sandbox script: {script_path}'}
    run_dir_s = str(run_dir or '').strip()
    if not run_dir_s:
        run_dir_s = os.path.join(_repo_root_dir(), 'runs', '_sandbox')
    task_dir = os.path.join(run_dir_s, 'stage0_sandbox', f'gen{int(generation):03d}_pair{int(pair_index):03d}_{str(g_id)[:16]}_{str(f_id)[:16]}')
    os.makedirs(task_dir, exist_ok=True)
    payload_path = os.path.join(task_dir, 'payload.json')
    result_path = os.path.join(task_dir, 'result.json')
    log_path = os.path.join(task_dir, 'subprocess.log')
    payload = {'generation': int(generation), 'pair_index': int(pair_index), 'g_entry': {'id': str(g_id), 'ir': asdict(g_ir)}, 'f_entry': {'id': str(f_id), 'ir': asdict(f_ir)}, 'operator_whitelist': list(operator_whitelist)}
    _atomic_write_json(payload_path, payload)
    timeout_raw = cfg_yaml.get('stage0_sandbox_timeout_s', 25.0)
    try:
        timeout_s = max(1.0, float(timeout_raw))
    except (TypeError, ValueError):
        timeout_s = 25.0
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')
    proc: subprocess.Popen[Any] | None = None
    log_fh = open(log_path, 'w', encoding='utf-8')
    try:
        proc = subprocess.Popen([sys.executable, '-u', script_path, '--payload', payload_path, '--result', result_path], cwd=_repo_root_dir(), env=env, stdout=log_fh, stderr=log_fh)
        try:
            proc.wait(timeout=float(timeout_s))
        except subprocess.TimeoutExpired:
            try:
                proc.terminate()
                proc.wait(timeout=5.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            return {'ok': False, 'failure_kind': 'sandbox_timeout', 'reason': f'Sandbox gate timed out after {timeout_s:.1f}s', 'exit_code': proc.poll(), 'sandbox_log': os.path.relpath(log_path, start=run_dir_s)}
    except Exception as exc:
        return {'ok': False, 'failure_kind': 'sandbox_runtime_error', 'reason': f'Failed launching sandbox subprocess: {exc}', 'exception_type': type(exc).__name__}
    finally:
        try:
            log_fh.close()
        except Exception:
            pass
    if not os.path.isfile(result_path):
        return {'ok': False, 'failure_kind': 'sandbox_no_result', 'reason': 'Sandbox subprocess exited without a result payload', 'exit_code': None if proc is None else proc.poll(), 'sandbox_log': os.path.relpath(log_path, start=run_dir_s)}
    try:
        loaded = _load_json(result_path)
    except Exception as exc:
        return {'ok': False, 'failure_kind': 'sandbox_result_invalid', 'reason': f'Failed to parse sandbox result: {exc}', 'exit_code': None if proc is None else proc.poll(), 'sandbox_log': os.path.relpath(log_path, start=run_dir_s)}
    if not isinstance(loaded, Mapping):
        return {'ok': False, 'failure_kind': 'sandbox_result_invalid', 'reason': 'Sandbox result payload must be a dict', 'exit_code': None if proc is None else proc.poll(), 'sandbox_log': os.path.relpath(log_path, start=run_dir_s)}
    out = dict(loaded)
    out.setdefault('ok', False)
    out.setdefault('failure_kind', None if bool(out.get('ok')) else 'sandbox_gate_failed')
    out.setdefault('reason', 'ok' if bool(out.get('ok')) else 'sandbox gate failed')
    out['exit_code'] = None if proc is None else proc.poll()
    out['sandbox_log'] = os.path.relpath(log_path, start=run_dir_s)
    out['sandbox_result'] = os.path.relpath(result_path, start=run_dir_s)
    return out

def _worker_device_fields(payload_like: Mapping[str, Any]) -> Tuple[str, str]:
    logical_device = str(payload_like.get('device_str') or '').strip()
    physical_device = str(payload_like.get('device_physical_str') or logical_device).strip()
    return (physical_device, logical_device)

def _sig(obj: Mapping[str, Any]) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return sha1(blob).hexdigest()

def _sig_free_loss(ir: FreeLossIR) -> str:
    return _sig(asdict(ir))
_LOSS_FINGERPRINT_CACHE: Dict[str, Dict[str, Any]] = {}
_BUILDER_FINGERPRINT_CACHE: Dict[str, Dict[str, Any]] = {}
_BUILDER_TAG_KEYS = ('geometry', 'cap', 'weighting', 'constraint')
_LOSS_TAG_KEYS = ('signal', 'link', 'aggregation', 'constraint')
_MECHANISM_FAMILIES = ('cost_calibrated', 'rank_based', 'pairwise_margin')

def _loss_fingerprint(ir: FreeLossIR) -> Dict[str, Any]:
    sig = _sig_free_loss(ir)
    cached = _LOSS_FINGERPRINT_CACHE.get(sig)
    if isinstance(cached, dict):
        return cached
    code = str(getattr(ir, 'code', '') or '')
    tokens: List[str] = []
    call_names: List[str] = []

    def _call_name(expr: ast.AST) -> str | None:
        if isinstance(expr, ast.Name):
            return str(expr.id)
        if isinstance(expr, ast.Attribute):
            parts: List[str] = []
            cur: ast.AST | None = expr
            while isinstance(cur, ast.Attribute):
                parts.append(str(cur.attr))
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(str(cur.id))
            if parts:
                return '.'.join(reversed(parts))
        return None
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name:
                    tokens.append(f'call:{name}')
                    call_names.append(name)
            elif isinstance(node, ast.BinOp):
                tokens.append(f'binop:{type(node.op).__name__}')
            elif isinstance(node, ast.UnaryOp):
                tokens.append(f'unop:{type(node.op).__name__}')
            elif isinstance(node, ast.Compare):
                for op in node.ops:
                    tokens.append(f'cmp:{type(op).__name__}')
            elif isinstance(node, ast.BoolOp):
                tokens.append(f'bool:{type(node.op).__name__}')
            elif isinstance(node, (ast.IfExp, ast.If)):
                tokens.append('if')
            elif isinstance(node, (ast.For, ast.While)):
                tokens.append('loop')
            elif isinstance(node, ast.Return):
                tokens.append('return')
    except Exception:
        try:
            keep_ops = {'+', '-', '*', '/', '**', '<', '>', '<=', '>=', '==', '!=', '%'}
            for tok in tokenize.generate_tokens(io.StringIO(code).readline):
                if tok.type in {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}:
                    continue
                if tok.type in {tokenize.STRING, tokenize.NUMBER}:
                    continue
                if tok.type == tokenize.OP and tok.string not in keep_ops:
                    continue
                if tok.string:
                    tokens.append(tok.string)
        except Exception:
            tokens = []
    unigrams: set[str] = set(tokens)
    bigrams: set[str] = set()
    for a, b in zip(tokens, tokens[1:]):
        bigrams.add(f'{a}->{b}')
    call_set: set[str] = set(call_names)
    fp = {'sig': sig, 'token_unigrams': unigrams, 'token_bigrams': bigrams, 'call_names_set': call_set, 'call_names_top': sorted(set(call_names))[:32]}
    _LOSS_FINGERPRINT_CACHE[sig] = fp
    return fp

def _normalize_tag_value(value: Any) -> str:
    text = str(value or '').strip()
    return text if text else 'unknown'

def _operator_tags_from_hparams(hparams: Mapping[str, Any] | None, *, keys: Sequence[str]) -> Dict[str, str]:
    hp = hparams if isinstance(hparams, Mapping) else {}
    out: Dict[str, str] = {}
    for key in keys:
        raw = hp.get(str(key)) if isinstance(hp, Mapping) else None
        out[str(key)] = _normalize_tag_value(raw)
    return out

def _builder_operator_tags(ir: PreferenceBuilderIR) -> Dict[str, str]:
    return _operator_tags_from_hparams(getattr(ir, 'hyperparams', {}), keys=_BUILDER_TAG_KEYS)

def _loss_operator_tags(ir: FreeLossIR) -> Dict[str, str]:
    return _operator_tags_from_hparams(getattr(ir, 'hyperparams', {}), keys=_LOSS_TAG_KEYS)

def _canonical_mechanism_family(value: Any) -> str | None:
    family = str(value or '').strip().lower()
    return family if family in _MECHANISM_FAMILIES else None

def _mechanism_family_candidates(ir: Any) -> List[str]:
    hyperparams = getattr(ir, 'hyperparams', {})
    hyperparams = hyperparams if isinstance(hyperparams, Mapping) else {}
    raw = hyperparams.get('mechanism_families')
    if not isinstance(raw, (list, tuple)) or not raw:
        return []
    candidates: List[str] = []
    for value in raw:
        family = _canonical_mechanism_family(value)
        if family is None:
            return []
        if family not in candidates:
            candidates.append(family)
    return candidates

def _entry_mechanism_family(entry: Mapping[str, Any]) -> str:
    assigned = _canonical_mechanism_family(entry.get('mechanism_family'))
    if assigned is None:
        raise ValueError('Candidate has no assigned mechanism family.')
    return assigned

def _assign_mechanism_family(ir: Any, rng: random.Random, *, allowed: Sequence[str] | None=None) -> str | None:
    candidates = _mechanism_family_candidates(ir)
    if allowed is not None:
        allowed_set = {family for family in (_canonical_mechanism_family(value) for value in allowed) if family is not None}
        candidates = [family for family in candidates if family in allowed_set]
    return rng.choice(candidates) if candidates else None

def _proposal_mechanism_family(ir: Any, rng: random.Random, *, op: str, parents: Sequence[Mapping[str, Any]]) -> str | None:
    parent_families = [_entry_mechanism_family(parent) for parent in parents]
    if op == 'PARADIGM_SHIFT' and parent_families:
        counts = collections.Counter(parent_families)
        dominant = counts.most_common(1)[0][0]
        return _assign_mechanism_family(ir, rng, allowed=[family for family in _MECHANISM_FAMILIES if family != dominant])
    if op in {'TUNE', 'STRUCTURE_SHIFT', 'CONSTRAINT_INJECT'} and parent_families:
        return _assign_mechanism_family(ir, rng, allowed=[parent_families[0]])
    return _assign_mechanism_family(ir, rng)

def _family_aware_survivors(entries: Sequence[Mapping[str, Any]], *, slots: int) -> List[Dict[str, Any]]:
    limit = max(0, int(slots))
    if limit <= 0:
        return []
    ranked = [dict(entry) for entry in entries]
    ranked.sort(key=lambda entry: float(entry.get('fitness', float('inf'))))
    cap = max(1, int(math.floor(0.375 * limit)))
    represented = [family for family in _MECHANISM_FAMILIES if any((_entry_mechanism_family(entry) == family for entry in ranked))]
    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    counts: collections.Counter[str] = collections.Counter()
    for family in represented:
        candidate = next((entry for entry in ranked if _entry_mechanism_family(entry) == family), None)
        if candidate is None or len(selected) >= limit:
            continue
        selected.append(candidate)
        selected_ids.add(str(candidate.get('id', '')))
        counts[family] += 1
    for entry in ranked:
        if len(selected) >= limit:
            break
        entry_id = str(entry.get('id', ''))
        family = _entry_mechanism_family(entry)
        if entry_id in selected_ids or counts[family] >= cap:
            continue
        selected.append(entry)
        selected_ids.add(entry_id)
        counts[family] += 1
    selected.sort(key=lambda entry: float(entry.get('fitness', float('inf'))))
    return selected

def _best_pair_artifact_entry(*, cid: str, best_pair: Mapping[str, Any] | None, cid_key: str, ir_key: str, candidate_map: Mapping[str, Any], compiled_map: Mapping[str, Any], ref_ir_fn: Any) -> Dict[str, Any] | None:
    cid_s = str(cid or '').strip()
    if not cid_s:
        return None
    if isinstance(best_pair, Mapping) and str(best_pair.get(cid_key, '')).strip() == cid_s:
        ir = best_pair.get(ir_key)
        if isinstance(ir, Mapping) and isinstance(ir.get('code'), str):
            out = {'id': cid_s, 'ir': dict(ir)}
            stored = candidate_map.get(cid_s) if isinstance(candidate_map, Mapping) else None
            if isinstance(stored, Mapping):
                out['mechanism_family'] = _entry_mechanism_family(stored)
            return out
    entry = candidate_map.get(cid_s) if isinstance(candidate_map, Mapping) else None
    if isinstance(entry, Mapping) and entry.get('id'):
        ir = entry.get('ir')
        if isinstance(ir, Mapping) and isinstance(ir.get('code'), str):
            out = dict(entry)
            out['id'] = str(out.get('id') or cid_s)
            return out
    comp = compiled_map.get(cid_s) if isinstance(compiled_map, Mapping) else None
    ir = getattr(comp, 'ir', None)
    if ir is not None:
        try:
            irj = asdict(ir)
        except Exception:
            irj = None
        if isinstance(irj, Mapping) and isinstance(irj.get('code'), str):
            return {'id': cid_s, 'ir': dict(irj)}
    try:
        ref_ir = ref_ir_fn() if callable(ref_ir_fn) else None
        if ref_ir is not None:
            irj = asdict(ref_ir)
            if isinstance(irj, Mapping) and isinstance(irj.get('code'), str):
                return {'id': cid_s, 'ir': dict(irj)}
    except Exception:
        pass
    return None

def _best_pair_eval_metadata(best_pair: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(best_pair, Mapping):
        return {}
    out: Dict[str, Any] = {}
    top_level_aliases = ('stage', 'stage_final', 'score', 'final_score', 'pair_ok', 'pair_reason', 'compare_target', 'metric_mode', 'improve_eps', 'reference_score', 'better_than_incumbent')
    for key in top_level_aliases:
        if key not in best_pair:
            continue
        value = best_pair.get(key)
        if isinstance(value, dict):
            out[key] = dict(value)
        elif isinstance(value, list):
            out[key] = list(value)
        else:
            out[key] = value
    alias_map = {'generation': 'best_pair_generation', 'phase': 'best_pair_phase', 'last_phase_label': 'best_pair_last_phase_label', 'last_phase_reference_score': 'best_pair_last_phase_reference_score'}
    for src, dst in alias_map.items():
        if src in best_pair:
            out[dst] = best_pair.get(src)
    return out

def _pair_record_effective_score(rec: Mapping[str, Any] | None) -> float | None:
    if not isinstance(rec, Mapping):
        return None
    for key in ('final_score', 'score'):
        try:
            value = float(rec.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None

def _pair_record_beats_reference_record(candidate: Mapping[str, Any] | None, reference: Mapping[str, Any] | None, *, metric_mode: str, improve_eps: float) -> bool:
    cand_score = _pair_record_effective_score(candidate)
    ref_score = _pair_record_effective_score(reference)
    if cand_score is None or not math.isfinite(cand_score):
        return False
    return _is_better_than_reference(cand_score=float(cand_score), reference_score=float(ref_score) if ref_score is not None else None, metric_mode=metric_mode, improve_eps=improve_eps)

def _pair_record_sort_key(rec: Mapping[str, Any], *, metric_mode: str) -> Tuple[Any, ...]:
    score = _pair_record_effective_score(rec)
    score_key = float('inf')
    if score is not None and math.isfinite(score):
        score_key = float(score)
    return (float(score_key),)

def _resolve_best_pair_record(*, best_so_far: Mapping[str, Any] | None, pair_records: Sequence[Mapping[str, Any]] | None, pair_cache_records: Sequence[Mapping[str, Any]] | None=None, metric_mode: str='minimize') -> Dict[str, Any] | None:
    if not isinstance(best_so_far, Mapping):
        return None
    gid_best = str(best_so_far.get('builder_id', '')).strip()
    fid_best = str(best_so_far.get('loss_id', '')).strip()
    if not gid_best or not fid_best:
        return None
    target_score = _pair_record_effective_score({'score': best_so_far.get('score')})
    target_generation = _safe_int(best_so_far.get('generation', -1), -1)
    target_phase = str(best_so_far.get('phase', '')).strip()
    target_stage_final = str(best_so_far.get('stage_final', '')).strip()
    candidates: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, ...]] = set()
    for source in (pair_records or [], pair_cache_records or []):
        for rec in source:
            if not isinstance(rec, Mapping):
                continue
            if str(rec.get('g_id', '')).strip() != gid_best or str(rec.get('f_id', '')).strip() != fid_best:
                continue
            sig = (_safe_int(rec.get('generation', -1), -1), _safe_int(rec.get('pair_index', -1), -1), str(rec.get('phase', '')), str(rec.get('stage', '')), str(rec.get('stage_final', '')), _pair_record_effective_score(rec))
            if sig in seen:
                continue
            seen.add(sig)
            candidates.append(dict(rec))
    if not candidates:
        return None

    def _stage_rank(rec: Mapping[str, Any]) -> int:
        stage_final = str(rec.get('stage_final', rec.get('stage', ''))).strip().lower()
        return 0 if stage_final == 'high_fidelity' else 1

    def _score_matches(rec: Mapping[str, Any]) -> bool:
        cand_score = _pair_record_effective_score(rec)
        if cand_score is None or target_score is None:
            return False
        return abs(cand_score - target_score) <= 1e-12

    def _meta_matches(rec: Mapping[str, Any]) -> bool:
        return _safe_int(rec.get('generation', -1), -1) == target_generation and str(rec.get('phase', '')).strip() == target_phase and (str(rec.get('stage_final', rec.get('stage', ''))).strip() == target_stage_final)
    matched = [rec for rec in candidates if _score_matches(rec) or _meta_matches(rec)]
    pool = matched or candidates

    def _sort_key(rec: Mapping[str, Any]) -> Tuple[Any, ...]:
        return (0 if _score_matches(rec) else 1, 0 if _meta_matches(rec) else 1, _stage_rank(rec), 0 if isinstance(rec.get('fitness'), Mapping) else 1, *_pair_record_sort_key(rec, metric_mode=metric_mode), -_safe_int(rec.get('generation', -1), -1), -_safe_int(rec.get('pair_index', -1), -1))
    return dict(min(pool, key=_sort_key))

def _pair_history_key(g_id: Any, f_id: Any) -> str:
    return f'{str(g_id or '').strip()}::{str(f_id or '').strip()}'

def _pair_score_history_entry(rec: Mapping[str, Any]) -> Dict[str, Any] | None:
    if rec.get('pair_ok') is False:
        return None
    try:
        final_score = float(rec.get('final_score'))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(final_score):
        return None
    out: Dict[str, Any] = {'generation': _safe_int(rec.get('generation', -1), -1), 'pair_index': _safe_int(rec.get('pair_index', -1), -1), 'phase': str(rec.get('phase', 'unknown')), 'stage': str(rec.get('stage', 'unknown')), 'stage_final': str(rec.get('stage_final', rec.get('stage', 'none'))), 'score': float(final_score)}
    try:
        reference_score = rec.get('reference_score')
        if reference_score is not None:
            out['reference_score'] = float(reference_score)
    except (TypeError, ValueError):
        pass
    return out

def _append_pair_score_history(pair_score_history_map: Dict[str, List[Dict[str, Any]]], rec: Mapping[str, Any]) -> None:
    key = _pair_history_key(rec.get('g_id'), rec.get('f_id'))
    if key == '::':
        return
    entry = _pair_score_history_entry(rec)
    if entry is None:
        return
    history = pair_score_history_map.setdefault(key, [])
    entry_sig = (int(entry.get('generation', -1)), int(entry.get('pair_index', -1)), str(entry.get('stage_final')), float(entry.get('score')))
    if history:
        last = history[-1]
        last_sig = (_safe_int(last.get('generation', -1), -1), _safe_int(last.get('pair_index', -1), -1), str(last.get('stage_final')), float(last.get('score', float('nan'))))
        if last_sig == entry_sig:
            return
    history.append(entry)

def _score_history_summary(history: Sequence[Mapping[str, Any]] | None) -> Dict[str, Any]:
    items = list(history or [])
    scores: List[float] = []
    for item in items:
        try:
            score = float(item.get('score'))
        except (TypeError, ValueError, AttributeError):
            continue
        if math.isfinite(score):
            scores.append(score)
    if not scores:
        return {'count': 0, 'scores': []}
    return {'count': int(len(scores)), 'scores': list(scores), 'best': float(min(scores)), 'worst': float(max(scores)), 'mean': float(sum(scores) / len(scores)), 'latest': float(scores[-1])}

def _sig_pref_builder(ir: PreferenceBuilderIR) -> str:
    return _sig(asdict(ir))

def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)

def _std(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if len(vals) < 2:
        return 0.0
    mu = sum(vals) / len(vals)
    var = sum(((v - mu) ** 2 for v in vals)) / float(len(vals) - 1)
    return float(math.sqrt(max(var, 0.0)))

def _run_loss_invariance_gate(compiled_f: CompiledFreeLoss, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    del cfg
    result = run_affine_invariance_gate(compiled_f, max_abs_delta=0.05, variant='visible')
    return {'affine_gate_ok': bool(result.ok), 'affine_gate_reason': str(result.reason), 'affine_gate_abs_delta': result.abs_delta, 'affine_gate_trace': result.trace}

def _normalize_operator_name(name: str, side: str) -> str:
    raw = str(name or '').strip().upper()
    mapping = {'GEN': 'GEN', 'GENERATE': 'GEN', 'E1_GENERATE': 'GEN', 'XOVER': 'XOVER', 'CROSSOVER': 'XOVER', 'E1': 'XOVER', 'TUNE': 'TUNE', 'M2': 'TUNE', 'PARADIGM_SHIFT': 'PARADIGM_SHIFT', 'STRUCTURE_SHIFT': 'STRUCTURE_SHIFT', 'CONSTRAINT_INJECT': 'CONSTRAINT_INJECT'}
    if str(side) == 'loss' and raw == 'AGG_SHIFT':
        return 'STRUCTURE_SHIFT'
    if str(side) == 'builder' and raw == 'CAP_SHIFT':
        return 'STRUCTURE_SHIFT'
    return mapping.get(raw, raw)

def _expand_operator_bank(side_cfg: Mapping[str, Any], generation: int, rng: random.Random, *, side: str) -> List[str]:
    bank = side_cfg.get('operator_bank', {}) if isinstance(side_cfg, Mapping) else {}
    section = bank.get('init' if int(generation) == 0 else 'per_gen', []) if isinstance(bank, Mapping) else []
    plan = []
    for item in section:
        if not isinstance(item, Mapping):
            continue
        operator = _normalize_operator_name(str(item.get('name', '')), side=side)
        if operator not in {'GEN', 'XOVER', 'TUNE', 'PARADIGM_SHIFT', 'STRUCTURE_SHIFT', 'CONSTRAINT_INJECT'}:
            continue
        plan.extend([operator] * max(0, _safe_int(item.get('count', 0), 0)))
    rng.shuffle(plan)
    return plan

def _majority_parent_tags(parent_irs: Sequence[Any], *, keys: Sequence[str], kind: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key in keys:
        counts: collections.Counter[str] = collections.Counter()
        ordered: List[str] = []
        for ir in parent_irs:
            if str(kind) == 'builder':
                tags = _builder_operator_tags(ir)
            else:
                tags = _loss_operator_tags(ir)
            val = _normalize_tag_value(tags.get(str(key)))
            if val == 'unknown':
                continue
            counts[val] += 1
            ordered.append(val)
        if not counts:
            out[str(key)] = 'unknown'
            continue
        top_n = max(counts.values())
        tied = {k for k, v in counts.items() if int(v) == int(top_n)}
        picked = next((v for v in ordered if v in tied), 'unknown')
        out[str(key)] = picked
    return out

def _missing_required_operator_tags(tags: Mapping[str, Any], *, keys: Sequence[str]) -> List[str]:
    missing: List[str] = []
    for key in keys:
        if _normalize_tag_value(tags.get(str(key))) == 'unknown':
            missing.append(str(key))
    return missing

def _score_delta(*, cand_score: float, ref_score: float | None, metric_mode: str) -> float | None:
    if ref_score is None:
        return None
    return float(ref_score - cand_score)

def _is_better_than_reference(*, cand_score: float, reference_score: float | None, metric_mode: str, improve_eps: float) -> bool:
    if reference_score is None:
        return True
    return bool(cand_score < float(reference_score) - float(improve_eps))

def _score_threshold(*, reference_score: float | None, metric_mode: str, improve_eps: float) -> float | None:
    if reference_score is None:
        return None
    return float(reference_score) - float(improve_eps)

def _build_fixed_side_pairs(*, rng: random.Random, g_id_pool: Sequence[str], f_id_pool: Sequence[str], fixed_builder_id: str, fixed_loss_id: str, budget_loss: int, budget_builder: int) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], List[str]], Dict[Tuple[str, str], str]]:
    pairs: List[Tuple[str, str]] = []
    reasons: Dict[Tuple[str, str], List[str]] = {}
    phases: Dict[Tuple[str, str], str] = {}
    used: set[Tuple[str, str]] = set()

    def _add_pair(gid: str, fid: str, *, reason: str, phase: str) -> None:
        key = (str(gid), str(fid))
        if key in used:
            reasons.setdefault(key, []).append(str(reason))
            return
        used.add(key)
        pairs.append(key)
        reasons.setdefault(key, []).append(str(reason))
        phases[key] = str(phase)
    loss_pool = [str(fid) for fid in f_id_pool if str(fid) != F_REF_ID]
    builder_pool = [str(gid) for gid in g_id_pool if str(gid) != G_REF_ID]
    rng.shuffle(loss_pool)
    rng.shuffle(builder_pool)
    for fid in loss_pool:
        if len([1 for p in pairs if phases.get(p) == 'loss']) >= int(max(0, budget_loss)):
            break
        _add_pair(str(fixed_builder_id), str(fid), reason='stage1_loss_search', phase='loss')
    for gid in builder_pool:
        if len([1 for p in pairs if phases.get(p) == 'builder']) >= int(max(0, budget_builder)):
            break
        _add_pair(str(gid), str(fixed_loss_id), reason='stage2_weight_search', phase='builder')
    return (pairs, reasons, phases)

def _make_builtin_builder_irs(rng: random.Random, n: int) -> List[PreferenceBuilderIR]:
    pool = []
    for index in range(max(1, int(n))):
        power = rng.uniform(0.5, 2.0)
        code = f'def generated_builder(feature_cache, extra):\n    objective = feature_cache["objective"]\n    mask = objective[:, :, None] < objective[:, None, :]\n    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)\n    gap = ops.clamp(objective[b_idx, loser_idx] - objective[b_idx, winner_idx], min=0.0, max=1000000.0)\n    power = float(extra.get("weight_power", {power!r}))\n    weight = ops.pow(gap, power)\n    return PrefBatch(mode="pairwise", pair_idx=(b_idx, winner_idx, loser_idx), weight=weight, meta={{"builder": "all_pairs"}})\n'
        pool.append(PreferenceBuilderIR(name=f'builder_all_pairs_{index:03d}', intuition='all-pairs nonnegative weighting', implementation_hint=PreferenceBuilderImplementationHint(expects=['objective'], returns='PrefBatch', mode='pairwise'), hyperparams={'mechanism_families': ['cost_calibrated'], 'geometry': 'dense_all_pairs', 'cap': 'uncapped_full', 'weighting': 'gap_power', 'constraint': 'nonnegative'}, operators_used=['clamp', 'pow'], code=code))
    return pool

def _truncate_code(s: str, *, max_chars: int=1600) -> str:
    s2 = str(s or '')
    if len(s2) <= max_chars:
        return s2
    return s2[:max_chars - 12] + '\n# ... truncated'

def summarize_best_builder(entry: Mapping[str, Any] | None, *, max_code_chars: int=500) -> Dict[str, Any] | None:
    if not isinstance(entry, Mapping):
        return None
    ir = entry.get('ir')
    if not isinstance(ir, Mapping):
        return None
    code = ir.get('code', '')
    if isinstance(code, str):
        code = _truncate_code(code, max_chars=int(max_code_chars))
    impl = ir.get('implementation_hint') if isinstance(ir.get('implementation_hint'), Mapping) else {}
    return {'id': entry.get('id'), 'fitness': entry.get('fitness'), 'signature': entry.get('signature'), 'name': ir.get('name'), 'mode': (impl or {}).get('mode'), 'expects': (impl or {}).get('expects'), 'intuition': ir.get('intuition'), 'code': code}

def summarize_best_loss(entry: Mapping[str, Any] | None, *, max_code_chars: int=500) -> Dict[str, Any] | None:
    if not isinstance(entry, Mapping):
        return None
    ir = entry.get('ir')
    if not isinstance(ir, Mapping):
        return None
    code = ir.get('code', '')
    if isinstance(code, str):
        code = _truncate_code(code, max_chars=int(max_code_chars))
    hint = ir.get('implementation_hint') if isinstance(ir.get('implementation_hint'), Mapping) else {}
    return {'id': entry.get('id'), 'fitness': entry.get('fitness'), 'signature': entry.get('signature'), 'name': ir.get('name'), 'operators_used': ir.get('operators_used'), 'hyperparams': ir.get('hyperparams'), 'mode': (hint or {}).get('mode'), 'expects': (hint or {}).get('expects'), 'intuition': ir.get('intuition'), 'pseudocode': ir.get('pseudocode'), 'code': code}

def _make_builtin_loss_irs(rng: random.Random, n: int) -> List[FreeLossIR]:
    pool: List[FreeLossIR] = []
    for i in range(max(1, int(n))):
        scale = float(rng.uniform(0.5, 2.0))
        name = f'loss_logsigmoid_{i:03d}'
        expects = ['log_prob_w', 'log_prob_l', 'delta_z']
        code = f"def generated_loss(batch, model_output, extra):\n    lpw = batch['log_prob_w']\n    lpl = batch['log_prob_l']\n    alpha = float(extra.get('alpha', extra.get('hyperparams', {{}}).get('alpha', 1.0)))\n    scale = float(extra.get('hyperparams', {{}}).get('scale', {scale}))\n"
        code += "    quality = 1.0 + 0.1 * ops.tanh(batch['delta_z'])\n    x = alpha * scale * (lpw - lpl) * quality\n    x = ops.clamp(x, -20.0, 20.0)\n    loss = -ops.logsigmoid(x)\n    return loss\n"
        ir_obj = {'name': name, 'intuition': 'rule_based: pairwise logsigmoid loss', 'pseudocode': 'loss = -logsigmoid(clamp(alpha*scale*(lpw-lpl), -20, 20))', 'hyperparams': {'scale': scale, 'mechanism_families': ['cost_calibrated', 'pairwise_margin'], 'signal': 'standardized_gap', 'link': 'logsigmoid', 'aggregation': 'framework_per_instance_weighted_mean', 'constraint': 'clamp_stabilized'}, 'operators_used': ['logsigmoid', 'clamp', 'tanh'], 'implementation_hint': {'expects': expects, 'returns': 'per_pair', 'mode': 'pairwise'}, 'code': code}
        pool.append(free_loss_ir_from_json(ir_obj))
    return pool

def _build_hf_cfg(cfg: Mapping[str, Any], *, seed: int, device_str: str) -> HighFidelityConfig:
    env_name = cfg.get('env_name') or cfg.get('problem', 'tsp')
    generator_params = cfg.get('generator_params', {}) or {}
    env_kwargs = cfg.get('env_kwargs', {}) or {}
    policy_name = cfg.get('policy_name', '') or ''
    policy_kwargs = cfg.get('policy_kwargs', {}) or {}
    rollout_strategy = cfg.get('rollout_strategy', 'auto')
    return HighFidelityConfig(problem=str(cfg.get('problem', 'tsp')), env_name=str(env_name), env_kwargs=dict(env_kwargs), generator_params=dict(generator_params), policy_name=str(policy_name), policy_kwargs=dict(policy_kwargs), rollout_strategy=str(rollout_strategy), hf_steps=int(cfg.get('f1_steps', 32) or 32), hf_epochs=int(cfg.get('hf_epochs', 0) or 0), hf_instances_per_epoch=int(cfg.get('hf_instances_per_epoch', 0) or 0), train_problem_size=int(cfg.get('train_problem_size', 20)), valid_problem_sizes=tuple((int(v) for v in cfg.get('valid_problem_sizes', [100]))), train_batch_size=int(cfg.get('train_batch_size', 64)), pomo_size=int(cfg.get('pomo_size')) if cfg.get('pomo_size', None) is not None else None, learning_rate=float(cfg.get('learning_rate', 0.0003)), weight_decay=float(cfg.get('weight_decay', 1e-06)), alpha=float(cfg.get('alpha', 0.05)), precision='32-true', device=str(device_str), seed=int(seed), num_validation_episodes=int(cfg.get('num_validation_episodes', 128)), validation_batch_size=int(cfg.get('validation_batch_size', 64)))

def _dummy_feature_cache(*, batch_size: int, k: int, variant: str) -> Dict[str, torch.Tensor]:
    b = max(1, int(batch_size))
    kk = max(2, int(k))
    idx = torch.arange(kk, dtype=torch.float32)[None, :].repeat(b, 1)
    objective = idx + torch.arange(b, dtype=torch.float32)[:, None] * 0.01
    log_prob = -0.1 * idx
    if variant == 'hidden':
        objective = objective * 10.0
        log_prob = log_prob * 12.0
    return extract_feature_cache(objective, log_prob)

def _builder_failure_report(*, stage: str, reason: str, trace: Mapping[str, Any] | None=None, error: str | None=None) -> Dict[str, Any]:
    out: Dict[str, Any] = {'stage': str(stage), 'reason': str(reason)}
    if error is not None:
        out['error'] = str(error)
    if trace is not None:
        try:
            out['trace'] = dict(trace)
        except Exception:
            out['trace'] = {'_unserializable_trace': True}
    return out

def _operator_contract_failure(*, op_type: str, reason: str, parent_tags: Any, cand_tags: Mapping[str, Any], required_change: Any) -> Dict[str, Any]:
    return {'stage': 'operator_contract', 'reason': str(reason), 'op_type': str(op_type), 'parent_tags': parent_tags, 'cand_tags': dict(cand_tags), 'required_change': required_change, 'trace': {'failed_gate': 'OperatorContract', 'failure_kind': str(reason)}}

def _validate_loss_operator_contract(ir: FreeLossIR, op_type: str, parent_irs: Sequence[FreeLossIR], parent_entries: Sequence[Mapping[str, Any]] | None=None) -> Tuple[bool, Dict[str, Any]]:
    del parent_entries
    if not _mechanism_family_candidates(ir):
        return (False, _operator_contract_failure(op_type=op_type, reason='invalid_mechanism_families', parent_tags=[], cand_tags={}, required_change={'allowed': list(_MECHANISM_FAMILIES)}))
    op = str(op_type or '').strip().upper()
    if op not in {'LOSS_PARADIGM_SHIFT', 'LOSS_STRUCTURE_SHIFT', 'LOSS_CONSTRAINT_INJECT'}:
        return (True, {})
    cand_tags = _loss_operator_tags(ir)
    missing = _missing_required_operator_tags(cand_tags, keys=_LOSS_TAG_KEYS)
    if missing:
        return (False, _operator_contract_failure(op_type=op, reason='missing_operator_tags', parent_tags=[_loss_operator_tags(p) for p in parent_irs], cand_tags=cand_tags, required_change={'required_keys': list(_LOSS_TAG_KEYS), 'missing': missing}))
    if op == 'LOSS_PARADIGM_SHIFT':
        return (True, {})
    elif op == 'LOSS_STRUCTURE_SHIFT':
        parent = parent_irs[0] if parent_irs else None
        p_tags = _loss_operator_tags(parent) if parent is not None else {}
        if _normalize_tag_value(cand_tags.get('link')) == _normalize_tag_value(p_tags.get('link')):
            return (False, _operator_contract_failure(op_type=op, reason='link_not_changed', parent_tags=p_tags, cand_tags=cand_tags, required_change={'must_change': ['link']}))
    elif op == 'LOSS_CONSTRAINT_INJECT':
        parent = parent_irs[0] if parent_irs else None
        p_tags = _loss_operator_tags(parent) if parent is not None else {}
        if _normalize_tag_value(cand_tags.get('constraint')) == _normalize_tag_value(p_tags.get('constraint')):
            return (False, _operator_contract_failure(op_type=op, reason='constraint_not_changed', parent_tags=p_tags, cand_tags=cand_tags, required_change={'must_change': ['constraint']}))
    return (True, {})

def _validate_builder_operator_contract(ir: PreferenceBuilderIR, op_type: str | None, parent_irs: Sequence[PreferenceBuilderIR]) -> bool:
    op = str(op_type or '').upper()
    if op not in {'BUILDER_PARADIGM_SHIFT', 'BUILDER_STRUCTURE_SHIFT', 'BUILDER_CONSTRAINT_INJECT'}:
        return True
    candidate = _builder_operator_tags(ir)
    if any((_normalize_tag_value(candidate.get(key)) == 'unknown' for key in _BUILDER_TAG_KEYS)):
        return False
    if op == 'BUILDER_PARADIGM_SHIFT':
        parent = _majority_parent_tags(parent_irs, keys=_BUILDER_TAG_KEYS, kind='builder')
        return _normalize_tag_value(candidate.get('weighting')) != _normalize_tag_value(parent.get('weighting'))
    parent = _builder_operator_tags(parent_irs[0]) if parent_irs else {}
    if op == 'BUILDER_CONSTRAINT_INJECT':
        return _normalize_tag_value(candidate.get('constraint')) != _normalize_tag_value(parent.get('constraint'))
    return all((_normalize_tag_value(candidate.get(key)) == _normalize_tag_value(parent.get(key)) for key in _BUILDER_TAG_KEYS))

def validate_builder_candidate(ir: PreferenceBuilderIR, *, operator_whitelist: Sequence[str], op_type: str | None=None, parent_irs: Sequence[PreferenceBuilderIR] | None=None) -> Tuple[bool, Dict[str, Any]]:
    if not _mechanism_family_candidates(ir):
        return (False, _builder_failure_report(stage='family', reason='invalid_mechanism_families'))
    if not str(getattr(ir, 'intuition', '') or '').strip():
        return (False, _builder_failure_report(stage='interpretability', reason='missing_intuition', trace={'failed_gate': 'Interpretability', 'failure_kind': 'missing_intuition'}))
    try:
        compiled = compile_preference_builder(ir, operator_whitelist=operator_whitelist)
    except Exception as exc:
        return (False, _builder_failure_report(stage='compile', reason='compile_failed', error=str(exc)))
    try:
        fc = _dummy_feature_cache(batch_size=8, k=16, variant='visible')
        pb = compiled.build_fn(fc, {'stage': 'builder_validate', 'seed': 0})
    except Exception as exc:
        return (False, _builder_failure_report(stage='runtime', reason='runtime_failed', error=str(exc)))
    try:
        bg = run_preference_builder_gates(pb, feature_cache=fc)
    except Exception as exc:
        return (False, _builder_failure_report(stage='gate', reason='builder_gate_exception', error=str(exc)))
    if not bool(bg.ok):
        return (False, _builder_failure_report(stage='gate', reason=str(bg.reason), trace=bg.trace))
    if not _validate_builder_operator_contract(ir, op_type, list(parent_irs or [])):
        return (False, _builder_failure_report(stage='operator_contract', reason='operator_contract_failed'))
    return (True, {})

def _rank_weighted_sample_without_replacement(rng: random.Random, items: Sequence[Any], *, k: int) -> List[Any]:
    k = max(0, min(int(k), len(items)))
    if k <= 0:
        return []
    if k >= len(items):
        return list(items)
    weights = [1.0 / (index + 1.0) for index in range(len(items))]
    chosen: List[Any] = []
    pool = list(items)
    current_weights = list(weights)
    for _ in range(k):
        target = rng.random() * float(sum(current_weights))
        total = 0.0
        selected = 0
        for index, weight in enumerate(current_weights):
            total += float(weight)
            if total >= target:
                selected = index
                break
        chosen.append(pool.pop(selected))
        current_weights.pop(selected)
    return chosen

def _parent_mechanism_family(entry: Mapping[str, Any]) -> str:
    return _entry_mechanism_family(entry)

def _sample_parent_entries(rng: random.Random, entries: Sequence[Mapping[str, Any]], *, side: str, op: str, k: int) -> List[Dict[str, Any]]:
    ranked = [dict(entry) for entry in entries]
    if op != 'PARADIGM_SHIFT':
        return _rank_weighted_sample_without_replacement(rng, ranked, k=min(k, len(ranked)))
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for entry in ranked:
        groups.setdefault(_parent_mechanism_family(entry), []).append(entry)
    selected: List[Dict[str, Any]] = []
    if len(groups) >= 2:
        signatures = _rank_weighted_sample_without_replacement(rng, list(groups), k=2)
        selected.extend((groups[signature][0] for signature in signatures))
    used = {str(entry.get('id', '')) for entry in selected}
    remaining = [entry for entry in ranked if str(entry.get('id', '')) not in used]
    selected.extend(_rank_weighted_sample_without_replacement(rng, remaining, k=min(max(0, k - len(selected)), len(remaining))))
    return selected

def _propose_builders_for_generation(*, generation: int, pop_g: int, elites_g: Sequence[Mapping[str, Any]], diverse_elites_g: Sequence[Mapping[str, Any]], rng: random.Random, llm_cfg: Mapping[str, Any] | None=None, operator_whitelist: Sequence[str] | None=None, global_feedback: Mapping[str, Any] | None=None, llm_init_only: bool=True, carry_elites: bool=False) -> List[Dict[str, Any]]:
    del diverse_elites_g, llm_init_only, carry_elites
    cfg = dict((llm_cfg or {}).get('builder') or {})
    prompts = dict((llm_cfg or {}).get('prompts') or {})
    context = (llm_cfg or {}).get('builder_prompt_context')
    parent_p = max(2, int(cfg.get('parent_p', 4) or 4))
    parent_entries = [dict(entry) for entry in elites_g if isinstance(entry, Mapping) and isinstance(entry.get('ir'), Mapping)]
    parent_entries.sort(key=lambda entry: float(entry.get('fitness', float('inf'))))
    if not parent_entries:
        seeds = [_ref_builder_ir()] + _make_builtin_builder_irs(rng, parent_p)
        parent_entries = []
        for index, ir in enumerate(seeds):
            assigned = _assign_mechanism_family(ir, rng) or 'pairwise_margin'
            parent_entries.append({'id': f'bootstrap_g_{index:03d}', 'fitness': 0.0, 'ir': asdict(ir), 'mechanism_family': assigned})
    plan = _expand_operator_bank(cfg, int(generation), rng, side='builder') or ['GEN'] * int(pop_g)
    out: List[Dict[str, Any]] = []
    for requested in plan:
        if len(out) >= int(pop_g):
            break
        op = _normalize_operator_name(str(requested), side='builder')
        selected = _sample_parent_entries(rng, parent_entries, side='builder', op=op, k=parent_p)
        parents = [pref_builder_ir_from_json(entry['ir']) for entry in selected]
        fitness = [{'fitness': float(entry.get('fitness', float('inf'))), 'mechanism_family': _entry_mechanism_family(entry)} for entry in selected]
        parent_ids = [str(entry.get('id', '')) for entry in selected]
        feedback = dict(global_feedback or {})
        feedback['llm_call'] = {'side': 'builder', 'search_operator': op, 'seed': int(rng.randint(0, 2 ** 31 - 1))}
        try:
            if op == 'XOVER':
                ir, meta = builder_llm_ops.crossover_pref_builder_with_meta(prompts['builder_crossover'], parents=parents, parents_fitness=fitness, global_feedback=feedback, prompt_context=context)
                op_type = 'E1'
            elif op == 'TUNE':
                ir, meta = builder_llm_ops.m2_tune_builder_with_meta(prompts['builder_m2'], parent=parents[0], parent_fitness=fitness[0], global_feedback=feedback, prompt_context=context)
                op_type = 'M2'
            elif op == 'PARADIGM_SHIFT':
                ir, meta = builder_llm_ops.paradigm_shift_builder_with_meta(prompts['builder_paradigm_shift'], parents=parents, parents_fitness=fitness, global_feedback=feedback, prompt_context=context)
                op_type = 'BUILDER_PARADIGM_SHIFT'
            elif op == 'STRUCTURE_SHIFT':
                ir, meta = builder_llm_ops.structure_shift_builder_with_meta(prompts['builder_structure_shift'], parent=parents[0], parent_fitness=fitness[0], global_feedback=feedback, prompt_context=context)
                op_type = 'BUILDER_STRUCTURE_SHIFT'
            elif op == 'CONSTRAINT_INJECT':
                ir, meta = builder_llm_ops.constraint_inject_builder_with_meta(prompts['builder_constraint_inject'], parent=parents[0], parent_fitness=fitness[0], global_feedback=feedback, prompt_context=context)
                op_type = 'BUILDER_CONSTRAINT_INJECT'
            else:
                ir, meta = builder_llm_ops.generate_pref_builder_candidate_with_meta(prompts['builder_generation'], operator_whitelist=operator_whitelist or [], global_feedback=feedback, prompt_context=context)
                op_type = 'E1_GENERATE'
                parent_ids = []
            ok, _ = validate_builder_candidate(ir, operator_whitelist=operator_whitelist or [], op_type=op_type, parent_irs=parents)
            assigned = _proposal_mechanism_family(ir, rng, op=op, parents=selected) if ok else None
            if ok and assigned is not None:
                out.append({'ir': ir, 'mechanism_family': assigned, 'origin': op, 'origin_base': op, 'op_type': op_type, 'parents': parent_ids, 'attempt': 0, 'prompt_sha1': meta.get('prompt_sha1'), 'prompt_path': meta.get('prompt_path'), 'history': [dict(meta)], 'llm_seed': feedback['llm_call']['seed']})
        except Exception as exc:
            LOGGER.warning('Builder proposal failed gen=%d op=%s: %s', int(generation), op, str(exc))
    if not out:
        raise RuntimeError(f'Builder generation {int(generation)} produced no valid candidates.')
    return out[:int(pop_g)]

def _propose_losses_for_generation(*, generation: int, pop_f: int, elites_f: Sequence[Mapping[str, Any]], diverse_elites_f: Sequence[Mapping[str, Any]], rng: random.Random, llm_cfg: Mapping[str, Any] | None=None, operator_whitelist: Sequence[str] | None=None, global_feedback: Mapping[str, Any] | None=None, llm_init_only: bool=True, carry_elites: bool=False) -> List[Dict[str, Any]]:
    del diverse_elites_f, llm_init_only, carry_elites
    cfg = dict((llm_cfg or {}).get('loss') or {})
    prompts = dict((llm_cfg or {}).get('prompts') or {})
    context = (llm_cfg or {}).get('loss_prompt_context')
    parent_p = max(2, int(cfg.get('parent_p', 4) or 4))
    parent_entries = [dict(entry) for entry in elites_f if isinstance(entry, Mapping) and isinstance(entry.get('ir'), Mapping)]
    parent_entries.sort(key=lambda entry: float(entry.get('fitness', float('inf'))))
    if not parent_entries:
        seeds = [_ref_loss_ir()] + _make_builtin_loss_irs(rng, parent_p)
        parent_entries = []
        for index, ir in enumerate(seeds):
            assigned = _assign_mechanism_family(ir, rng) or 'pairwise_margin'
            parent_entries.append({'id': f'bootstrap_f_{index:03d}', 'fitness': 0.0, 'ir': asdict(ir), 'mechanism_family': assigned})
    plan = _expand_operator_bank(cfg, int(generation), rng, side='loss') or ['GEN'] * int(pop_f)
    out: List[Dict[str, Any]] = []
    for requested in plan:
        if len(out) >= int(pop_f):
            break
        op = _normalize_operator_name(str(requested), side='loss')
        selected = _sample_parent_entries(rng, parent_entries, side='loss', op=op, k=parent_p)
        parents = [free_loss_ir_from_json(entry['ir']) for entry in selected]
        fitness = [{'fitness': float(entry.get('fitness', float('inf'))), 'mechanism_family': _entry_mechanism_family(entry)} for entry in selected]
        parent_ids = [str(entry.get('id', '')) for entry in selected]
        feedback = dict(global_feedback or {})
        feedback['llm_call'] = {'side': 'loss', 'search_operator': op, 'seed': int(rng.randint(0, 2 ** 31 - 1))}
        try:
            if op == 'XOVER':
                ir = loss_llm_ops.crossover_free_loss(prompts['loss_crossover'], parents=parents, parents_fitness=fitness, global_feedback=feedback, prompt_context=context)
                op_type = 'E1'
            elif op == 'TUNE':
                ir = loss_llm_ops.m2_tune_hparams(prompts['loss_m2'], parent=parents[0], parent_fitness=fitness[0], global_feedback=feedback, prompt_context=context)
                op_type = 'M2'
            elif op == 'PARADIGM_SHIFT':
                ir = loss_llm_ops.paradigm_shift_free_loss(prompts['loss_paradigm_shift'], parents=parents, parents_fitness=fitness, global_feedback=feedback, prompt_context=context)
                op_type = 'LOSS_PARADIGM_SHIFT'
            elif op == 'STRUCTURE_SHIFT':
                ir = loss_llm_ops.structure_shift_free_loss(prompts['loss_structure_shift'], parent=parents[0], parent_fitness=fitness[0], global_feedback=feedback, prompt_context=context)
                op_type = 'LOSS_STRUCTURE_SHIFT'
            elif op == 'CONSTRAINT_INJECT':
                ir = loss_llm_ops.constraint_inject_free_loss(prompts['loss_constraint_inject'], parent=parents[0], parent_fitness=fitness[0], global_feedback=feedback, prompt_context=context)
                op_type = 'LOSS_CONSTRAINT_INJECT'
            else:
                ir = loss_llm_ops.generate_free_loss_candidate(prompts['loss_generation'], operator_whitelist=operator_whitelist or [], global_feedback=feedback, prompt_context=context)
                op_type = 'E1_GENERATE'
                parent_ids = []
            static = run_static_gates(ir, operator_whitelist=operator_whitelist or [])
            if not static.ok:
                continue
            compile_free_loss(ir, operator_whitelist=operator_whitelist or [])
            contract_ok, _ = _validate_loss_operator_contract(ir, op_type, parents)
            assigned = _proposal_mechanism_family(ir, rng, op=op, parents=selected) if contract_ok else None
            if contract_ok and assigned is not None:
                out.append({'ir': ir, 'mechanism_family': assigned, 'origin': op, 'origin_base': op, 'op_type': op_type, 'parents': parent_ids, 'attempt': 0, 'prompt_sha1': None, 'prompt_path': prompts.get('loss_' + op.lower()), 'history': [], 'llm_seed': feedback['llm_call']['seed']})
        except Exception as exc:
            LOGGER.warning('Loss proposal failed gen=%d op=%s: %s', int(generation), op, str(exc))
    if not out:
        raise RuntimeError(f'Loss generation {int(generation)} produced no valid candidates.')
    return out[:int(pop_f)]

def _trimmed_mean(values: Sequence[float], *, trim: float) -> float:
    xs = [float(v) for v in values if v == v and v not in (float('inf'), float('-inf'))]
    if not xs:
        return float('inf')
    xs.sort()
    t = float(trim)
    if not 0.0 <= t < 0.5:
        t = 0.0
    k = int(t * len(xs))
    core = xs[k:len(xs) - k] if len(xs) - 2 * k > 0 else xs
    return float(sum(core) / len(core)) if core else float('inf')

def _best_k_mean(values: Sequence[float], *, k: int) -> float:
    xs = [float(v) for v in values if v == v and v not in (float('inf'), float('-inf'))]
    if not xs:
        return float('inf')
    kk = max(1, min(int(k), len(xs)))
    xs.sort()
    best = xs[:kk]
    return float(sum(best) / len(best))

def _std(values: Sequence[float]) -> float:
    xs = [float(v) for v in values if v == v and v not in (float('inf'), float('-inf'))]
    if len(xs) <= 1:
        return 0.0
    m = float(sum(xs) / len(xs))
    v = float(sum(((x - m) ** 2 for x in xs)) / (len(xs) - 1))
    return float(v ** 0.5)

def _credit_assignment_v2(*, pair_records: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, float], Dict[str, float]]:

    def _score(rec: Mapping[str, Any]) -> float:
        v = rec.get('score')
        if v is None and isinstance(rec.get('fitness'), dict):
            v = rec['fitness'].get('fitness_score', rec['fitness'].get('validation_objective'))
        try:
            return float(v)
        except (TypeError, ValueError):
            return float('inf')
    scores_by_g: Dict[str, List[float]] = {}
    scores_by_f: Dict[str, List[float]] = {}
    for rec in pair_records:
        g_id = str(rec.get('g_id'))
        f_id = str(rec.get('f_id'))
        s = _score(rec)
        scores_by_g.setdefault(g_id, []).append(s)
        scores_by_f.setdefault(f_id, []).append(s)

    def _fitness(scores: Sequence[float]) -> float:
        n = len(scores)
        if n <= 0:
            return float('inf')
        tm = _trimmed_mean(scores, trim=0.2)
        bk = _best_k_mean(scores, k=max(1, n // 3))
        se = _std(scores) / float(max(n, 1) ** 0.5)
        return float(0.6 * tm + 0.4 * bk + 0.2 * se)
    return ({k: _fitness(v) for k, v in scores_by_g.items()}, {k: _fitness(v) for k, v in scores_by_f.items()})
G_REF_ID = 'g_ref'
F_REF_ID = 'f_ref'

def _ref_builder_ir() -> PreferenceBuilderIR:
    code = "def generated_builder(feature_cache, extra):\n    objective = feature_cache['objective']\n    mask = objective[:, :, None] < objective[:, None, :]\n    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)\n    return PrefBatch(mode='pairwise', pair_idx=(b_idx, winner_idx, loser_idx), weight=None, meta={'builder': 'ref_all_pairs'})\n"
    return PreferenceBuilderIR(name='ref_all_pairs_builder', intuition='Reference builder: all winner/loser pairs by objective ordering.', implementation_hint=PreferenceBuilderImplementationHint(expects=['objective', 'log_prob'], returns='PrefBatch', mode='pairwise'), hyperparams={'mechanism_families': ['cost_calibrated', 'rank_based', 'pairwise_margin'], 'geometry': 'dense_all_pairs', 'cap': 'uncapped_full', 'weighting': 'uniform_none', 'constraint': 'none'}, operators_used=['ref_all_pairs'], code=code)

def _ref_loss_ir() -> FreeLossIR:
    code = "def generated_loss(batch, model_output, extra):\n    alpha = float(extra.get('alpha', 1.0))\n    delta_p = batch['log_prob_w'] - batch['log_prob_l']\n    quality = 1.0 + 0.1 * ops.tanh(batch['delta_z'])\n    return -ops.logsigmoid(alpha * quality * delta_p)\n"
    return FreeLossIR(name='ref_logsigmoid', intuition='Reference pairwise loss using a normalized objective signal.', pseudocode='elementwise pairwise margin with normalized quality modulation', hyperparams={'alpha': 1.0, 'mechanism_families': ['cost_calibrated', 'pairwise_margin'], 'signal': 'standardized_gap', 'link': 'logsigmoid', 'aggregation': 'framework_per_instance_weighted_mean', 'constraint': 'elementwise_pair'}, operators_used=['logsigmoid', 'tanh'], implementation_hint=FreeLossImplementationHint(expects=['delta_z', 'log_prob_w', 'log_prob_l'], returns='per_pair', mode='pairwise'), code=code, theoretical_basis='')

def _ensure_reference_compiled(*, compiled_g: Dict[str, CompiledPreferenceBuilder], compiled_f: Dict[str, CompiledFreeLoss], operator_whitelist: Sequence[str]) -> None:
    if G_REF_ID not in compiled_g:
        compiled_g[G_REF_ID] = compile_preference_builder(_ref_builder_ir(), operator_whitelist=operator_whitelist)
    if F_REF_ID not in compiled_f:
        ref_ir = _ref_loss_ir()
        static_ref = run_static_gates(ref_ir, operator_whitelist=operator_whitelist)
        if not static_ref.ok:
            raise RuntimeError(f'Reference loss failed static gates: {static_ref.reason}')
        compiled_f[F_REF_ID] = compile_free_loss(ref_ir, operator_whitelist=operator_whitelist)

def _evaluate_pair_worker(payload: Mapping[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    generation = int(payload['generation'])
    pair_index = int(payload['pair_index'])
    g_entry = dict(payload['g_entry'])
    f_entry = dict(payload['f_entry'])
    cfg = dict(payload['cfg_yaml'])
    physical_device_str, device_str = _worker_device_fields(payload)
    run_dir = payload.get('run_dir')
    run_dir_s = str(run_dir) if isinstance(run_dir, (str, os.PathLike)) and run_dir else None
    operator_whitelist = list(payload.get('operator_whitelist', []))
    loss_prompt_context_raw = cfg.get('loss_prompt_context')
    if isinstance(loss_prompt_context_raw, Mapping):
        loss_prompt_context = dict(loss_prompt_context_raw)
    else:
        loss_prompt_context = dict(loss_llm_ops.build_runtime_prompt_context())
    g_ir = pref_builder_ir_from_json(g_entry['ir'])
    f_ir = free_loss_ir_from_json(f_entry['ir'])
    record: Dict[str, Any] = {'generation': generation, 'pair_index': pair_index, 'g_id': str(g_entry['id']), 'f_id': str(f_entry['id']), 'g_ir': dict(g_entry['ir']), 'f_ir': dict(f_entry['ir']), 'device': physical_device_str, 'device_str': physical_device_str, 'device_physical_str': physical_device_str, 'device_logical_str': device_str, 'score': None}
    try:
        compiled_g = compile_preference_builder(g_ir, operator_whitelist=operator_whitelist)
        record['g_compile_ok'] = True
        record['g_compile_reason'] = 'ok'
    except PreferenceBuilderCompileError as exc:
        record['pair_ok'] = False
        record['pair_reason'] = 'g_compile_failed'
        record['g_compile_ok'] = False
        record['g_compile_reason'] = str(exc)
        record['score'] = float('inf')
        record['elapsed_s'] = float(time.time() - t0)
        return record
    try:
        static_res: StaticGateResult = run_static_gates(f_ir, operator_whitelist=operator_whitelist)
        record['f_static_ok'] = bool(static_res.ok)
        record['f_static_reason'] = str(static_res.reason)
        if not static_res.ok:
            raise CompileError(f'static_gate_failed: {static_res.reason}')
        compiled_f = compile_free_loss(f_ir, operator_whitelist=operator_whitelist)
        record['f_compile_ok'] = True
        record['f_compile_reason'] = 'ok'
    except Exception as exc:
        record['pair_ok'] = False
        record['pair_reason'] = 'f_compile_failed'
        record['f_compile_ok'] = False
        record['f_compile_reason'] = str(exc)
        record['score'] = float('inf')
        record['elapsed_s'] = float(time.time() - t0)
        return record
    variant = 'visible'
    feature_cache = _dummy_feature_cache(batch_size=8, k=16, variant='visible')
    pref_batch = compiled_g.build_fn(feature_cache, {'stage': 'stage0_gate'})
    builder_gate = run_preference_builder_gates(pref_batch, feature_cache=feature_cache)
    joint_gate = run_joint_preference_gates(compiled_f, pref_batch=pref_batch, feature_cache=feature_cache, min_pass_rate=0.8, swap_tolerance=0.001, swap_check_mode='data', swap_test_margin=1.0, grad_eps=1e-08, min_effective_grad_ratio=0.1, numeric_stress_enabled=True, numeric_stress_margin=120.0, numeric_stress_aux_scale=32.0, variant='visible')
    sandbox_gate_result = _run_stage0_sandbox_gate(run_dir=str(run_dir_s), generation=int(generation), pair_index=int(pair_index), g_id=str(record.get('g_id', '')), f_id=str(record.get('f_id', '')), g_ir=g_ir, f_ir=f_ir, operator_whitelist=list(operator_whitelist), cfg_yaml=cfg)
    record['sandbox_gate'] = dict(sandbox_gate_result)
    record['sandbox_gate_ok'] = bool(sandbox_gate_result.get('ok'))
    record['sandbox_gate_reason'] = str(sandbox_gate_result.get('reason', ''))
    record.update({'builder_gate_ok': bool(builder_gate.ok), 'builder_gate_reason': str(builder_gate.reason), 'builder_gate_trace': builder_gate.trace, 'joint_gate_ok': bool(joint_gate.ok), 'joint_gate_reason': str(joint_gate.reason), 'joint_gate_trace': joint_gate.trace})
    if not bool(sandbox_gate_result.get('ok')):
        record['pair_ok'] = False
        record['pair_reason'] = 'stage0_sandbox_failed'
        record['score'] = float('inf')
        record['elapsed_s'] = float(time.time() - t0)
        return record
    if not builder_gate.ok or not joint_gate.ok:
        record['pair_ok'] = False
        record['pair_reason'] = 'stage0_gate_failed'
        record['score'] = float('inf')
        record['elapsed_s'] = float(time.time() - t0)
        return record
    record.update(_run_loss_invariance_gate(compiled_f, cfg))
    if not bool(record.get('affine_gate_ok', False)):
        record['pair_ok'] = False
        record['pair_reason'] = 'affine_gate_failed'
        record['score'] = float('inf')
        record['elapsed_s'] = float(time.time() - t0)
        return record
    try:
        file_handler: logging.Handler | None = None
        fl_logger = logging.getLogger('fitness.free_loss_fidelity')
        if run_dir_s:
            safe_gid = str(record.get('g_id', 'g')).replace(os.sep, '_').replace(':', '_')[:24]
            safe_fid = str(record.get('f_id', 'f')).replace(os.sep, '_').replace(':', '_')[:24]
            safe_dev = str(physical_device_str or device_str).replace(os.sep, '_').replace(':', '_')
            log_path = os.path.join(run_dir_s, f'gen{generation:03d}_pair{pair_index:03d}_{safe_dev}_{safe_gid}_{safe_fid}.log')
            fmt = logging.Formatter('[%(asctime)s] %(levelname)s:%(name)s: %(message)s')
            root_logger = logging.getLogger()
            for handler in list(root_logger.handlers):
                root_logger.removeHandler(handler)
            root_logger.setLevel(logging.INFO)
            for handler in list(fl_logger.handlers):
                try:
                    fl_logger.removeHandler(handler)
                finally:
                    try:
                        handler.close()
                    except Exception:
                        pass
            fl_logger.setLevel(logging.INFO)
            try:
                file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
                file_handler.setFormatter(fmt)
                fl_logger.propagate = True
                root_logger.addHandler(file_handler)
                record['hf_log_file'] = os.path.basename(log_path)
            except Exception as exc:
                print(f'[two_stage_search][worker] failed to open high_fidelity log file: {log_path}: {exc}', flush=True)
                file_handler = None
        adapter = _CompiledBuilderAdapter(compiled_g)
        eval_cfg = dict(cfg)
        baseline_cfg = eval_cfg.get('baseline', {}) or {}
        mini_eval_path = _resolve_high_fidelity_baseline_mini_eval_path(eval_cfg, baseline_cfg)
        if not mini_eval_path:
            raise ValueError(f'baseline mini_eval_path is required for high_fidelity (fidelity={_high_fidelity_fidelity_key(eval_cfg)})')
        baseline_payload = _load_baseline_mini_eval(str(mini_eval_path))
        expected_sig = _build_high_fidelity_eval_signature(eval_cfg)
        got_sig = baseline_payload.get('eval_signature')
        if got_sig != expected_sig:
            raise ValueError(f'baseline eval_signature mismatch; regenerate the baseline cache')
        per_init_base = baseline_payload.get('per_init')
        if not isinstance(per_init_base, dict):
            raise ValueError(f'baseline JSON missing per_init dict')
        init_specs = _high_fidelity_init_specs_from_baseline_cfg(eval_cfg)
        if not init_specs:
            raise ValueError(f'has no init sources configured')
        scratch_init_seed = int(_resolve_training_seed(eval_cfg))
        hf_epochs_cfg = int(eval_cfg.get('hf_epochs', 0) or 0)
        hf_inst_cfg = int(eval_cfg.get('hf_instances_per_epoch', 0) or 0)
        K = int(eval_cfg.get('f1_steps', 32) or 32)
        cfg_hf = dict(eval_cfg)
        cfg_hf['f1_steps'] = int(K)
        if not (hf_epochs_cfg > 0 and hf_inst_cfg > 0):
            cfg_hf['hf_epochs'] = 0
            cfg_hf['hf_instances_per_epoch'] = 0
        hf_cfg = _build_hf_cfg(cfg_hf, seed=int(scratch_init_seed), device_str=device_str)
        valid_sizes = [int(v) for v in cfg_hf.get('valid_problem_sizes', list(hf_cfg.valid_problem_sizes))]
        if not valid_sizes:
            valid_sizes = [int(hf_cfg.train_problem_size)]
        valid_sizes = list(dict.fromkeys([int(v) for v in valid_sizes]))
        all_per_init: Dict[str, Any] = {}
        all_deltas: List[float] = []
        any_error = False
        fl_logger.info('High-fidelity evaluation gen=%d pair_index=%d K=%d seed=%d valid_sizes=%s baseline_json=%s', int(generation), int(pair_index), int(K), int(scratch_init_seed), list(valid_sizes), str(mini_eval_path))
        for init_name, init_ckpt in init_specs:
            base_entry = per_init_base.get(str(init_name))
            base_source = 'mini_eval'
            if not isinstance(base_entry, dict):
                raise ValueError(f'baseline JSON missing per_init[{init_name}]')
            try:
                base_agg = float(base_entry.get('aggregated_objective'))
            except (TypeError, ValueError):
                raise ValueError(f'baseline per_init[{init_name}].aggregated_objective is invalid')
            base_by_size_raw = base_entry.get('val_objective_by_size', {})
            base_by_size: Dict[int, float] = {}
            if isinstance(base_by_size_raw, dict):
                for k, v in base_by_size_raw.items():
                    try:
                        base_by_size[int(k)] = float(v)
                    except Exception:
                        continue
            cand_by_size: Dict[int, float] = {}
            cand_agg: float
            error: str | None = None
            error_traceback: str | None = None
            preflight_rollout_smoke_test: Dict[str, Any] | None = None
            init_ckpt_abs = _abs_from_repo_root(str(init_ckpt)) if init_ckpt else None
            try:
                try:
                    preflight_rollout_smoke_test = run_rl4co_rollout_smoke_test(hf_cfg, init_checkpoint_path=init_ckpt_abs, phase='train', device=device_str, batch_size=1, num_rollouts=1)
                except Exception as smoke_exc:
                    preflight_rollout_smoke_test = {'ok': False, 'error': f'{type(smoke_exc).__name__}: {smoke_exc}', 'error_traceback': traceback.format_exc()}
                if not bool((preflight_rollout_smoke_test or {}).get('ok', False)):
                    raise RuntimeError('Preflight rollout smoke test failed')
                free_cfg = FreeLossFidelityConfig(hf=hf_cfg, init_checkpoint_path=init_ckpt_abs, scratch_hf_epochs=int(eval_cfg.get('scratch_hf_epochs', 0) or 0), warmstart_hf_epochs=int(eval_cfg.get('warmstart_hf_epochs', 0) or 0))
                fitness = evaluate_free_loss_candidate(compiled_f, free_cfg, pref_builder=adapter)
                size_objectives_raw = fitness.get('size_objectives', {})
                size_objectives: Dict[int, float] = {}
                if isinstance(size_objectives_raw, dict):
                    for k, v in size_objectives_raw.items():
                        try:
                            size_objectives[int(k)] = float(v)
                        except Exception:
                            continue
                for sz in valid_sizes:
                    if int(sz) not in size_objectives:
                        raise RuntimeError(f'Missing fitness.size_objectives[{int(sz)}]')
                    cand_by_size[int(sz)] = float(size_objectives[int(sz)])
                cand_agg = float(sum((float(cand_by_size[int(sz)]) for sz in valid_sizes)) / float(len(valid_sizes)))
                if not math.isfinite(cand_agg):
                    raise RuntimeError('Non-finite candidate aggregated objective')
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
                error_traceback = traceback.format_exc()
                try:
                    fl_logger.exception('High-fidelity mini-train failed init=%s g_id=%s f_id=%s device=%s logical_device=%s', str(init_name), str(record.get('g_id')), str(record.get('f_id')), str(physical_device_str or device_str), str(device_str))
                except Exception:
                    pass
                cand_by_size = {int(sz): 1000000000.0 for sz in valid_sizes}
                cand_agg = 1000000000.0
            delta = float(cand_agg) - float(base_agg)
            all_deltas.append(float(delta))
            any_error = any_error or bool(error)
            init_record = {'obj_cand_by_size': {str(int(k)): float(v) for k, v in cand_by_size.items()}, 'obj_base_by_size': {str(int(k)): float(base_by_size.get(int(k), float('nan'))) for k in valid_sizes}, 'obj_cand': float(cand_agg), 'obj_base': float(base_agg), 'delta': float(delta), 'baseline_source': str(base_source), 'baseline_seed': None, 'init_checkpoint': str(init_ckpt) if init_ckpt else None, 'error': error, 'error_traceback': error_traceback, 'preflight_rollout_smoke_test': preflight_rollout_smoke_test}
            all_per_init[str(init_name)] = dict(init_record)
        delta_mean = float(sum(all_deltas) / float(len(all_deltas))) if all_deltas else float('inf')
        delta_worst = float(max(all_deltas)) if all_deltas else float('inf')
        runtime_error_inits = [str(init_name) for init_name, init_record in all_per_init.items() if isinstance(init_record, Mapping) and str(init_record.get('error') or '').strip()]
        record['pair_ok'] = not any_error
        record['pair_reason'] = 'ok_high_fidelity_minitrain' if not any_error else 'high_fidelity_runtime_error'
        record['fitness'] = {'per_init': all_per_init, 'delta_mean': float(delta_mean), 'delta_worst': float(delta_worst), 'baseline_mini_eval_path': str(mini_eval_path), 'eval_signature': expected_sig, 'any_error': bool(any_error), 'runtime_error_count': int(len(runtime_error_inits)), 'runtime_error_inits': list(runtime_error_inits)}
        record['score'] = float('inf') if any_error else float(delta_mean)
        record['better_than_baseline_mean'] = bool(delta_mean < 0.0)
        record['better_than_baseline_strict'] = bool(all_deltas and all((float(d) < 0.0 for d in all_deltas)))
        record['high_fidelity_any_error'] = bool(any_error)
        record['high_fidelity_runtime_error_count'] = int(len(runtime_error_inits))
        record['high_fidelity_runtime_error_inits'] = list(runtime_error_inits)
        fl_logger.info('High-fidelity mini-train done gen=%d pair_index=%d score=%s any_error=%s', int(generation), int(pair_index), str(record.get('score')), str(any_error))
        record['elapsed_s'] = float(time.time() - t0)
        return record
    except Exception as exc:
        record['pair_ok'] = False
        record['pair_reason'] = 'high_fidelity_fatal'
        record['fatal_error'] = str(exc)
        record['score'] = float('inf')
        record['elapsed_s'] = float(time.time() - t0)
        return record
    finally:
        if 'file_handler' in locals() and file_handler is not None:
            try:
                logging.getLogger('fitness.free_loss_fidelity').removeHandler(file_handler)
            except Exception:
                pass
            try:
                logging.getLogger().removeHandler(file_handler)
            except Exception:
                pass
            try:
                file_handler.close()
            except Exception:
                pass

def run_search_stage(config_path: str, *, search_side: str, fixed_loss_path: str | None=None) -> None:
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg_yaml = yaml.safe_load(f) or {}
    if not isinstance(cfg_yaml, dict):
        raise ValueError(f'Invalid YAML config: {config_path}')
    seed = int(cfg_yaml.get('seed', 0))
    _set_seed(seed)
    rng = random.Random(seed)
    generations = int(cfg_yaml.get('generations', 1))
    pop_g = int(cfg_yaml.get('pop_g', 8))
    pop_f = int(cfg_yaml.get('pop_f', 8))
    elite_g = int(cfg_yaml.get('elite_g', 4))
    elite_f = int(cfg_yaml.get('elite_f', 4))
    pairing_budget = int(cfg_yaml.get('pairing_budget_per_gen', 16))
    search_mode = 'loss_only' if search_side == 'loss' else 'builder_only'
    metric_mode = 'minimize'
    improve_eps = 0.0
    operator_whitelist = list(cfg_yaml.get('operator_whitelist', []))
    devices = cfg_yaml.get('devices')
    if isinstance(devices, (list, tuple)) and devices:
        device_list = [_normalize_device_alias(str(d)) for d in devices]
    else:
        device_list = [_normalize_device_alias(str(cfg_yaml.get('device', 'cuda')))]
    out_root = str(cfg_yaml.get('output_root', 'runs/two_stage_search'))
    os.makedirs(out_root, exist_ok=True)
    run_dir = _timestamp_dir(out_root)
    LOGGER.info('Run directory: %s', os.path.abspath(run_dir))
    prepared = _ensure_high_fidelity_baseline_mini_eval(cfg_yaml=cfg_yaml, operator_whitelist=operator_whitelist, device_str=str(device_list[0] if device_list else 'cuda'))
    LOGGER.info('High-fidelity baseline prepared path=%s cached=%s regenerated=%s', str(prepared.get('path')), str(bool(prepared.get('cached', False))), str(bool(prepared.get('regenerated', False))))
    if search_mode == 'loss_only':
        LOGGER.info('Stage 1: builder population is frozen; only pairwise losses are searched.')
    else:
        LOGGER.info('Stage 2: loss is frozen; only set-aware pair weighting is searched.')
    builder_llm_raw = cfg_yaml.get('builder_llm', {}) or {}
    loss_llm_raw = cfg_yaml.get('loss_llm', {}) or {}
    if not isinstance(builder_llm_raw, dict):
        builder_llm_raw = {}
    if not isinstance(loss_llm_raw, dict):
        loss_llm_raw = {}
    llm_prompts_raw = cfg_yaml.get('llm_prompts', {}) or {}
    if not isinstance(llm_prompts_raw, dict):
        llm_prompts_raw = {}
    loss_prompt_context = loss_llm_ops.build_runtime_prompt_context()
    builder_prompt_context = builder_llm_ops.build_runtime_prompt_context()
    llm_prompts_defaults = {'builder_generation': 'PTP/prompts/pref_builder_generation.txt', 'builder_crossover': 'PTP/prompts/pref_builder_crossover.txt', 'builder_paradigm_shift': 'PTP/prompts/pref_builder_paradigm_shift.txt', 'builder_structure_shift': 'PTP/prompts/pref_builder_structure_shift.txt', 'builder_constraint_inject': 'PTP/prompts/pref_builder_constraint_inject.txt', 'builder_m2': 'PTP/prompts/pref_builder_m2.txt', 'loss_generation': 'PTP/prompts/free_loss_generation.txt', 'loss_crossover': 'PTP/prompts/free_loss_crossover.txt', 'loss_paradigm_shift': 'PTP/prompts/free_loss_paradigm_shift.txt', 'loss_structure_shift': 'PTP/prompts/free_loss_structure_shift.txt', 'loss_constraint_inject': 'PTP/prompts/free_loss_constraint_inject.txt', 'loss_m2': 'PTP/prompts/free_loss_m2.txt'}
    llm_prompts: Dict[str, str] = {}
    for k, v in llm_prompts_defaults.items():
        val = llm_prompts_raw.get(k, v)
        llm_prompts[k] = _abs_from_repo_root(str(val))
    builder_cfg: Dict[str, Any] = {'parent_p': int(builder_llm_raw.get('parent_p', 4) or 4), 'operator_bank': dict(builder_llm_raw.get('operator_bank') or {})}
    loss_cfg: Dict[str, Any] = {'parent_p': int(loss_llm_raw.get('parent_p', 4) or 4), 'operator_bank': dict(loss_llm_raw.get('operator_bank') or {})}
    llm_cfg: Dict[str, Any] = {'prompts': dict(llm_prompts), 'builder_prompt_context': dict(builder_prompt_context), 'loss_prompt_context': dict(loss_prompt_context), 'builder': builder_cfg, 'loss': loss_cfg}
    loss_llm_ops.configure_llm_run(run_dir=run_dir)
    builder_llm_ops.configure_llm_run(run_dir=run_dir)
    try:
        LOGGER.info('LLM cache stats: %s', dict(loss_llm_ops.llm_cache_stats()))
    except Exception:
        pass
    builders_jsonl = os.path.join(run_dir, 'builders.jsonl')
    losses_jsonl = os.path.join(run_dir, 'losses.jsonl')
    pairs_jsonl = os.path.join(run_dir, 'pairs.jsonl')
    gate_jsonl = os.path.join(run_dir, 'gate_reports.jsonl')
    summary_json = os.path.join(run_dir, 'summary.json')
    eval_protocol_json = os.path.join(run_dir, 'eval_protocol.json')
    for path in (builders_jsonl, losses_jsonl, pairs_jsonl, gate_jsonl):
        with open(path, 'w', encoding='utf-8'):
            pass
    gen_start = 0
    seen_g: set[str] = set()
    seen_f: set[str] = set()
    resident_pop_g: List[Dict[str, Any]] = []
    resident_pop_f: List[Dict[str, Any]] = []
    elites_g: List[Dict[str, Any]] = []
    elites_f: List[Dict[str, Any]] = []
    imported_loss_baseline_entry: Dict[str, Any] | None = None
    if fixed_loss_path is not None:
        imported_loss = _load_stage1_loss_entry(os.path.abspath(fixed_loss_path))
        resident_pop_f = [dict(imported_loss)]
        elites_f = [dict(imported_loss)]
        imported_loss_baseline_entry = dict(imported_loss)
        seen_f.add(str(imported_loss['signature']))
        LOGGER.info('Initialized Stage 2 with Stage 1 loss: %s', os.path.abspath(fixed_loss_path))
    best_so_far: Dict[str, Any] | None = None
    pair_score_history_map: Dict[str, List[Dict[str, Any]]] = {}
    _atomic_write_json(eval_protocol_json, {'protocol': 'stage0_high_fidelity', 'search_mode': str(search_mode), 'generations': int(generations), 'pairing_budget_per_gen': int(pairing_budget), 'hf_epochs': int(cfg_yaml.get('hf_epochs', 0) or 0)})
    _atomic_write_json(summary_json, _summary_state(gen_start - 1))
    llm_feedback_state: Dict[str, Any] = {}
    phase_block_label: str | None = None
    phase_block_best_score: float | None = None
    last_phase_block_label: str | None = None
    last_phase_block_best_score: float | None = None
    if isinstance(best_so_far, dict):
        try:
            last_phase_block_best_score = float(best_so_far.get('score'))
        except (TypeError, ValueError):
            last_phase_block_best_score = None
        last_phase_block_label = str(best_so_far.get('phase') or 'baseline')
    baseline_incumbent_calibrated = False
    for gen in range(gen_start, generations):
        stage_phase = 'loss' if search_mode == 'loss_only' else 'builder'
        stage_loss_budget = int(pairing_budget) if stage_phase == 'loss' else 0
        stage_builder_budget = int(pairing_budget) if stage_phase == 'builder' else 0
        generation_phase_label = str(stage_phase)
        if phase_block_label is None:
            phase_block_label = str(generation_phase_label)
        llm_cfg_for_gen = llm_cfg
        LOGGER.info('=== %s generation %d/%d ===', str(search_mode), gen, generations - 1)
        if isinstance(best_so_far, dict):
            LOGGER.info('INCUMBENT best_score=%s best_pair=(%s,%s) best_stage=%s metric_mode=%s improve_eps=%s', best_so_far.get('score'), best_so_far.get('builder_id'), best_so_far.get('loss_id'), best_so_far.get('stage_final'), str(metric_mode), float(improve_eps))
        else:
            LOGGER.info('INCUMBENT best_score=None best_pair=(None,None) best_stage=none metric_mode=%s improve_eps=%s', str(metric_mode), float(improve_eps))
        LOGGER.info('Generation %d started', int(gen))
        best_builder_ir = dict(elites_g[0].get('ir')) if elites_g else None
        if isinstance(best_builder_ir, dict) and isinstance(best_builder_ir.get('code'), str):
            best_builder_ir['code'] = _truncate_code(best_builder_ir.get('code', ''))
        best_loss_ir = dict(elites_f[0].get('ir')) if elites_f else None
        if isinstance(best_loss_ir, dict) and isinstance(best_loss_ir.get('code'), str):
            best_loss_ir['code'] = _truncate_code(best_loss_ir.get('code', ''))
        best_builder_summary = summarize_best_builder(elites_g[0]) if elites_g else None
        best_loss_summary = summarize_best_loss(elites_f[0]) if elites_f else None
        try:
            b_len = len(str((best_builder_summary or {}).get('code', '')))
            f_len = len(str((best_loss_summary or {}).get('code', '')))
        except Exception:
            b_len = -1
            f_len = -1
        LOGGER.info('Gen %d prompt context sizes: best_builder_summary.code=%d chars best_loss_summary.code=%d chars', int(gen), int(b_len), int(f_len))
        global_feedback: Dict[str, Any] = dict(llm_feedback_state)
        global_feedback.update({'generation': int(gen), 'objective': 'lower_is_better', 'pairing_budget_per_gen': int(pairing_budget), 'generations': int(generations), 'operator_whitelist': list(operator_whitelist), 'best_builder': best_builder_ir, 'best_loss': best_loss_ir, 'best_builder_summary': best_builder_summary, 'best_loss_summary': best_loss_summary})
        builder_population_active = stage_phase == 'builder'
        loss_population_active = stage_phase == 'loss'
        builder_offspring_target = 0
        loss_offspring_target = 0
        if builder_population_active:
            builder_offspring_target = max(1, int(pop_g))
        if loss_population_active:
            loss_offspring_target = max(1, int(pop_f))
        llm_init_only = True
        proposed_g: List[Dict[str, Any]] = []
        proposed_f: List[Dict[str, Any]] = []
        if builder_population_active:
            proposed_g = _propose_builders_for_generation(generation=int(gen), pop_g=int(max(builder_offspring_target, 1)), elites_g=resident_pop_g, diverse_elites_g=[], rng=rng, llm_cfg=llm_cfg_for_gen, operator_whitelist=operator_whitelist, global_feedback=global_feedback, llm_init_only=bool(llm_init_only), carry_elites=False)
        if loss_population_active:
            proposed_f = _propose_losses_for_generation(generation=int(gen), pop_f=int(max(loss_offspring_target, 1)), elites_f=resident_pop_f, diverse_elites_f=[], rng=rng, llm_cfg=llm_cfg_for_gen, operator_whitelist=operator_whitelist, global_feedback=global_feedback, llm_init_only=bool(llm_init_only), carry_elites=False)

        def _fill_unique_builders(proposals: Sequence[Mapping[str, Any]], *, target_size: int, resident_entries: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
            unique = []
            signatures = {str(item.get('signature')) for item in resident_entries if isinstance(item, Mapping) and item.get('signature')} | set(seen_g)
            for proposal in proposals:
                if not isinstance(proposal, dict) or not isinstance(proposal.get('ir'), PreferenceBuilderIR):
                    continue
                signature = _sig_pref_builder(proposal['ir'])
                if signature in signatures:
                    continue
                unique.append(dict(proposal))
                signatures.add(signature)
            return unique[:int(target_size)]

        def _fill_unique_losses(proposals: Sequence[Mapping[str, Any]], *, target_size: int, resident_entries: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
            unique = []
            signatures = {str(item.get('signature')) for item in resident_entries if isinstance(item, Mapping) and item.get('signature')} | set(seen_f)
            for proposal in proposals:
                if not isinstance(proposal, dict) or not isinstance(proposal.get('ir'), FreeLossIR):
                    continue
                signature = _sig_free_loss(proposal['ir'])
                if signature in signatures:
                    continue
                unique.append(dict(proposal))
                signatures.add(signature)
            return unique[:int(target_size)]
        proposed_g = _fill_unique_builders(proposed_g, target_size=int(builder_offspring_target), resident_entries=resident_pop_g) if builder_population_active else []
        proposed_f = _fill_unique_losses(proposed_f, target_size=int(loss_offspring_target), resident_entries=resident_pop_f) if loss_population_active else []
        if len(proposed_g) < int(builder_offspring_target) or len(proposed_f) < int(loss_offspring_target):
            LOGGER.warning('Offspring fill shortfall at gen=%d: proposed_g=%d/%d proposed_f=%d/%d resident_g=%d resident_f=%d', int(gen), int(len(proposed_g)), int(builder_offspring_target), int(len(proposed_f)), int(loss_offspring_target), int(len(resident_pop_g)), int(len(resident_pop_f)))
        g_entries: List[Dict[str, Any]] = []
        for idx, proposal in enumerate(proposed_g):
            ir: PreferenceBuilderIR = proposal['ir']
            sig = _sig_pref_builder(ir)
            mechanism_family = _canonical_mechanism_family(proposal.get('mechanism_family')) or _assign_mechanism_family(ir, rng) or 'pairwise_margin'
            entry: Dict[str, Any] = {'generation': int(gen), 'index': int(idx), 'id': f'g{gen:03d}_{idx:03d}_{sig[:8]}', 'signature': sig, 'mechanism_family': mechanism_family, 'origin': str(proposal.get('origin', 'unknown')), 'origin_base': proposal.get('origin_base'), 'op_type': proposal.get('op_type'), 'parents': list(proposal.get('parents', [])), 'attempt': proposal.get('attempt', 0), 'prompt_sha1': proposal.get('prompt_sha1'), 'prompt_path': proposal.get('prompt_path'), 'llm_seed': proposal.get('llm_seed'), 'history': list(proposal.get('history', [])) if isinstance(proposal.get('history', []), list) else [], 'ir': asdict(ir)}
            try:
                compiled = compile_preference_builder(ir, operator_whitelist=operator_whitelist)
                entry['compile_ok'] = True
                entry['compile_reason'] = 'ok'
                fc = _dummy_feature_cache(batch_size=8, k=16, variant='visible')
                pb = compiled.build_fn(fc, {'stage': 'builder_static'})
                bg = run_preference_builder_gates(pb, feature_cache=fc)
                entry['builder_static_ok'] = bool(bg.ok)
                entry['builder_static_reason'] = str(bg.reason)
                entry['builder_static_trace'] = bg.trace
            except Exception as exc:
                entry['compile_ok'] = False
                entry['compile_reason'] = str(exc)
                entry['builder_static_ok'] = False
                entry['builder_static_reason'] = 'compile_or_static_failed'
            g_entries.append(entry)
        _append_jsonl(builders_jsonl, g_entries)
        f_entries: List[Dict[str, Any]] = []
        for idx, proposal in enumerate(proposed_f):
            ir: FreeLossIR = proposal['ir']
            sig = _sig_free_loss(ir)
            static_res = run_static_gates(ir, operator_whitelist=operator_whitelist)
            mechanism_family = _canonical_mechanism_family(proposal.get('mechanism_family')) or _assign_mechanism_family(ir, rng) or 'pairwise_margin'
            entry: Dict[str, Any] = {'generation': int(gen), 'index': int(idx), 'id': f'f{gen:03d}_{idx:03d}_{sig[:8]}', 'signature': sig, 'mechanism_family': mechanism_family, 'origin': str(proposal.get('origin', 'unknown')), 'origin_base': proposal.get('origin_base'), 'op_type': proposal.get('op_type'), 'parents': list(proposal.get('parents', [])), 'attempt': proposal.get('attempt', 0), 'prompt_sha1': proposal.get('prompt_sha1'), 'prompt_path': proposal.get('prompt_path'), 'llm_seed': proposal.get('llm_seed'), 'history': list(proposal.get('history', [])) if isinstance(proposal.get('history', []), list) else [], 'novelty': proposal.get('novelty'), 'ir': asdict(ir), 'static_ok': bool(static_res.ok), 'static_reason': str(static_res.reason), 'static_trace': static_res.trace}
            if static_res.ok:
                try:
                    _ = compile_free_loss(ir, operator_whitelist=operator_whitelist)
                    entry['compile_ok'] = True
                    entry['compile_reason'] = 'ok'
                except Exception as exc:
                    entry['compile_ok'] = False
                    entry['compile_reason'] = str(exc)
            else:
                entry['compile_ok'] = False
                entry['compile_reason'] = 'static_gate_failed'
            f_entries.append(entry)
        _append_jsonl(losses_jsonl, f_entries)
        for entry in g_entries:
            sig = entry.get('signature')
            if sig:
                seen_g.add(str(sig))
        for entry in f_entries:
            sig = entry.get('signature')
            if sig:
                seen_f.add(str(sig))
        g_llm_ops = collections.Counter()
        f_llm_ops = collections.Counter()
        try:
            cache_stats = dict(loss_llm_ops.llm_cache_stats())
        except Exception:
            cache_stats = {}
        LOGGER.info('Gen %d LLM ops: builders=%s losses=%s cache=%s', int(gen), dict(g_llm_ops), dict(f_llm_ops), cache_stats)
        g_pool = [e for e in g_entries if bool(e.get('compile_ok')) and bool(e.get('builder_static_ok', True))]
        f_pool = [e for e in f_entries if bool(e.get('compile_ok'))]
        resident_g_ids = [str(e['id']) for e in resident_pop_g if isinstance(e, dict) and 'id' in e]
        resident_f_ids = [str(e['id']) for e in resident_pop_f if isinstance(e, dict) and 'id' in e]
        g_id_pool = list(dict.fromkeys(resident_g_ids + [str(e['id']) for e in g_pool]))
        f_id_pool = list(dict.fromkeys(resident_f_ids + [str(e['id']) for e in f_pool]))
        g_population_map = {str(e['id']): e for e in [e for e in resident_pop_g if isinstance(e, dict) and 'id' in e] + g_pool}
        f_population_map = {str(e['id']): e for e in [e for e in resident_pop_f if isinstance(e, dict) and 'id' in e] + f_pool}
        g_map = dict(g_population_map)
        f_map = dict(f_population_map)
        new_g_ids = [str(e['id']) for e in g_pool]
        new_f_ids = [str(e['id']) for e in f_pool]
        pairs: List[Tuple[str, str]] = []
        compiled_g: Dict[str, CompiledPreferenceBuilder] = {}
        compiled_f: Dict[str, CompiledFreeLoss] = {}
        for gid in g_id_pool:
            if gid not in g_map:
                continue
            try:
                compiled_g[gid] = compile_preference_builder(pref_builder_ir_from_json(g_map[gid]['ir']), operator_whitelist=operator_whitelist)
            except Exception:
                continue
        for fid in f_id_pool:
            if fid not in f_map:
                continue
            try:
                compiled_f[fid] = compile_free_loss(free_loss_ir_from_json(f_map[fid]['ir']), operator_whitelist=operator_whitelist)
            except Exception:
                continue
        pairs: List[Tuple[str, str]] = []
        reasons_by_pair: Dict[Tuple[str, str], List[str]] = {}
        fixed_builder_id_for_stage: str | None = None
        fixed_loss_id_for_stage: str | None = None
        try:
            _ensure_reference_compiled(compiled_g=compiled_g, compiled_f=compiled_f, operator_whitelist=operator_whitelist)
        except Exception as exc:
            LOGGER.warning('Failed to compile fixed-side reference candidates: %s', str(exc))
        fixed_builder_id = None
        if not fixed_builder_id and resident_pop_g:
            candidate_id = str(resident_pop_g[0].get('id') or '')
            if candidate_id in compiled_g:
                fixed_builder_id = candidate_id
        if not fixed_builder_id and G_REF_ID in compiled_g:
            fixed_builder_id = str(G_REF_ID)
        if not fixed_builder_id:
            fixed_builder_id = next((str(item) for item in g_id_pool if str(item) in compiled_g), None)
        fixed_loss_id = None
        if resident_pop_f:
            candidate_id = str(resident_pop_f[0].get('id') or '')
            if candidate_id in compiled_f:
                fixed_loss_id = candidate_id
        if not fixed_loss_id and isinstance(best_so_far, dict):
            candidate_id = str(best_so_far.get('loss_id') or '')
            if candidate_id in compiled_f:
                fixed_loss_id = candidate_id
        if not fixed_loss_id and F_REF_ID in compiled_f:
            fixed_loss_id = str(F_REF_ID)
        if not fixed_loss_id:
            fixed_loss_id = next((str(item) for item in f_id_pool if str(item) in compiled_f), None)
        if not fixed_builder_id or not fixed_loss_id:
            raise RuntimeError(f'Unable to resolve the frozen candidate required by the two-stage protocol: builder={fixed_builder_id!r}, loss={fixed_loss_id!r}.')
        if stage_phase == 'loss':
            fixed_builder_id_for_stage = str(fixed_builder_id)
        else:
            fixed_loss_id_for_stage = str(fixed_loss_id)
        pairs, reasons_by_pair, pair_phase_by_pair = _build_fixed_side_pairs(rng=rng, g_id_pool=list(new_g_ids if stage_phase == 'builder' else g_id_pool), f_id_pool=list(new_f_ids if stage_phase == 'loss' else f_id_pool), fixed_builder_id=str(fixed_builder_id), fixed_loss_id=str(fixed_loss_id), budget_loss=int(stage_loss_budget), budget_builder=int(stage_builder_budget))
        LOGGER.info('Fixed-side stage=%s frozen_builder=%s frozen_loss=%s scheduled_pairs=%d', str(stage_phase), str(fixed_builder_id_for_stage), str(fixed_loss_id_for_stage), int(len(pairs)))
        pair_records_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for p_idx, (gid, fid) in enumerate(pairs):
            g_entry = g_map.get(str(gid))
            f_entry = f_map.get(str(fid))
            if not isinstance(g_entry, dict) and str(gid) == G_REF_ID:
                g_entry = {'id': str(G_REF_ID), 'ir': asdict(_ref_builder_ir())}
            if not isinstance(f_entry, dict) and str(fid) == F_REF_ID:
                f_entry = {'id': str(F_REF_ID), 'ir': asdict(_ref_loss_ir())}
            if not isinstance(g_entry, dict) or not isinstance(f_entry, dict):
                continue
            rec = _evaluate_pair_worker({'generation': int(gen), 'pair_index': int(p_idx), 'g_entry': dict(g_entry), 'f_entry': dict(f_entry), 'cfg_yaml': dict(cfg_yaml), 'device_str': str(device_list[int(p_idx) % len(device_list)]), 'operator_whitelist': list(operator_whitelist), 'run_dir': str(run_dir)})
            rec['stage'] = 'high_fidelity'
            rec['phase'] = pair_phase_by_pair.get((str(gid), str(fid)), 'fixed_side')
            pair_records_map[str(gid), str(fid)] = rec
        pair_records: List[Dict[str, Any]] = [pair_records_map[str(g), str(f)] for g, f in pairs if (str(g), str(f)) in pair_records_map]
        for rec in pair_records:
            k = (str(rec.get('g_id')), str(rec.get('f_id')))
            rec['phase'] = pair_phase_by_pair.get(k, rec.get('phase', 'fixed_side'))
        for rec in pair_records:
            rec['stage_final'] = 'high_fidelity' if bool(rec.get('pair_ok')) else 'none'
            rec['final_score'] = rec.get('score') if bool(rec.get('pair_ok')) else None
            rec['metric_mode'] = str(metric_mode)
            rec['compare_target'] = 'incumbent'
            rec['improve_eps'] = float(improve_eps)
            rec['last_phase_label'] = str(last_phase_block_label) if last_phase_block_label is not None else None
            rec['last_phase_reference_score'] = float(last_phase_block_best_score) if last_phase_block_best_score is not None else None
        last_phase_reference_score = None
        if last_phase_block_best_score is not None:
            try:
                last_phase_reference_score = float(last_phase_block_best_score)
            except (TypeError, ValueError):
                last_phase_reference_score = None
        current_ref = None
        if isinstance(best_so_far, dict):
            try:
                current_ref = float(best_so_far.get('score'))
            except (TypeError, ValueError):
                current_ref = None
        LOGGER.info('COMPARE target=incumbent reference_score=%s threshold=%s metric_mode=%s improve_eps=%s', current_ref, _score_threshold(reference_score=current_ref, metric_mode=metric_mode, improve_eps=improve_eps), str(metric_mode), float(improve_eps))
        LOGGER.info('COMPARE target=last_phase last_phase=%s reference_score=%s threshold=%s metric_mode=%s improve_eps=%s', str(last_phase_block_label) if last_phase_block_label is not None else None, last_phase_reference_score, _score_threshold(reference_score=last_phase_reference_score, metric_mode=metric_mode, improve_eps=improve_eps), str(metric_mode), float(improve_eps))
        for rec in sorted(pair_records, key=lambda r: (_safe_int(r.get('pair_index', 10 ** 9), 10 ** 9), str(r.get('g_id', '')), str(r.get('f_id', '')))):
            reference_score = None
            if isinstance(best_so_far, dict):
                try:
                    reference_score = float(best_so_far.get('score'))
                except (TypeError, ValueError):
                    reference_score = None
            rec['reference_score'] = reference_score
            if str(rec.get('g_id')) == G_REF_ID and str(rec.get('f_id')) == F_REF_ID:
                rec['better_than_incumbent'] = False
                rec['delta_vs_incumbent'] = None
                rec['better_than_last_phase'] = False if last_phase_reference_score is not None else None
                rec['delta_vs_last_phase'] = None
                continue
            if not bool(rec.get('pair_ok')):
                rec['better_than_incumbent'] = False
                rec['delta_vs_incumbent'] = None
                rec['better_than_last_phase'] = False if last_phase_reference_score is not None else None
                rec['delta_vs_last_phase'] = None
                rec['final_score'] = None
                rec['stage_final'] = 'none'
                continue
            final_score = rec.get('final_score')
            if final_score is None:
                rec['better_than_incumbent'] = False
                rec['delta_vs_incumbent'] = None
                rec['better_than_last_phase'] = False if last_phase_reference_score is not None else None
                rec['delta_vs_last_phase'] = None
                continue
            try:
                cand_score_f = float(final_score)
            except (TypeError, ValueError):
                rec['better_than_incumbent'] = False
                rec['delta_vs_incumbent'] = None
                rec['better_than_last_phase'] = False if last_phase_reference_score is not None else None
                rec['delta_vs_last_phase'] = None
                rec['final_score'] = None
                rec['stage_final'] = 'none'
                continue
            if not math.isfinite(cand_score_f):
                rec['better_than_incumbent'] = False
                rec['delta_vs_incumbent'] = None
                rec['better_than_last_phase'] = False if last_phase_reference_score is not None else None
                rec['delta_vs_last_phase'] = None
                continue
            delta = _score_delta(cand_score=cand_score_f, ref_score=reference_score, metric_mode=metric_mode)
            better = _pair_record_beats_reference_record(rec, best_so_far if isinstance(best_so_far, Mapping) else None, metric_mode=metric_mode, improve_eps=improve_eps)
            rec['better_than_incumbent'] = bool(better)
            rec['delta_vs_incumbent'] = delta
            if last_phase_reference_score is None:
                rec['better_than_last_phase'] = None
                rec['delta_vs_last_phase'] = None
            else:
                rec['delta_vs_last_phase'] = _score_delta(cand_score=cand_score_f, ref_score=last_phase_reference_score, metric_mode=metric_mode)
                rec['better_than_last_phase'] = bool(_is_better_than_reference(cand_score=cand_score_f, reference_score=last_phase_reference_score, metric_mode=metric_mode, improve_eps=improve_eps))
            rec['score'] = float(cand_score_f)
            if bool(better):
                threshold = _score_threshold(reference_score=reference_score, metric_mode=metric_mode, improve_eps=improve_eps)
                best_so_far = {'score': float(cand_score_f), 'builder_id': str(rec.get('g_id')), 'loss_id': str(rec.get('f_id')), 'stage_final': str(rec.get('stage_final', 'none')), 'generation': int(gen), 'phase': str(rec.get('phase', 'fixed_side'))}
                LOGGER.info('NEW BEST: score=%s ref=%s delta=%s pair=(%s,%s) stage=%s gen=%d phase=%s threshold=%s compare_target=incumbent improve_eps=%s', float(cand_score_f), reference_score, delta, rec.get('g_id'), rec.get('f_id'), rec.get('stage_final'), int(gen), rec.get('phase', 'fixed_side'), threshold, float(improve_eps))
        for rec in pair_records:
            _append_pair_score_history(pair_score_history_map, rec)
        improved_this_gen = False
        if isinstance(best_so_far, dict):
            try:
                improved_this_gen = int(best_so_far.get('generation', -999)) == int(gen)
            except (TypeError, ValueError):
                improved_this_gen = False
        stagnation_generations = 0 if improved_this_gen else int(stagnation_generations) + 1
        gen_phase_best_score: float | None = None
        for rec in pair_records:
            if str(rec.get('g_id')) == G_REF_ID and str(rec.get('f_id')) == F_REF_ID:
                continue
            if not bool(rec.get('pair_ok')):
                continue
            try:
                score_f = float(rec.get('final_score'))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score_f):
                continue
            if _is_better_than_reference(cand_score=score_f, reference_score=gen_phase_best_score, metric_mode=metric_mode, improve_eps=0.0):
                gen_phase_best_score = score_f
        if gen_phase_best_score is not None and _is_better_than_reference(cand_score=float(gen_phase_best_score), reference_score=phase_block_best_score, metric_mode=metric_mode, improve_eps=0.0):
            phase_block_best_score = float(gen_phase_best_score)
        stage_ctr = collections.Counter((str(r.get('stage', '')) for r in pair_records))
        ok_ctr = sum((1 for r in pair_records if bool(r.get('pair_ok'))))
        reason_ctr = collections.Counter((str(r.get('pair_reason', '')) for r in pair_records))
        LOGGER.info('Gen %d summary: g_pool=%d f_pool=%d pairs=%d ok=%d stages=%s top_reasons=%s', int(gen), int(len(g_pool)), int(len(f_pool)), int(len(pair_records)), int(ok_ctr), dict(stage_ctr), reason_ctr.most_common(3))
        g_fail_compile = sum((1 for e in g_entries if not bool(e.get('compile_ok'))))
        g_fail_gate = sum((1 for e in g_entries if bool(e.get('compile_ok')) and (not bool(e.get('builder_static_ok', True)))))
        f_fail_static = sum((1 for e in f_entries if not bool(e.get('static_ok', True))))
        f_fail_compile = sum((1 for e in f_entries if bool(e.get('static_ok', True)) and (not bool(e.get('compile_ok', False)))))
        gate_kind_ctr: collections.Counter[str] = collections.Counter()
        for rec in pair_records:
            for k in ('builder_gate_trace', 'joint_gate_trace', 'affine_gate_trace'):
                t = rec.get(k)
                if not isinstance(t, dict):
                    continue
                kind = t.get('failure_kind') or t.get('failed_gate')
                if kind is None:
                    continue
                gate_kind_ctr[str(kind)] += 1
        best_pair_preview: Dict[str, Any] | None = None
        for rec in pair_records:
            if str(rec.get('g_id')) == G_REF_ID and str(rec.get('f_id')) == F_REF_ID:
                continue
            if not bool(rec.get('pair_ok')):
                continue
            if best_pair_preview is None or _pair_record_beats_reference_record(rec, best_pair_preview, metric_mode=metric_mode, improve_eps=0.0):
                score_f = _pair_record_effective_score(rec)
                if score_f is None:
                    continue
                best_pair_preview = {'g_id': rec.get('g_id'), 'f_id': rec.get('f_id'), 'score': score_f, 'stage': rec.get('stage'), 'pair_reason': rec.get('pair_reason'), 'builder_gate_reason': rec.get('builder_gate_reason'), 'joint_gate_reason': rec.get('joint_gate_reason')}
        llm_feedback_state = {'prev_gen_summary': {'stage_counts': dict(stage_ctr), 'ok_pairs': int(ok_ctr), 'top_pair_reasons': list(reason_ctr.most_common(8)), 'top_gate_failure_kinds': list(gate_kind_ctr.most_common(8))}}
        llm_feedback_state['stagnation_generations'] = int(stagnation_generations)
        try:
            fam_ctr = collections.Counter((_entry_mechanism_family(e) for e in elites_g if isinstance(e, Mapping)))
            llm_feedback_state['prev_gen_builder_families'] = list(fam_ctr.most_common(10))
        except Exception:
            llm_feedback_state['prev_gen_builder_families'] = []
        try:
            fam_ctr = collections.Counter((_entry_mechanism_family(e) for e in elites_f if isinstance(e, Mapping)))
            llm_feedback_state['prev_gen_loss_families'] = list(fam_ctr.most_common(10))
        except Exception:
            llm_feedback_state['prev_gen_loss_families'] = []
        llm_feedback_state['prev_gen_candidates'] = {'g_fail_compile': int(g_fail_compile), 'g_fail_gate': int(g_fail_gate), 'f_fail_static': int(f_fail_static), 'f_fail_compile': int(f_fail_compile), 'pop_g': int(len(g_entries)), 'pop_f': int(len(f_entries))}
        llm_feedback_state['prev_gen_best_pair_preview'] = best_pair_preview
        try:
            llm_feedback_state['prev_gen_llm_cache'] = dict(loss_llm_ops.llm_cache_stats())
        except Exception:
            pass
        gate_records: List[Dict[str, Any]] = []
        for rec in pair_records:
            gate_records.append({'generation': int(rec.get('generation', gen)), 'pair_index': int(rec.get('pair_index', -1)), 'g_id': rec.get('g_id'), 'f_id': rec.get('f_id'), 'stage': rec.get('stage'), 'phase': rec.get('phase'), 'score': rec.get('score'), 'stage_final': rec.get('stage_final'), 'final_score': rec.get('final_score'), 'reference_score': rec.get('reference_score'), 'improve_eps': rec.get('improve_eps'), 'better_than_incumbent': rec.get('better_than_incumbent'), 'delta_vs_incumbent': rec.get('delta_vs_incumbent'), 'better_than_last_phase': rec.get('better_than_last_phase'), 'delta_vs_last_phase': rec.get('delta_vs_last_phase'), 'last_phase_label': rec.get('last_phase_label'), 'last_phase_reference_score': rec.get('last_phase_reference_score'), 'better_than_incumbent_note': rec.get('better_than_incumbent_note'), 'compare_target': rec.get('compare_target'), 'static_ok': rec.get('f_static_ok'), 'static_reason': rec.get('f_static_reason'), 'builder_gate_ok': rec.get('builder_gate_ok'), 'builder_gate_reason': rec.get('builder_gate_reason'), 'builder_gate_trace': rec.get('builder_gate_trace'), 'joint_gate_ok': rec.get('joint_gate_ok'), 'joint_gate_reason': rec.get('joint_gate_reason'), 'joint_gate_trace': rec.get('joint_gate_trace'), 'pair_ok': rec.get('pair_ok'), 'pair_reason': rec.get('pair_reason')})
        _append_jsonl(pairs_jsonl, pair_records)
        _append_jsonl(gate_jsonl, gate_records)
        fitness_g, fitness_f = _credit_assignment_v2(pair_records=pair_records)
        builder_selection_active = stage_phase == 'builder'
        loss_selection_active = stage_phase == 'loss'
        def _rank_entries(entries: Sequence[Mapping[str, Any]], fit_map: Mapping[str, float], *, kind: str) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for e in entries:
                eid = str(e.get('id'))
                if kind == 'g' and eid == G_REF_ID:
                    continue
                if kind == 'f' and eid == F_REF_ID:
                    continue
                e2 = dict(e)
                e2['fitness'] = float(fit_map[eid]) if eid in fit_map else float(e2.get('fitness', float('inf')))
                mechanism_family = _entry_mechanism_family(e2)
                e2['mechanism_family'] = mechanism_family
                out.append(e2)
            out.sort(key=lambda x: float(x.get('fitness', float('inf'))))
            return out
        if builder_selection_active:
            ranked_g = _rank_entries(list(g_population_map.values()), fitness_g, kind='g')
            resident_pop_g = _family_aware_survivors(ranked_g, slots=elite_g)
            elites_g = list(resident_pop_g)
        else:
            ranked_g = [dict(entry) for entry in resident_pop_g]
            elites_g = resident_pop_g[:max(0, elite_g)]
        if loss_selection_active:
            ranked_f = _rank_entries(list(f_population_map.values()), fitness_f, kind='f')
            resident_pop_f = _family_aware_survivors(ranked_f, slots=elite_f)
            elites_f = list(resident_pop_f)
        else:
            ranked_f = [dict(entry) for entry in resident_pop_f]
            elites_f = resident_pop_f[:max(0, elite_f)]
        if builder_selection_active and elites_g:
            _atomic_write_json(os.path.join(run_dir, 'best_elite_builder.json'), dict(elites_g[0]))
        if elites_f:
            _atomic_write_json(os.path.join(run_dir, 'best_elite_loss.json'), dict(elites_f[0]))
        best_pair: Dict[str, Any] | None = None
        if isinstance(best_so_far, dict):
            gid_best = str(best_so_far.get('builder_id'))
            fid_best = str(best_so_far.get('loss_id'))
            best_pair = _resolve_best_pair_record(best_so_far=best_so_far, pair_records=pair_records, pair_cache_records=[], metric_mode=metric_mode)
            if best_pair is None:
                best_pair = {'g_id': gid_best, 'f_id': fid_best, 'score': best_so_far.get('score'), 'final_score': best_so_far.get('score'), 'stage_final': best_so_far.get('stage_final'), 'generation': best_so_far.get('generation'), 'phase': best_so_far.get('phase'), 'compare_target': 'incumbent', 'metric_mode': str(metric_mode), 'improve_eps': float(improve_eps), 'reference_score': None, 'better_than_incumbent': True}
        if best_pair is not None:
            best_pair_score_history = _current_best_pair_score_history()
            best_pair['score_history'] = list(best_pair_score_history)
            best_pair['score_history_summary'] = _score_history_summary(best_pair_score_history)
            _atomic_write_json(os.path.join(run_dir, 'best_pair.json'), best_pair)
            gid_best = str(best_pair.get('g_id', '')).strip()
            fid_best = str(best_pair.get('f_id', '')).strip()
            best_pair_eval_meta = _best_pair_eval_metadata(best_pair)
            best_builder = _best_pair_artifact_entry(cid=gid_best, best_pair=best_pair, cid_key='g_id', ir_key='g_ir', candidate_map=g_map, compiled_map=compiled_g, ref_ir_fn=_ref_builder_ir if gid_best == G_REF_ID else None)
            if best_builder is not None:
                best_builder.update(best_pair_eval_meta)
                best_builder['score_history'] = list(best_pair_score_history)
                best_builder['score_history_summary'] = _score_history_summary(best_pair_score_history)
                _atomic_write_json(os.path.join(run_dir, 'best_builder.json'), dict(best_builder))
            else:
                LOGGER.warning('Failed to resolve best_builder.json for best_pair g_id=%s', gid_best)
            best_loss = _best_pair_artifact_entry(cid=fid_best, best_pair=best_pair, cid_key='f_id', ir_key='f_ir', candidate_map=f_map, compiled_map=compiled_f, ref_ir_fn=_ref_loss_ir if fid_best == F_REF_ID else None)
            if best_loss is not None:
                best_loss.update(best_pair_eval_meta)
                best_loss['score_history'] = list(best_pair_score_history)
                best_loss['score_history_summary'] = _score_history_summary(best_pair_score_history)
                _atomic_write_json(os.path.join(run_dir, 'best_loss.json'), dict(best_loss))
            else:
                LOGGER.warning('Failed to resolve best_loss.json for best_pair f_id=%s', fid_best)
            _atomic_write_json(os.path.join(run_dir, 'best_pair_score_history.json'), {'g_id': gid_best, 'f_id': fid_best, 'history': list(best_pair_score_history), 'summary': _score_history_summary(best_pair_score_history)})
        _atomic_write_json(summary_json, _summary_state(gen))
    if generations <= gen_start:
        _atomic_write_json(summary_json, _summary_state(gen_start - 1))
    LOGGER.info('Search stage complete. Artifacts saved under: %s', os.path.abspath(run_dir))
