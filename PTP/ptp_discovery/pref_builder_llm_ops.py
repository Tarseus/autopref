from __future__ import annotations
import hashlib
import json
import logging
import os
import re
from dataclasses import asdict
from typing import Any, Mapping, Sequence
from fitness.co_features import INSTANCE_FEATURE_KEYS, SOLUTION_FEATURE_KEYS
from .free_loss_llm_ops import _call_llm, _extract_json_object, configure_llm_run
from .pref_builder_ir import PreferenceBuilderIR, ir_from_json as pref_builder_ir_from_json
LOGGER = logging.getLogger(__name__)

def _sha1(text: str) -> str:
    return hashlib.sha1(str(text).encode('utf-8')).hexdigest()

def _read_prompt(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def _parse_pref_builder_from_text(text: str) -> PreferenceBuilderIR:
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError('Builder JSON must be an object')
    code = str(obj.get('code', '') or '').strip()
    if not code:
        raise ValueError('Builder JSON missing code')
    if 'def generated_builder' not in code:
        raise ValueError('Builder code must define generated_builder')
    impl = obj.get('implementation_hint', {}) or {}
    if not isinstance(impl, dict):
        impl = {}
    expects = impl.get('expects', ['objective', 'log_prob']) or ['objective', 'log_prob']
    if not isinstance(expects, (list, tuple)):
        expects = [str(expects)]
    expects = [str(x) for x in expects if str(x).strip()]
    if not expects:
        expects = ['objective', 'log_prob']
    impl['expects'] = expects
    impl['returns'] = str(impl.get('returns', 'PrefBatch') or 'PrefBatch')
    impl['mode'] = 'pairwise'
    obj['implementation_hint'] = impl
    name = str(obj.get('name', '') or '').strip() or 'unnamed_preference_builder'
    name = re.sub('[^a-zA-Z0-9_\\-]+', '_', name)[:80]
    obj['name'] = name
    return pref_builder_ir_from_json(obj)

def build_runtime_prompt_context() -> Mapping[str, Any]:
    instance_keys = [str(key) for key in INSTANCE_FEATURE_KEYS]
    return {'mode': 'pairwise', 'available_keys': [*SOLUTION_FEATURE_KEYS, *instance_keys], 'solution_feature_keys': list(SOLUTION_FEATURE_KEYS), 'instance_feature_keys': instance_keys}

def _append_prompt_context_block(prompt: str, prompt_context: Mapping[str, Any] | None) -> str:
    if not prompt_context:
        return prompt
    return prompt + '\n\nRUNTIME_CONTEXT_JSON:\n' + json.dumps(dict(prompt_context), indent=2, ensure_ascii=False)

def _append_global_feedback(prompt: str, global_feedback: Mapping[str, Any] | None) -> str:
    if global_feedback is None:
        return prompt
    contract = '\n\nBUILDER_CONTRACT:\n- Construct every strict ordered preference pair for each instance.\n- Keep pair_idx identical to the full all-pairs construction.\n- Search only the finite nonnegative pair weight function.\n- Individual pair weights may be exactly zero.\n- Every instance must retain a positive total pair weight.\n- The per-instance weight coefficient of variation must exceed 0.1.\n- Return raw weights without builder-side sum or mean normalization.\n'
    return prompt + contract + '\nGLOBAL_FEEDBACK_JSON:\n' + json.dumps(global_feedback, indent=2, ensure_ascii=False)

def build_generation_prompt(generation_prompt_path: str, *, global_feedback: Mapping[str, Any] | None=None, prompt_context: Mapping[str, Any] | None=None) -> tuple[str, str]:
    prompt = _read_prompt(generation_prompt_path)
    prompt = _append_prompt_context_block(prompt, prompt_context)
    prompt = _append_global_feedback(prompt, global_feedback)
    return (prompt, _sha1(prompt))

def build_crossover_prompt(crossover_prompt_path: str, *, parents: Sequence[PreferenceBuilderIR], parents_fitness: Sequence[Mapping[str, Any]] | None=None, global_feedback: Mapping[str, Any] | None=None, prompt_context: Mapping[str, Any] | None=None) -> tuple[str, str]:
    prompt = _read_prompt(crossover_prompt_path)
    prompt = _append_prompt_context_block(prompt, prompt_context)
    blobs = []
    for idx, parent in enumerate(parents):
        metrics: Mapping[str, Any] = {}
        if parents_fitness is not None and idx < len(parents_fitness):
            metrics = parents_fitness[idx]
        blobs.append({'index': idx, 'name': parent.name, 'intuition': parent.intuition, 'hyperparams': parent.hyperparams, 'operators_used': parent.operators_used, 'implementation_hint': asdict(parent.implementation_hint), 'code': parent.code, 'metrics': {'fitness': float(metrics.get('fitness', float('inf'))) if metrics else None, 'mechanism_family': str(metrics.get('mechanism_family')) if metrics and metrics.get('mechanism_family') else None}})
    prompt = prompt + '\n\nPARENTS_JSON:\n' + json.dumps(blobs, indent=2, ensure_ascii=False)
    prompt = _append_global_feedback(prompt, global_feedback)
    return (prompt, _sha1(prompt))

def build_m2_prompt(m2_prompt_path: str, *, parent: PreferenceBuilderIR, parent_fitness: Mapping[str, Any] | None=None, global_feedback: Mapping[str, Any] | None=None, prompt_context: Mapping[str, Any] | None=None) -> tuple[str, str]:
    prompt = _read_prompt(m2_prompt_path)
    prompt = _append_prompt_context_block(prompt, prompt_context)
    metrics: Mapping[str, Any] = parent_fitness or {}
    blob = {'name': parent.name, 'intuition': parent.intuition, 'hyperparams': parent.hyperparams, 'operators_used': parent.operators_used, 'implementation_hint': asdict(parent.implementation_hint), 'code': parent.code, 'metrics': {'fitness': float(metrics.get('fitness', float('inf'))) if metrics else None, 'mechanism_family': str(metrics.get('mechanism_family')) if metrics and metrics.get('mechanism_family') else None}}
    prompt = prompt + '\n\nPARENT_JSON:\n' + json.dumps(blob, indent=2, ensure_ascii=False)
    prompt = _append_global_feedback(prompt, global_feedback)
    return (prompt, _sha1(prompt))

def build_paradigm_shift_prompt(prompt_path: str, *, parents: Sequence[PreferenceBuilderIR], parents_fitness: Sequence[Mapping[str, Any]] | None=None, global_feedback: Mapping[str, Any] | None=None, prompt_context: Mapping[str, Any] | None=None) -> tuple[str, str]:
    prompt = _read_prompt(prompt_path)
    prompt = _append_prompt_context_block(prompt, prompt_context)
    blobs = []
    for idx, parent in enumerate(parents):
        metrics: Mapping[str, Any] = {}
        if parents_fitness is not None and idx < len(parents_fitness):
            metrics = parents_fitness[idx]
        blobs.append({'index': idx, 'name': parent.name, 'intuition': parent.intuition, 'hyperparams': parent.hyperparams, 'operators_used': parent.operators_used, 'implementation_hint': asdict(parent.implementation_hint), 'code': parent.code, 'metrics': {'fitness': float(metrics.get('fitness', float('inf'))) if metrics else None, 'mechanism_family': str(metrics.get('mechanism_family')) if metrics and metrics.get('mechanism_family') else None}})
    prompt = prompt + '\n\nPARENTS_JSON:\n' + json.dumps(blobs, indent=2, ensure_ascii=False)
    prompt = _append_global_feedback(prompt, global_feedback)
    return (prompt, _sha1(prompt))

def build_structure_shift_prompt(prompt_path: str, *, parent: PreferenceBuilderIR, parent_fitness: Mapping[str, Any] | None=None, global_feedback: Mapping[str, Any] | None=None, prompt_context: Mapping[str, Any] | None=None) -> tuple[str, str]:
    prompt = _read_prompt(prompt_path)
    prompt = _append_prompt_context_block(prompt, prompt_context)
    metrics: Mapping[str, Any] = parent_fitness or {}
    blob = {'name': parent.name, 'intuition': parent.intuition, 'hyperparams': parent.hyperparams, 'operators_used': parent.operators_used, 'implementation_hint': asdict(parent.implementation_hint), 'code': parent.code, 'metrics': {'fitness': float(metrics.get('fitness', float('inf'))) if metrics else None, 'mechanism_family': str(metrics.get('mechanism_family')) if metrics and metrics.get('mechanism_family') else None}}
    prompt = prompt + '\n\nPARENT_JSON:\n' + json.dumps(blob, indent=2, ensure_ascii=False)
    prompt = _append_global_feedback(prompt, global_feedback)
    return (prompt, _sha1(prompt))

def build_constraint_inject_prompt(prompt_path: str, *, parent: PreferenceBuilderIR, parent_fitness: Mapping[str, Any] | None=None, global_feedback: Mapping[str, Any] | None=None, prompt_context: Mapping[str, Any] | None=None) -> tuple[str, str]:
    prompt = _read_prompt(prompt_path)
    prompt = _append_prompt_context_block(prompt, prompt_context)
    metrics: Mapping[str, Any] = parent_fitness or {}
    blob = {'name': parent.name, 'intuition': parent.intuition, 'hyperparams': parent.hyperparams, 'operators_used': parent.operators_used, 'implementation_hint': asdict(parent.implementation_hint), 'code': parent.code, 'metrics': {'fitness': float(metrics.get('fitness', float('inf'))) if metrics else None, 'mechanism_family': str(metrics.get('mechanism_family')) if metrics and metrics.get('mechanism_family') else None}}
    prompt = prompt + '\n\nPARENT_JSON:\n' + json.dumps(blob, indent=2, ensure_ascii=False)
    prompt = _append_global_feedback(prompt, global_feedback)
    return (prompt, _sha1(prompt))

def generate_pref_builder_candidate_with_meta(generation_prompt_path: str, *, operator_whitelist: Sequence[str], global_feedback: Mapping[str, Any] | None=None, prompt_context: Mapping[str, Any] | None=None) -> tuple[PreferenceBuilderIR, Mapping[str, Any]]:
    prompt, prompt_sha1 = build_generation_prompt(generation_prompt_path, global_feedback=global_feedback, prompt_context=prompt_context)
    raw = _call_llm(prompt, llm_op='E1_GENERATE', prompt_path=generation_prompt_path)
    json_str = _extract_json_object(raw)
    ir = _parse_pref_builder_from_text(json_str)
    return (ir, {'llm_op': 'E1_GENERATE', 'prompt_path': str(generation_prompt_path), 'prompt_sha1': str(prompt_sha1)})

def crossover_pref_builder_with_meta(crossover_prompt_path: str, *, parents: Sequence[PreferenceBuilderIR], parents_fitness: Sequence[Mapping[str, Any]] | None=None, global_feedback: Mapping[str, Any] | None=None, prompt_context: Mapping[str, Any] | None=None) -> tuple[PreferenceBuilderIR, Mapping[str, Any]]:
    prompt, prompt_sha1 = build_crossover_prompt(crossover_prompt_path, parents=parents, parents_fitness=parents_fitness, global_feedback=global_feedback, prompt_context=prompt_context)
    raw = _call_llm(prompt, llm_op='E1', prompt_path=crossover_prompt_path)
    json_str = _extract_json_object(raw)
    ir = _parse_pref_builder_from_text(json_str)
    return (ir, {'llm_op': 'E1', 'prompt_path': str(crossover_prompt_path), 'prompt_sha1': str(prompt_sha1)})

def m2_tune_builder_with_meta(m2_prompt_path: str, *, parent: PreferenceBuilderIR, parent_fitness: Mapping[str, Any] | None=None, global_feedback: Mapping[str, Any] | None=None, prompt_context: Mapping[str, Any] | None=None) -> tuple[PreferenceBuilderIR, Mapping[str, Any]]:
    prompt, prompt_sha1 = build_m2_prompt(m2_prompt_path, parent=parent, parent_fitness=parent_fitness, global_feedback=global_feedback, prompt_context=prompt_context)
    raw = _call_llm(prompt, llm_op='M2', prompt_path=m2_prompt_path)
    json_str = _extract_json_object(raw)
    ir = _parse_pref_builder_from_text(json_str)
    return (ir, {'llm_op': 'M2', 'prompt_path': str(m2_prompt_path), 'prompt_sha1': str(prompt_sha1)})

def paradigm_shift_builder_with_meta(prompt_path: str, *, parents: Sequence[PreferenceBuilderIR], parents_fitness: Sequence[Mapping[str, Any]] | None=None, global_feedback: Mapping[str, Any] | None=None, prompt_context: Mapping[str, Any] | None=None) -> tuple[PreferenceBuilderIR, Mapping[str, Any]]:
    prompt, prompt_sha1 = build_paradigm_shift_prompt(prompt_path, parents=parents, parents_fitness=parents_fitness, global_feedback=global_feedback, prompt_context=prompt_context)
    raw = _call_llm(prompt, llm_op='BUILDER_PARADIGM_SHIFT', prompt_path=prompt_path)
    json_str = _extract_json_object(raw)
    ir = _parse_pref_builder_from_text(json_str)
    return (ir, {'llm_op': 'BUILDER_PARADIGM_SHIFT', 'prompt_path': str(prompt_path), 'prompt_sha1': str(prompt_sha1)})

def structure_shift_builder_with_meta(prompt_path: str, *, parent: PreferenceBuilderIR, parent_fitness: Mapping[str, Any] | None=None, global_feedback: Mapping[str, Any] | None=None, prompt_context: Mapping[str, Any] | None=None) -> tuple[PreferenceBuilderIR, Mapping[str, Any]]:
    prompt, prompt_sha1 = build_structure_shift_prompt(prompt_path, parent=parent, parent_fitness=parent_fitness, global_feedback=global_feedback, prompt_context=prompt_context)
    raw = _call_llm(prompt, llm_op='BUILDER_STRUCTURE_SHIFT', prompt_path=prompt_path)
    json_str = _extract_json_object(raw)
    ir = _parse_pref_builder_from_text(json_str)
    return (ir, {'llm_op': 'BUILDER_STRUCTURE_SHIFT', 'prompt_path': str(prompt_path), 'prompt_sha1': str(prompt_sha1)})

def constraint_inject_builder_with_meta(prompt_path: str, *, parent: PreferenceBuilderIR, parent_fitness: Mapping[str, Any] | None=None, global_feedback: Mapping[str, Any] | None=None, prompt_context: Mapping[str, Any] | None=None) -> tuple[PreferenceBuilderIR, Mapping[str, Any]]:
    prompt, prompt_sha1 = build_constraint_inject_prompt(prompt_path, parent=parent, parent_fitness=parent_fitness, global_feedback=global_feedback, prompt_context=prompt_context)
    raw = _call_llm(prompt, llm_op='BUILDER_CONSTRAINT_INJECT', prompt_path=prompt_path)
    json_str = _extract_json_object(raw)
    ir = _parse_pref_builder_from_text(json_str)
    return (ir, {'llm_op': 'BUILDER_CONSTRAINT_INJECT', 'prompt_path': str(prompt_path), 'prompt_sha1': str(prompt_sha1)})
