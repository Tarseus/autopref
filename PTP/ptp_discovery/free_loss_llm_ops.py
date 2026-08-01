from __future__ import annotations
import json
import logging
import os
import random
import re
import shlex
import time
from dataclasses import asdict
from functools import lru_cache
from hashlib import sha1
from typing import Any, Mapping, Sequence, TYPE_CHECKING
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None
if TYPE_CHECKING:
    from openai import OpenAI
from .free_loss_compiler import CompiledFreeLoss, compile_free_loss, parse_free_loss_from_text
from .free_loss_gates import supported_keys_for_mode
from .free_loss_ir import FreeLossIR
LOGGER = logging.getLogger(__name__)
_OPENAI_CLIENT: Any | None = None
_ENV_LOADED = False
_LLM_CACHE_PATH: str | None = None
_LLM_CACHE_INDEX: dict[str, str] = {}
_LLM_CACHE_HITS = 0
_LLM_CACHE_MISSES = 0

def _repo_root() -> str:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(this_dir, '..', '..'))

def _dotenv_candidates() -> list[str]:
    candidates: list[str] = []
    explicit = str(os.getenv('OPENAI_DOTENV_PATH', '') or '').strip()
    if explicit:
        candidates.append(os.path.abspath(explicit))
    candidates.append(os.path.abspath(os.path.join(os.getcwd(), '.env')))
    candidates.append(os.path.abspath(os.path.join(_repo_root(), '.env')))
    unique: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique

def _load_dotenv_fallback(path: str) -> int:
    loaded = 0
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.lower().startswith('export '):
                    line = line[7:].lstrip()
                if '=' not in line:
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                if not key or any((ch.isspace() for ch in key)):
                    continue
                if value and value[0] in {"'", '"'}:
                    try:
                        parsed = shlex.split(value, posix=True)
                    except ValueError:
                        parsed = []
                    if parsed:
                        value = parsed[0]
                else:
                    value = value.split(' #', 1)[0].strip()
                if key not in os.environ:
                    os.environ[key] = value
                    loaded += 1
    except OSError:
        return loaded
    return loaded

def configure_llm_run(*, run_dir: str | None=None, cache_path: str | None=None) -> None:
    global _LLM_CACHE_PATH, _LLM_CACHE_INDEX, _LLM_CACHE_HITS, _LLM_CACHE_MISSES
    if cache_path is None and run_dir:
        cache_path = os.path.join(str(run_dir), 'llm_cache.jsonl')
    if cache_path:
        _LLM_CACHE_PATH = str(cache_path)
        _LLM_CACHE_INDEX = {}
        _LLM_CACHE_HITS = 0
        _LLM_CACHE_MISSES = 0
        _load_llm_cache()

def llm_cache_stats() -> Mapping[str, Any]:
    return {'cache_path': _LLM_CACHE_PATH, 'cache_entries': int(len(_LLM_CACHE_INDEX)), 'cache_hits': int(_LLM_CACHE_HITS), 'cache_misses': int(_LLM_CACHE_MISSES)}

def _load_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    dotenv_candidates = _dotenv_candidates()
    loaded_dotenv_path: str | None = None
    if load_dotenv is not None:
        for path in dotenv_candidates:
            if os.path.isfile(path):
                load_dotenv(dotenv_path=path, override=False)
                loaded_dotenv_path = path
                break
    else:
        for path in dotenv_candidates:
            if os.path.isfile(path):
                loaded = _load_dotenv_fallback(path)
                loaded_dotenv_path = path
                LOGGER.warning('python-dotenv is unavailable; loaded %d key(s) via fallback parser from %s', loaded, path)
                break
    if not os.getenv('OPENAI_API_KEY'):
        searched = ', '.join(dotenv_candidates)
        raise RuntimeError(f"OPENAI_API_KEY is not set. Set it in the environment (or a .env at repo root), searched .env paths: [{searched}]. If you don't have the EoH deps installed, run: pip install -e '.[eoh]'.")
    if loaded_dotenv_path:
        LOGGER.info('Loaded LLM environment from .env: %s', loaded_dotenv_path)
    _ENV_LOADED = True

@lru_cache(maxsize=1)
def _openai_symbols() -> tuple[Any, tuple[type[BaseException], ...], type[BaseException] | None]:
    try:
        from openai import APIConnectionError, APITimeoutError, BadRequestError, InternalServerError, OpenAI, RateLimitError
    except ModuleNotFoundError as exc:
        raise RuntimeError("openai package is not installed. Install EoH extras with: pip install -e '.[eoh]'.") from exc
    retryable = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
    return (OpenAI, retryable, BadRequestError)

def _normalize_openai_base_url(raw_base_url: str) -> str:
    base_url = str(raw_base_url or '').strip().rstrip('/')
    suffix = '/chat/completions'
    if base_url.endswith(suffix):
        base_url = base_url[:-len(suffix)].rstrip('/')
    return base_url or str(raw_base_url)

def _make_openai_client() -> Any:
    timeout_s = float(os.getenv('OPENAI_TIMEOUT_S', '60') or 60)
    max_retries = int(os.getenv('OPENAI_MAX_RETRIES', '2') or 2)
    api_key = os.environ['OPENAI_API_KEY']
    base_url = _normalize_openai_base_url(os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'))
    OpenAI, _, _ = _openai_symbols()
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=max_retries)

def _get_openai_client() -> Any:
    global _OPENAI_CLIENT
    _load_env()
    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = _make_openai_client()
    return _OPENAI_CLIENT

def _should_retry_llm_error(exc: Exception) -> bool:
    status_code = getattr(exc, 'status_code', None)
    try:
        status_code_i = int(status_code)
    except (TypeError, ValueError):
        status_code_i = 0
    if status_code_i in {408, 409, 425, 429} or status_code_i >= 500:
        return True
    text = str(exc).lower()
    retryable_markers = ('no available channel', 'remote end closed connection', 'connection reset', 'connection aborted', 'unexpected_eof', 'eof occurred', 'ssl', 'temporarily unavailable', 'timeout', 'llm returned empty content')
    if any((marker in text for marker in retryable_markers)):
        return True
    try:
        _, retryable_types, bad_request = _openai_symbols()
    except Exception:
        return False
    if isinstance(exc, retryable_types):
        return True
    if bad_request is not None and isinstance(exc, bad_request) and ('get_token_error' in str(exc)):
        return True
    return False

def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, 'response', None)
    headers = getattr(response, 'headers', None)
    if not headers:
        return None
    value = None
    try:
        value = headers.get('retry-after')
        if value is None:
            value = headers.get('Retry-After')
    except Exception:
        value = None
    if value is None:
        return None
    try:
        seconds = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if seconds <= 0.0:
        return None
    return float(seconds)

def _read_prompt(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def build_runtime_prompt_context() -> Mapping[str, Any]:
    keys = ['log_prob_w', 'log_prob_l', 'cost_a', 'cost_b', 'cost_gap', 'delta_z', 'delta_rank', 'delta_regret', 'advantage_w', 'advantage_l', 'advantage_gap']
    return {'mode': 'pairwise', 'available_keys': keys}

def _append_prompt_context_block(prompt: str, prompt_context: Mapping[str, Any] | None) -> str:
    if not prompt_context:
        return prompt
    return prompt + '\n\nRUNTIME_CONTEXT_JSON:\n' + json.dumps(dict(prompt_context), indent=2, ensure_ascii=False)

def _extract_json_object(text: str) -> str:
    start = text.find('{')
    if start == -1:
        raise ValueError('No JSON object found in model output.')
    depth = 0
    end = None
    for i, ch in enumerate(text[start:], start=start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None or end <= start:
        raise ValueError('Failed to locate a complete JSON object in model output.')
    snippet = text[start:end + 1]
    invalid_escape_pattern = re.compile('\\\\(?!["\\\\/bfnrtu])')
    sanitized = invalid_escape_pattern.sub('\\\\\\\\', snippet)

    def _escape_control_chars_in_strings(s: str) -> str:
        out_chars: list[str] = []
        in_string = False
        escape = False
        for ch in s:
            if escape:
                out_chars.append(ch)
                escape = False
                continue
            if ch == '\\':
                out_chars.append(ch)
                escape = True
                continue
            if ch == '"':
                out_chars.append(ch)
                in_string = not in_string
                continue
            if in_string and ch in ('\n', '\r', '\t'):
                if ch == '\n':
                    out_chars.append('\\n')
                elif ch == '\r':
                    out_chars.append('\\r')
                else:
                    out_chars.append('\\t')
                continue
            out_chars.append(ch)
        return ''.join(out_chars)
    return _escape_control_chars_in_strings(sanitized)

def _load_llm_cache() -> None:
    global _LLM_CACHE_INDEX
    if not _LLM_CACHE_PATH:
        return
    path = str(_LLM_CACHE_PATH)
    if not os.path.isfile(path):
        return
    loaded = 0
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                key = rec.get('key')
                content = rec.get('content')
                if isinstance(key, str) and isinstance(content, str):
                    _LLM_CACHE_INDEX[key] = content
                    loaded += 1
    except Exception as exc:
        LOGGER.warning('Failed to load LLM cache (%s): %s', path, exc)
        return
    if loaded:
        LOGGER.info('Loaded LLM cache entries: %d (%s)', loaded, path)

def _append_llm_cache_record(record: Mapping[str, Any]) -> None:
    if not _LLM_CACHE_PATH:
        return
    path = str(_LLM_CACHE_PATH)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(dict(record), ensure_ascii=False) + '\n')
    except Exception as exc:
        LOGGER.warning('Failed to append LLM cache record (%s): %s', path, exc)

def _cache_key(*, model: str, prompt: str) -> str:
    blob = json.dumps({'model': str(model), 'prompt': str(prompt)}, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return sha1(blob).hexdigest()

def _resolve_llm_model_for_op(llm_op: str) -> str:
    del llm_op
    return str(os.getenv('OPENAI_MODEL', 'gpt-4.1-nano') or 'gpt-4.1-nano')

def _call_llm(prompt: str, *, llm_op: str, prompt_path: str | None) -> str:
    global _LLM_CACHE_HITS, _LLM_CACHE_MISSES
    client = _get_openai_client()
    model_name = _resolve_llm_model_for_op(llm_op)
    key = _cache_key(model=model_name, prompt=prompt)
    if _LLM_CACHE_PATH and key in _LLM_CACHE_INDEX:
        _LLM_CACHE_HITS += 1
        return _LLM_CACHE_INDEX[key]
    _LLM_CACHE_MISSES += 1
    max_attempts = int(os.getenv('OPENAI_CALL_MAX_ATTEMPTS', '6') or 6)
    base_backoff_s = float(os.getenv('OPENAI_CALL_BACKOFF_S', '1') or 1)
    max_backoff_s = float(os.getenv('OPENAI_CALL_BACKOFF_MAX_S', '30') or 30)
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(model=model_name, messages=[{'role': 'user', 'content': prompt}], temperature=0.7)
            content = resp.choices[0].message.content
            if not content:
                raise RuntimeError('LLM returned empty content.')
            out = content.strip()
            if _LLM_CACHE_PATH:
                _LLM_CACHE_INDEX[key] = out
                _append_llm_cache_record({'key': key, 'ts': float(time.time()), 'model': str(model_name), 'llm_op': str(llm_op), 'prompt_path': str(prompt_path) if prompt_path else None, 'prompt_sha1': sha1(prompt.encode('utf-8')).hexdigest(), 'content': out})
            return out
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts or not _should_retry_llm_error(exc):
                raise
            retry_after_s = _retry_after_seconds(exc)
            sleep_s = min(max_backoff_s, base_backoff_s * 2 ** (attempt - 1))
            if retry_after_s is not None:
                sleep_s = min(max_backoff_s, max(sleep_s, retry_after_s))
            sleep_s = sleep_s * (0.5 + random.random())
            LOGGER.warning('LLM call failed (attempt %d/%d, model=%s): %s; retrying in %.1fs', attempt, max_attempts, model_name, str(exc), sleep_s)
            time.sleep(sleep_s)
    raise RuntimeError('LLM call failed after retries.') from last_exc

def generate_free_loss_candidate(generation_prompt_path: str, *, operator_whitelist: Sequence[str], global_feedback: Mapping[str, Any] | None=None, prompt_context: Mapping[str, Any] | None=None) -> FreeLossIR:
    del operator_whitelist
    base_prompt = _append_prompt_context_block(_read_prompt(generation_prompt_path), prompt_context)
    prompt = base_prompt
    if global_feedback is not None:
        feedback_blob = json.dumps(global_feedback, indent=2, ensure_ascii=False)
        prompt = prompt + '\n\nGLOBAL_FEEDBACK_JSON:\n' + feedback_blob
    raw = _call_llm(prompt, llm_op='E1_GENERATE', prompt_path=generation_prompt_path)
    json_str = _extract_json_object(raw)
    return parse_free_loss_from_text(json_str)

def crossover_free_loss(crossover_prompt_path: str, parents: Sequence[FreeLossIR], parents_fitness: Sequence[Mapping[str, Any]] | None=None, global_feedback: Mapping[str, Any] | None=None, prompt_context: Mapping[str, Any] | None=None) -> FreeLossIR:
    prompt = _append_prompt_context_block(_read_prompt(crossover_prompt_path), prompt_context)
    parent_blobs = []
    for idx, parent in enumerate(parents):
        metrics: Mapping[str, Any] = {}
        if parents_fitness is not None and idx < len(parents_fitness):
            metrics = parents_fitness[idx]
        blob = {'index': idx, 'name': parent.name, 'intuition': parent.intuition, 'pseudocode': parent.pseudocode, 'hyperparams': parent.hyperparams, 'operators_used': parent.operators_used, 'code': parent.code, 'theoretical_basis': getattr(parent, 'theoretical_basis', ''), 'metrics': {'fitness': float(metrics.get('fitness', float('inf'))) if metrics else None, 'mechanism_family': str(metrics.get('mechanism_family')) if metrics and metrics.get('mechanism_family') else None}}
        parent_blobs.append(blob)
    prompt = prompt + '\n\nPARENTS_JSON:\n' + json.dumps(parent_blobs, indent=2, ensure_ascii=False)
    if global_feedback is not None:
        feedback_blob = json.dumps(global_feedback, indent=2, ensure_ascii=False)
        prompt = prompt + '\n\nGLOBAL_FEEDBACK_JSON:\n' + feedback_blob
    raw = _call_llm(prompt, llm_op='E1', prompt_path=crossover_prompt_path)
    json_str = _extract_json_object(raw)
    return parse_free_loss_from_text(json_str)

def paradigm_shift_free_loss(prompt_path: str, parents: Sequence[FreeLossIR], parents_fitness: Sequence[Mapping[str, Any]] | None=None, global_feedback: Mapping[str, Any] | None=None, prompt_context: Mapping[str, Any] | None=None) -> FreeLossIR:
    prompt = _append_prompt_context_block(_read_prompt(prompt_path), prompt_context)
    parent_blobs = []
    for idx, parent in enumerate(parents):
        metrics: Mapping[str, Any] = {}
        if parents_fitness is not None and idx < len(parents_fitness):
            metrics = parents_fitness[idx]
        parent_blobs.append({'index': idx, 'name': parent.name, 'intuition': parent.intuition, 'pseudocode': parent.pseudocode, 'hyperparams': parent.hyperparams, 'operators_used': parent.operators_used, 'code': parent.code, 'theoretical_basis': getattr(parent, 'theoretical_basis', ''), 'metrics': {'fitness': float(metrics.get('fitness', float('inf'))) if metrics else None, 'mechanism_family': str(metrics.get('mechanism_family')) if metrics and metrics.get('mechanism_family') else None}})
    prompt = prompt + '\n\nPARENTS_JSON:\n' + json.dumps(parent_blobs, indent=2, ensure_ascii=False)
    if global_feedback is not None:
        prompt = prompt + '\n\nGLOBAL_FEEDBACK_JSON:\n' + json.dumps(global_feedback, indent=2, ensure_ascii=False)
    raw = _call_llm(prompt, llm_op='LOSS_PARADIGM_SHIFT', prompt_path=prompt_path)
    json_str = _extract_json_object(raw)
    return parse_free_loss_from_text(json_str)

def structure_shift_free_loss(prompt_path: str, parent: FreeLossIR, parent_fitness: Mapping[str, Any] | None=None, global_feedback: Mapping[str, Any] | None=None, prompt_context: Mapping[str, Any] | None=None) -> FreeLossIR:
    prompt = _append_prompt_context_block(_read_prompt(prompt_path), prompt_context)
    metrics: Mapping[str, Any] = parent_fitness or {}
    parent_blob = {'name': parent.name, 'intuition': parent.intuition, 'pseudocode': parent.pseudocode, 'hyperparams': parent.hyperparams, 'operators_used': parent.operators_used, 'code': parent.code, 'theoretical_basis': getattr(parent, 'theoretical_basis', ''), 'metrics': {'fitness': float(metrics.get('fitness', float('inf'))) if metrics else None, 'mechanism_family': str(metrics.get('mechanism_family')) if metrics and metrics.get('mechanism_family') else None}}
    prompt = prompt + '\n\nPARENT_JSON:\n' + json.dumps(parent_blob, indent=2, ensure_ascii=False)
    if global_feedback is not None:
        prompt = prompt + '\n\nGLOBAL_FEEDBACK_JSON:\n' + json.dumps(global_feedback, indent=2, ensure_ascii=False)
    raw = _call_llm(prompt, llm_op='LOSS_STRUCTURE_SHIFT', prompt_path=prompt_path)
    json_str = _extract_json_object(raw)
    return parse_free_loss_from_text(json_str)

def constraint_inject_free_loss(prompt_path: str, parent: FreeLossIR, parent_fitness: Mapping[str, Any] | None=None, global_feedback: Mapping[str, Any] | None=None, prompt_context: Mapping[str, Any] | None=None) -> FreeLossIR:
    prompt = _append_prompt_context_block(_read_prompt(prompt_path), prompt_context)
    metrics: Mapping[str, Any] = parent_fitness or {}
    parent_blob = {'name': parent.name, 'intuition': parent.intuition, 'pseudocode': parent.pseudocode, 'hyperparams': parent.hyperparams, 'operators_used': parent.operators_used, 'code': parent.code, 'theoretical_basis': getattr(parent, 'theoretical_basis', ''), 'metrics': {'fitness': float(metrics.get('fitness', float('inf'))) if metrics else None, 'mechanism_family': str(metrics.get('mechanism_family')) if metrics and metrics.get('mechanism_family') else None}}
    prompt = prompt + '\n\nPARENT_JSON:\n' + json.dumps(parent_blob, indent=2, ensure_ascii=False)
    if global_feedback is not None:
        prompt = prompt + '\n\nGLOBAL_FEEDBACK_JSON:\n' + json.dumps(global_feedback, indent=2, ensure_ascii=False)
    raw = _call_llm(prompt, llm_op='LOSS_CONSTRAINT_INJECT', prompt_path=prompt_path)
    json_str = _extract_json_object(raw)
    return parse_free_loss_from_text(json_str)

def m2_tune_hparams(m2_prompt_path: str, parent: FreeLossIR, parent_fitness: Mapping[str, Any] | None=None, global_feedback: Mapping[str, Any] | None=None, prompt_context: Mapping[str, Any] | None=None) -> FreeLossIR:
    prompt = _append_prompt_context_block(_read_prompt(m2_prompt_path), prompt_context)
    metrics: Mapping[str, Any] = parent_fitness or {}
    parent_blob = {'name': parent.name, 'intuition': parent.intuition, 'pseudocode': parent.pseudocode, 'hyperparams': parent.hyperparams, 'operators_used': parent.operators_used, 'code': parent.code, 'theoretical_basis': getattr(parent, 'theoretical_basis', ''), 'metrics': {'fitness': float(metrics.get('fitness', float('inf'))) if metrics else None, 'mechanism_family': str(metrics.get('mechanism_family')) if metrics and metrics.get('mechanism_family') else None}}
    prompt = prompt + '\n\nPARENT_JSON:\n' + json.dumps(parent_blob, indent=2, ensure_ascii=False)
    if global_feedback is not None:
        feedback_blob = json.dumps(global_feedback, indent=2, ensure_ascii=False)
        prompt = prompt + '\n\nGLOBAL_FEEDBACK_JSON:\n' + feedback_blob
    raw = _call_llm(prompt, llm_op='M2', prompt_path=m2_prompt_path)
    json_str = _extract_json_object(raw)
    tuned = parse_free_loss_from_text(json_str)
    parent_hp = dict(parent.hyperparams or {})
    tuned_hp = dict(tuned.hyperparams or {})
    if parent.code.strip():
        tuned_hp = {k: tuned_hp.get(k, parent_hp.get(k)) for k in parent_hp.keys()}
    else:
        tuned_hp = tuned_hp or parent_hp
    return FreeLossIR(name=tuned.name or f'{parent.name}_m2', intuition=tuned.intuition or parent.intuition, pseudocode=tuned.pseudocode or parent.pseudocode, hyperparams=tuned_hp, operators_used=list(parent.operators_used), implementation_hint=parent.implementation_hint, code=parent.code, theoretical_basis=tuned.theoretical_basis or getattr(parent, 'theoretical_basis', ''))
