from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping

import torch

if __package__ is None or __package__ == "":
    root = Path(__file__).resolve().parents[2]
    for path in (root, root / "PTP"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

from fitness.free_loss_fidelity import extract_feature_cache
from ptp_discovery.free_loss_compiler import compile_free_loss
from ptp_discovery.free_loss_gates import run_joint_preference_gates, run_preference_builder_gates
from ptp_discovery.free_loss_ir import ir_from_json as free_loss_ir_from_json
from ptp_discovery.pref_builder_compiler import compile_preference_builder
from ptp_discovery.pref_builder_ir import ir_from_json as pref_builder_ir_from_json


def _write(path: str, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, target)


def _features(variant: str) -> Dict[str, torch.Tensor]:
    index = torch.arange(16, dtype=torch.float32)[None, :].repeat(8, 1)
    objective = index + torch.arange(8, dtype=torch.float32)[:, None] * 0.01
    log_prob = -0.1 * index
    if variant == "hidden":
        objective = objective * 10.0
        log_prob = log_prob * 12.0
    return extract_feature_cache(objective, log_prob)


def _failure(payload: Mapping[str, Any], kind: str, reason: str, trace: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ok": False,
        "generation": int(payload.get("generation", -1)),
        "pair_index": int(payload.get("pair_index", -1)),
        "failure_kind": kind,
        "reason": reason,
    }
    if trace is not None:
        result["trace"] = dict(trace)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    payload: Dict[str, Any] = {}
    try:
        payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))
        builder_entry = payload["g_entry"]
        loss_entry = payload["f_entry"]
        whitelist = list(payload.get("operator_whitelist", []))
        builder = compile_preference_builder(pref_builder_ir_from_json(builder_entry["ir"]), operator_whitelist=whitelist)
        loss = compile_free_loss(free_loss_ir_from_json(loss_entry["ir"]), operator_whitelist=whitelist)
        checks = []
        for variant in ("visible", "hidden"):
            features = _features(variant)
            preference = builder.build_fn(features, {"stage": "stage0_gate"})
            builder_gate = run_preference_builder_gates(preference, feature_cache=features)
            if not builder_gate.ok:
                _write(args.result, _failure(payload, "builder_gate_failed", str(builder_gate.reason), builder_gate.trace))
                return 0
            joint_gate = run_joint_preference_gates(
                loss,
                pref_batch=preference,
                feature_cache=features,
                min_pass_rate=0.8,
                swap_tolerance=0.001,
                swap_check_mode="data",
                swap_test_margin=1.0,
                grad_eps=1e-8,
                min_effective_grad_ratio=0.1,
                numeric_stress_enabled=True,
                numeric_stress_margin=120.0,
                numeric_stress_aux_scale=32.0,
                variant=variant,
            )
            checks.append({"variant": variant, "builder_gate_ok": True, "joint_gate_ok": bool(joint_gate.ok)})
            if not joint_gate.ok:
                _write(args.result, _failure(payload, "joint_gate_failed", str(joint_gate.reason), joint_gate.trace))
                return 0
        _write(args.result, {
            "ok": True,
            "generation": int(payload.get("generation", -1)),
            "pair_index": int(payload.get("pair_index", -1)),
            "failure_kind": None,
            "reason": "ok",
            "checks": checks,
        })
        return 0
    except Exception as exc:
        _write(args.result, _failure(payload, "sandbox_runtime_error", f"{type(exc).__name__}: {exc}", {"traceback": traceback.format_exc()}))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
