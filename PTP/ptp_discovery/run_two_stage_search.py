from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


_PTP_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PTP_ROOT.parent
for _path in (_REPO_ROOT, _PTP_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ptp_discovery.staged_search_engine import run_search_stage


LOGGER = logging.getLogger("ptp_discovery.two_stage")
REPO_ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return dict(payload)


def _resolve_prompt_path(value: str, *, config_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    repo_candidate = REPO_ROOT / path
    if repo_candidate.exists() or str(value).startswith(("PTP/", "configs/")):
        return repo_candidate
    return config_path.parent / path


def _resolve_output_root(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _newest_run_dir(output_root: Path) -> Path:
    candidates = [path for path in output_root.iterdir() if path.is_dir()]
    if not candidates:
        raise RuntimeError(f"Search created no run directory under {output_root}.")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _validate_prompt_paths(config: Mapping[str, Any], *, config_path: Path) -> None:
    prompts = config.get("llm_prompts", {}) or {}
    if not isinstance(prompts, Mapping):
        raise ValueError("llm_prompts must be a mapping.")
    missing = []
    for key, raw_path in prompts.items():
        if not raw_path:
            continue
        resolved = _resolve_prompt_path(str(raw_path), config_path=config_path)
        if not resolved.is_file():
            missing.append(f"{key}={resolved}")
    if missing:
        raise FileNotFoundError("Missing calibrated prompt files: " + ", ".join(missing))


def _stage_configs(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    root = _load_yaml(config_path)
    common = root.get("common", {}) or {}
    stage1_override = root.get("stage1", {}) or {}
    stage2_override = root.get("stage2", {}) or {}
    if not all(isinstance(item, Mapping) for item in (common, stage1_override, stage2_override)):
        raise ValueError("common, stage1, and stage2 must all be mappings.")

    output_root = _resolve_output_root(
        str(root.get("output_root", "runs/two_stage_search"))
    )
    stage1 = _deep_merge(common, stage1_override)
    stage2 = _deep_merge(common, stage2_override)

    stage1_prompts = stage1.get("llm_prompts", {}) or {}
    stage2_prompts = stage2.get("llm_prompts", {}) or {}
    stage1["llm_prompts"] = {
        key: value for key, value in stage1_prompts.items() if str(key).startswith("loss_")
    }
    stage2["llm_prompts"] = {
        key: value for key, value in stage2_prompts.items() if str(key).startswith("builder_")
    }

    stage1["output_root"] = str(output_root / "stage1_loss")
    stage2["output_root"] = str(output_root / "stage2_weight")
    return stage1, stage2, output_root


def validate_two_stage_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path).resolve()
    stage1, stage2, output_root = _stage_configs(path)
    _validate_prompt_paths(stage1, config_path=path)
    _validate_prompt_paths(stage2, config_path=path)
    return {
        "config": str(path),
        "output_root": str(output_root),
        "stage1_mode": "loss_only",
        "stage2_mode": "builder_only",
        "stage1_prompts": sorted((stage1.get("llm_prompts", {}) or {}).keys()),
        "stage2_prompts": sorted((stage2.get("llm_prompts", {}) or {}).keys()),
    }


def run_two_stage_search(
    config_path: str,
    *,
    device: str | None = None,
) -> dict[str, Any]:
    path = Path(config_path).resolve()
    stage1, stage2, output_root = _stage_configs(path)
    _validate_prompt_paths(stage1, config_path=path)
    _validate_prompt_paths(stage2, config_path=path)
    if device:
        stage1["device"] = device
        stage2["device"] = device
        stage1["devices"] = [device]
        stage2["devices"] = [device]

    output_root.mkdir(parents=True, exist_ok=True)
    stage1_root = Path(stage1["output_root"])
    stage2_root = Path(stage2["output_root"])
    stage1_root.mkdir(parents=True, exist_ok=True)
    stage2_root.mkdir(parents=True, exist_ok=True)

    stage1_runtime = output_root / "stage1.runtime.yaml"
    stage1_runtime.write_text(
        yaml.safe_dump(stage1, sort_keys=False), encoding="utf-8"
    )
    LOGGER.info("Stage 1: running EoH-style loss search with the fixed reference builder.")
    run_search_stage(str(stage1_runtime), search_side='loss')
    stage1_run = _newest_run_dir(stage1_root)
    best_loss = stage1_run / "best_loss.json"
    if not best_loss.is_file():
        raise RuntimeError(f"Stage 1 produced no best_loss.json: {stage1_run}")

    stage2_runtime = output_root / "stage2.runtime.yaml"
    stage2_runtime.write_text(
        yaml.safe_dump(stage2, sort_keys=False), encoding="utf-8"
    )
    LOGGER.info("Stage 2: freezing %s and searching set-aware weighting.", best_loss)
    run_search_stage(str(stage2_runtime), search_side='builder', fixed_loss_path=str(best_loss))
    stage2_run = _newest_run_dir(stage2_root)
    best_builder = stage2_run / "best_builder.json"
    best_pair = stage2_run / "best_pair.json"
    if not best_builder.is_file() or not best_pair.is_file():
        raise RuntimeError(f"Stage 2 produced incomplete final artifacts: {stage2_run}")

    manifest = {
        "protocol": "loss_then_weighting",
        "config": str(path),
        "stage1_run": str(stage1_run),
        "stage1_best_loss": str(best_loss),
        "stage2_run": str(stage2_run),
        "final_loss": str(stage2_run / "best_loss.json"),
        "final_weighting": str(best_builder),
        "final_pair": str(best_pair),
    }
    manifest_path = output_root / "two_stage_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run AutoPref's fixed two-stage loss-then-weighting search."
    )
    parser.add_argument("--config", required=True, help="Two-stage YAML config.")
    parser.add_argument("--device", default=None, help="Optional device override.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate stage modes and calibrated prompt paths without searching.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    )
    args = _build_parser().parse_args()
    if args.validate_only:
        result = validate_two_stage_config(args.config)
    else:
        result = run_two_stage_search(args.config, device=args.device)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
