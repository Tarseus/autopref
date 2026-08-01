from typing import Any, Callable, Sequence

import json
from pathlib import Path
import sys

import torch
import torch.nn as nn

from rl4co.data.transforms import StateAugmentation
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.reinforce.free_loss import compile_free_loss, ir_from_json
from rl4co.models.rl.reinforce.reinforce import REINFORCE
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.utils.ops import gather_by_index, unbatchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

_DEFAULT_FREE_LOSS_OBSERVABLES = ("seq_len", "log_prob_mean", "advantage")


def _normalize_free_loss_observables(observables: Sequence[str] | None) -> tuple[str, ...]:
    values = observables if observables else _DEFAULT_FREE_LOSS_OBSERVABLES
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        key = str(raw or "").strip()
        if not key or key in seen:
            continue
        out.append(key)
        seen.add(key)
    return tuple(out) if out else _DEFAULT_FREE_LOSS_OBSERVABLES


class POMO(REINFORCE):

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module = None,
        policy_kwargs={},
        baseline: str = "shared",
        num_augment: int = 8,
        augment_fn: str | Callable = "dihedral8",
        first_aug_identity: bool = True,
        feats: list = None,
        num_starts: int = None,
        loss_type: str = "rl_loss",
        alpha: float = 1.0,
        free_loss_ir_json_path: str | None = None,
        pref_builder_ir_json_path: str | None = None,
        pref_pair_json_path: str | None = None,
        pref_builder_kwargs: dict | None = None,
        free_loss_observables: Sequence[str] | None = None,
        **kwargs,
    ):
        self.save_hyperparameters(logger=False)

        if policy is None:
            policy_kwargs_with_defaults = {
                "num_encoder_layers": 6,
                "normalization": "instance",
                "use_graph_context": False,
            }
            policy_kwargs_with_defaults.update(policy_kwargs)
            policy = AttentionModelPolicy(
                env_name=env.name, **policy_kwargs_with_defaults
            )

        assert baseline == "shared", "POMO only supports shared baseline"

        super(POMO, self).__init__(env, policy, baseline, **kwargs)

        self.num_starts = num_starts
        self.num_augment = num_augment
        if self.num_augment > 1:
            self.augment = StateAugmentation(
                num_augment=self.num_augment,
                augment_fn=augment_fn,
                first_aug_identity=first_aug_identity,
                feats=feats,
            )
        else:
            self.augment = None

        for phase in ["train", "val", "test"]:
            self.set_decode_type_multistart(phase)

        self.loss_type = loss_type
        self.alpha = float(alpha)
        self.free_loss_ir_json_path = free_loss_ir_json_path
        self.pref_builder_ir_json_path = pref_builder_ir_json_path
        self.pref_pair_json_path = pref_pair_json_path
        self.pref_builder_kwargs = {} if pref_builder_kwargs is None else dict(pref_builder_kwargs)
        self.free_loss_observables = _normalize_free_loss_observables(free_loss_observables)
        self.free_loss = None
        self.pref_builder = None
        self._pref_extract_feature_cache = None
        self._pref_build_runtime_observables = None
        self._pref_evaluate_pairwise_loss = None
        self._pref_batch_cls = None
        self._resolve_pref_pair_artifacts()
        if self.pref_pair_json_path and self.loss_type == "rl_loss" and self.free_loss_ir_json_path:
            self.loss_type = "free_loss"
            log.info("Resolved loss from pref_pair_json_path; switching loss_type to free_loss")
        if self.pref_builder_ir_json_path:
            self._load_pref_builder()
        if self.loss_type == "free_loss":
            self._load_free_loss()
        elif self.pref_builder is not None:
            log.warning(
                "pref_builder_ir_json_path is set but loss_type=%s; the builder will be ignored",
                self.loss_type,
            )

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        td = self.env.reset(batch)
        n_aug, n_start = self.num_augment, self.num_starts
        n_start = self.env.get_num_starts(td) if n_start is None else n_start

        if phase == "train":
            n_aug = 0
        elif n_aug > 1:
            td = self.augment(td)

        policy_kwargs: dict[str, Any] = {"phase": phase, "num_starts": n_start}
        if phase == "train" and self.loss_type == "free_loss":
            observables = set(self.free_loss_observables)
            want_seq_len = bool(observables & {"seq_len", "log_prob_mean", "entropy_mean"})
            want_entropy = bool(observables & {"entropy", "entropy_mean"})
            want_step_logp = "log_prob_step" in observables
            want_actions = bool(want_seq_len and not want_step_logp)
            policy_kwargs.update(
                {
                    "return_actions": want_actions,
                    "return_entropy": want_entropy,
                    "return_sum_log_likelihood": not want_step_logp,
                }
            )
        out = self.policy(td, self.env, **policy_kwargs)

        reward = unbatchify(out["reward"], (n_aug, n_start))

        if phase == "train":
            assert n_start > 1, "num_starts must be > 1 during training"
            raw_log_likelihood = out["log_likelihood"]
            if self.loss_type == "free_loss" and raw_log_likelihood.ndim > 1:
                log_likelihood_step = unbatchify(raw_log_likelihood, (n_aug, n_start))
                log_likelihood = log_likelihood_step.sum(dim=-1)
                out["log_likelihood_step"] = log_likelihood_step
                out["log_likelihood"] = log_likelihood
            else:
                log_likelihood = unbatchify(raw_log_likelihood, (n_aug, n_start))
                out["log_likelihood"] = log_likelihood
            if self.loss_type == "free_loss" and isinstance(out.get("entropy"), torch.Tensor):
                out["entropy"] = unbatchify(out["entropy"], (n_aug, n_start))
            if self.loss_type == "free_loss" and isinstance(out.get("actions"), torch.Tensor):
                out["actions"] = unbatchify(out["actions"], (n_aug, n_start))
            self.calculate_loss(td, batch, out, reward, log_likelihood)
            max_reward, max_idxs = reward.max(dim=-1)
            out.update({"max_reward": max_reward})
        else:
            if n_start > 1:
                max_reward, max_idxs = reward.max(dim=-1)
                out.update({"max_reward": max_reward})

                if out.get("actions", None) is not None:
                    actions = unbatchify(out["actions"], (n_aug, n_start))
                    out.update(
                        {
                            "best_multistart_actions": gather_by_index(
                                actions, max_idxs, dim=max_idxs.dim()
                            )
                        }
                    )
                    out["actions"] = actions

            if n_aug > 1:
                reward_ = max_reward if n_start > 1 else reward
                max_aug_reward, max_idxs = reward_.max(dim=1)
                out.update({"max_aug_reward": max_aug_reward})

                if out.get("actions", None) is not None:
                    actions_ = (
                        out["best_multistart_actions"] if n_start > 1 else out["actions"]
                    )
                    out.update({"best_aug_actions": gather_by_index(actions_, max_idxs)})

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}

    def calculate_loss(
        self,
        td,
        batch,
        policy_out: dict,
        reward: torch.Tensor | None = None,
        log_likelihood: torch.Tensor | None = None,
    ):
        reward = reward if reward is not None else policy_out["reward"]
        log_likelihood = (
            log_likelihood if log_likelihood is not None else policy_out["log_likelihood"]
        )

        if self.loss_type == "rl_loss":
            return super().calculate_loss(td, batch, policy_out, reward, log_likelihood)
        if self.loss_type == "free_loss":
            loss, pair_count = self._free_loss_loss_fn(reward, log_likelihood, policy_out)
            policy_out.update(
                {
                    "loss": loss,
                    "free_loss": loss.detach(),
                    "free_loss_pair_count": pair_count,
                }
            )
            return policy_out

        raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def _load_free_loss(self) -> None:
        if self.free_loss_ir_json_path is None:
            raise ValueError(
                "When loss_type is 'free_loss', free_loss_ir_json_path must be set."
            )
        path = Path(self.free_loss_ir_json_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"free_loss_ir_json_path does not exist: {path.as_posix()}"
            )
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        ir_obj = payload.get("ir", payload)
        ir = ir_from_json(ir_obj)
        self.free_loss = compile_free_loss(ir)

    def _load_free_loss_runtime_helpers(self) -> None:
        if (
            self._pref_extract_feature_cache is not None
            and self._pref_build_runtime_observables is not None
            and self._pref_evaluate_pairwise_loss is not None
            and self._pref_batch_cls is not None
        ):
            return

        self._ensure_ptp_root_on_path()
        try:
            from fitness.free_loss_fidelity import (
                PrefBatch,
                build_runtime_observables,
                evaluate_pairwise_loss,
                extract_feature_cache,
            )
        except ImportError as exc:
            raise ImportError(
                "Failed to import PTP free-loss runtime modules. "
                "Ensure the repository still contains the PTP/ directory."
            ) from exc

        self._pref_extract_feature_cache = extract_feature_cache
        self._pref_build_runtime_observables = build_runtime_observables
        self._pref_evaluate_pairwise_loss = evaluate_pairwise_loss
        self._pref_batch_cls = PrefBatch

    def _resolve_pref_pair_artifacts(self) -> None:
        if self.pref_pair_json_path is None:
            return

        pair_path = Path(self.pref_pair_json_path).expanduser()
        if not pair_path.is_file():
            raise FileNotFoundError(f"pref_pair_json_path does not exist: {pair_path.as_posix()}")

        run_dir = pair_path.parent
        builder_path = run_dir / "best_builder.json"
        loss_path = run_dir / "best_loss.json"
        if self.pref_builder_ir_json_path is None:
            if not builder_path.is_file():
                raise FileNotFoundError(
                    f"best_builder.json not found next to pref_pair_json_path: {builder_path.as_posix()}"
                )
            self.pref_builder_ir_json_path = builder_path.as_posix()
        if self.free_loss_ir_json_path is None:
            if not loss_path.is_file():
                raise FileNotFoundError(
                    f"best_loss.json not found next to pref_pair_json_path: {loss_path.as_posix()}"
                )
            self.free_loss_ir_json_path = loss_path.as_posix()

        try:
            with pair_path.open("r", encoding="utf-8") as f:
                pair_payload = json.load(f)
            pair_gid = str(pair_payload.get("g_id", "")).strip()
            pair_fid = str(pair_payload.get("f_id", "")).strip()
        except Exception:
            return

        for expected_id, artifact_path, key in (
            (pair_gid, self.pref_builder_ir_json_path, "id"),
            (pair_fid, self.free_loss_ir_json_path, "id"),
        ):
            if not expected_id or not artifact_path:
                continue
            try:
                with Path(artifact_path).expanduser().open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                actual_id = str(payload.get(key, "")).strip()
            except Exception:
                continue
            if actual_id and actual_id != expected_id:
                log.warning(
                    "Resolved artifact %s id=%s does not match pref_pair expected id=%s",
                    artifact_path,
                    actual_id,
                    expected_id,
                )

    @staticmethod
    def _ensure_ptp_root_on_path() -> None:
        repo_root = Path(__file__).resolve().parents[4]
        ptp_root = repo_root / "PTP"
        ptp_root_str = str(ptp_root.resolve())
        if ptp_root.is_dir() and ptp_root_str not in sys.path:
            sys.path.insert(0, ptp_root_str)

    def _load_pref_builder(self) -> None:
        if self.pref_builder_ir_json_path is None:
            raise ValueError("pref_builder_ir_json_path must be set before loading a preference builder.")

        self._ensure_ptp_root_on_path()
        try:
            from ptp_discovery.pref_builder_compiler import compile_preference_builder
            from ptp_discovery.pref_builder_ir import ir_from_json as pref_builder_ir_from_json
        except ImportError as exc:
            raise ImportError(
                "Failed to import PTP preference-builder modules. "
                "Ensure the repository still contains the PTP/ directory."
            ) from exc
        self._load_free_loss_runtime_helpers()

        path = Path(self.pref_builder_ir_json_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"pref_builder_ir_json_path does not exist: {path.as_posix()}"
            )
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        ir_obj = payload.get("ir", payload)
        ir = pref_builder_ir_from_json(ir_obj)
        self.pref_builder = compile_preference_builder(ir)

    def _free_loss_loss_fn(
        self, reward: torch.Tensor, log_likelihood: torch.Tensor, policy_out: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.free_loss is None:
            raise RuntimeError(
                "free_loss is not compiled; check free_loss_ir_json_path."
            )
        self._load_free_loss_runtime_helpers()
        if self._pref_extract_feature_cache is None or self._pref_build_runtime_observables is None:
            raise RuntimeError("Free-loss runtime helpers are not initialized.")

        objective = -reward
        seq_len = None
        seq_len_fallback = None
        actions = policy_out.get("actions")
        if isinstance(actions, torch.Tensor):
            seq_len = torch.full_like(log_likelihood, float(actions.shape[-1]))
        elif isinstance(policy_out.get("log_likelihood_step"), torch.Tensor):
            seq_len = torch.full_like(
                log_likelihood,
                float(policy_out["log_likelihood_step"].shape[-1]),
            )
        else:
            size_value = None
            generator = getattr(self.env, "generator", None)
            for attr in ("num_loc", "num_jobs", "num_job"):
                value = getattr(generator, attr, None)
                if value is not None:
                    size_value = float(value)
                    break
            if size_value is not None:
                seq_len_fallback = size_value
                seq_len = torch.full_like(log_likelihood, size_value)

        extra = self._pref_build_runtime_observables(
            reward,
            log_likelihood,
            observables=self.free_loss_observables,
            seq_len=seq_len,
            log_prob_step=policy_out.get("log_likelihood_step"),
            entropy=policy_out.get("entropy"),
            seq_len_fallback=seq_len_fallback,
        )
        feature_cache = self._pref_extract_feature_cache(
            objective=objective,
            log_prob=log_likelihood,
            extra=extra,
        )

        loss_batch: dict[str, torch.Tensor]
        pair_count_value = 0

        if self.pref_builder is not None:
            pref_batch = self.pref_builder.build_fn(
                feature_cache,
                {
                    "alpha": self.alpha,
                    "hyperparams": dict(self.pref_builder_kwargs),
                    **self.pref_builder_kwargs,
                },
            )
            pair_count_value = int(pref_batch.num_examples())
            if pair_count_value > 0:
                loss_batch = pref_batch.to_pairwise_loss_batch(feature_cache)
            else:
                loss_batch = {}
        else:
            loss_batch = {}

        if not loss_batch:
            mask = objective[:, :, None] < objective[:, None, :]
            b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)
            pair_count_value = int(b_idx.numel())
            if pair_count_value > 0:
                if self._pref_batch_cls is None:
                    raise RuntimeError("Preference batch class is not initialized.")
                pref_batch = self._pref_batch_cls(
                    mode="pairwise",
                    pair_idx=(b_idx, winner_idx, loser_idx),
                )
                loss_batch = pref_batch.to_pairwise_loss_batch(feature_cache)

        pair_count = torch.tensor(
            float(pair_count_value), device=reward.device, dtype=reward.dtype
        )

        if pair_count_value == 0:
            advantage = reward - reward.float().mean(dim=1, keepdim=True)
            loss = -(advantage * log_likelihood).mean()
            return loss, pair_count

        if self._pref_evaluate_pairwise_loss is None:
            raise RuntimeError("Pairwise loss evaluator is not initialized.")
        loss = self._pref_evaluate_pairwise_loss(
            self.free_loss,
            full_batch=loss_batch,
            model_output=feature_cache,
            extra={"alpha": self.alpha},
            num_instances=int(objective.shape[0]),
        )
        return loss, pair_count
