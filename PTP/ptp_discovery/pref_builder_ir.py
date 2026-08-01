from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence


LOGGER = logging.getLogger("ptp_discovery.pref_builder_ir")


@dataclass
class PreferenceBuilderImplementationHint:
    expects: Sequence[str]
    returns: str
    mode: str = "pairwise"


@dataclass
class PreferenceBuilderIR:

    name: str
    intuition: str
    implementation_hint: PreferenceBuilderImplementationHint
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    operators_used: List[str] = field(default_factory=list)
    code: str = ""


def ir_from_json(obj: Mapping[str, Any]) -> PreferenceBuilderIR:
    name = str(obj.get("name", "")).strip()
    intuition = str(obj.get("intuition", "")).strip()
    code = str(obj.get("code", "")).strip()
    hyperparams_raw = obj.get("hyperparams", {}) or {}
    operators_raw = obj.get("operators_used", []) or []

    if not isinstance(hyperparams_raw, dict):
        LOGGER.debug("hyperparams not object; raw=%r", hyperparams_raw)
        hyperparams_raw = {}

    if isinstance(operators_raw, (list, tuple)):
        operators_list = [str(op) for op in operators_raw]
    elif operators_raw is None:
        operators_list = []
    elif isinstance(operators_raw, Mapping):
        operators_list = [str(k) for k in operators_raw.keys()]
    else:
        LOGGER.debug(
            "operators_used not array or Mapping; type=%s, raw=%r",
            type(operators_raw),
            operators_raw,
        )
        operators_list = [str(operators_raw)]

    impl_raw = obj.get("implementation_hint", {}) or {}
    if not isinstance(impl_raw, Mapping):
        LOGGER.debug("implementation_hint not object; raw=%r", impl_raw)
        impl_raw = {}

    expects_raw = impl_raw.get("expects", []) or []
    if isinstance(expects_raw, (list, tuple)):
        expects = [str(x) for x in expects_raw]
    elif expects_raw is None:
        expects = []
    elif isinstance(expects_raw, Mapping):
        expects = [str(k) for k in expects_raw.keys()]
    else:
        LOGGER.debug(
            "implementation_hint.expects not array or Mapping; type=%s, raw=%r",
            type(expects_raw),
            expects_raw,
        )
        expects = [str(expects_raw)]

    returns = str(impl_raw.get("returns", "")).strip()


    impl = PreferenceBuilderImplementationHint(
        expects=expects,
        returns=returns or "PrefBatch",
        mode="pairwise",
    )

    return PreferenceBuilderIR(
        name=name or "unnamed_preference_builder",
        intuition=intuition,
        implementation_hint=impl,
        hyperparams=dict(hyperparams_raw),
        operators_used=operators_list,
        code=code,
    )

