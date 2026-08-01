from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F

from .free_loss_ir import FreeLossIR, ir_from_json


LossFn = Callable[[Mapping[str, Any], Mapping[str, torch.Tensor], Mapping[str, Any]], torch.Tensor]


class CompileError(Exception):
    pass


@dataclass
class CompiledFreeLoss:
    ir: FreeLossIR
    loss_fn: LossFn


def _extract_json_object(text: str) -> Mapping[str, Any]:

    start = text.find("{")
    if start == -1:
        raise CompileError("No JSON object found in LLM output.")

    depth = 0
    end = None
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end is None or end <= start:
        raise CompileError("Failed to locate a complete JSON object in LLM output.")

    snippet = text[start : end + 1]

    invalid_escape_pattern = re.compile(r'\\(?!["\\/bfnrtu])')
    sanitized_snippet = invalid_escape_pattern.sub(r"\\\\", snippet)

    def _escape_control_chars_in_strings(s: str) -> str:
        out_chars: list[str] = []
        in_string = False
        escape = False
        for ch in s:
            if escape:
                out_chars.append(ch)
                escape = False
                continue
            if ch == "\\":
                out_chars.append(ch)
                escape = True
                continue
            if ch == '"':
                out_chars.append(ch)
                in_string = not in_string
                continue
            if in_string and ch in ("\n", "\r", "\t"):
                if ch == "\n":
                    out_chars.append("\\n")
                elif ch == "\r":
                    out_chars.append("\\r")
                elif ch == "\t":
                    out_chars.append("\\t")
                continue
            out_chars.append(ch)
        return "".join(out_chars)

    sanitized_snippet = _escape_control_chars_in_strings(sanitized_snippet)

    try:
        return json.loads(sanitized_snippet)
    except json.JSONDecodeError as exc:
        raise CompileError(f"Failed to parse JSON from LLM output: {exc}") from exc


class _SafeCodeValidator(ast.NodeVisitor):

    _FORBIDDEN_CALL_NAMES = {
        "__import__",
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
    }

    _FORBIDDEN_ATTR_BASES = {
        "os",
        "sys",
        "subprocess",
        "socket",
        "pathlib",
    }
    _DISALLOWED_TENSOR_METHOD_CALLS = {
        "max",
        "min",
        "maximum",
        "minimum",
        "norm",
        "sign",
    }

    _PAIR_COUPLING_CALLS = {
        "mean",
        "sum",
        "std",
        "var",
        "normalize",
        "zscore",
        "stack",
        "cat",
        "max",
        "min",
        "norm",
        "rank_gap",
    }

    def __init__(self, *, elementwise_pair: bool = False) -> None:
        super().__init__()
        self.elementwise_pair = bool(elementwise_pair)

    _FORBIDDEN_COMPLEXITY_NODES = (
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )

    def visit_Import(self, node: ast.Import) -> None:
        raise CompileError("Loss code must not use import statements.")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise CompileError("Loss code must not use import-from statements.")

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in self._FORBIDDEN_CALL_NAMES:
            raise CompileError(f"Loss code calls forbidden function '{func.id}'.")
        if self.elementwise_pair and isinstance(func, ast.Name) and func.id in self._PAIR_COUPLING_CALLS:
            raise CompileError(
                f"Pairwise loss code must be elementwise; '{func.id}' couples pairs."
            )
        if isinstance(func, ast.Attribute):
            base = func.value
            if self.elementwise_pair and func.attr != "get" and not (isinstance(base, ast.Name) and base.id == "ops"):
                raise CompileError("Pairwise loss code may call only ops.* math and mapping.get(...).")
            if self.elementwise_pair and func.attr in self._PAIR_COUPLING_CALLS:
                raise CompileError(
                    f"Pairwise loss code must be elementwise; '{func.attr}' couples pairs."
                )
            if isinstance(base, ast.Name) and base.id in self._FORBIDDEN_ATTR_BASES:
                raise CompileError(
                    f"Loss code must not access '{base.id}.{func.attr}'. "
                    "Only ops.* math is allowed."
                )
            if (
                func.attr in self._DISALLOWED_TENSOR_METHOD_CALLS
                and not (isinstance(base, ast.Name) and base.id == "ops")
            ):
                raise CompileError(
                    f"Loss code must use 'ops.{func.attr}(...)' instead of tensor method '.{func.attr}(...)'."
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith('_'):
            raise CompileError('Loss code cannot access private attributes.')
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if not self.elementwise_pair:
            self.generic_visit(node)
            return
        allowed_mapping = isinstance(node.value, ast.Name) and node.value.id in {
            "batch",
            "model_output",
            "extra",
        }
        string_key = isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)
        if not (allowed_mapping and string_key):
            raise CompileError("Pairwise loss code must not index pair tensors.")
        self.generic_visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, self._FORBIDDEN_COMPLEXITY_NODES):
            raise CompileError(
                "Loss/builder code must be vectorized: Python loops/comprehensions are not allowed."
            )
        super().generic_visit(node)


def _validate_user_code(code_str: str, *, elementwise_pair: bool = False) -> None:

    try:
        tree = ast.parse(code_str, mode="exec")
    except SyntaxError as exc:
        raise CompileError(f"Loss code has syntax error: {exc}") from exc

    validator = _SafeCodeValidator(elementwise_pair=elementwise_pair)
    validator.visit(tree)


def _safe_normalize(
    x: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
    **kwargs: Any,
) -> torch.Tensor:
    if "epsilon" in kwargs and kwargs["epsilon"] is not None:
        eps = float(kwargs["epsilon"])
    if "eps" in kwargs and kwargs["eps"] is not None:
        eps = float(kwargs["eps"])
    if "dim" in kwargs and kwargs["dim"] is not None:
        dim = int(kwargs["dim"])
    keepdim = kwargs.get("keepdim", True)
    x = x - x.mean(dim=dim, keepdim=bool(keepdim))
    std = x.std(dim=dim, keepdim=bool(keepdim))
    return x / (std + eps)


def _safe_zscore(x: torch.Tensor, eps: float = 1e-8, **kwargs: Any) -> torch.Tensor:
    if "epsilon" in kwargs and kwargs["epsilon"] is not None:
        eps = float(kwargs["epsilon"])
    if "eps" in kwargs and kwargs["eps"] is not None:
        eps = float(kwargs["eps"])
    dim = kwargs.get("dim")
    keepdim = kwargs.get("keepdim", True)
    if dim is None:
        mean = x.mean()
        std = x.std()
    else:
        mean = x.mean(dim=dim, keepdim=bool(keepdim))
        std = x.std(dim=dim, keepdim=bool(keepdim))
    return (x - mean) / (std + eps)


def _rank_gap(cost_a: torch.Tensor, cost_b: torch.Tensor) -> torch.Tensor:
    return cost_b - cost_a


def _tensor_reduce_max(
    x: torch.Tensor,
    *,
    dim: int | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    if dim is None:
        return torch.max(x)
    values, _ = torch.max(x, dim=int(dim), keepdim=bool(keepdim))
    return values


def _tensor_reduce_min(
    x: torch.Tensor,
    *,
    dim: int | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    if dim is None:
        return torch.min(x)
    values, _ = torch.min(x, dim=int(dim), keepdim=bool(keepdim))
    return values


def _ops_max(
    x: torch.Tensor,
    other: torch.Tensor | float | int | None = None,
    *,
    dim: int | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    if other is not None:
        if not isinstance(other, torch.Tensor):
            other = torch.as_tensor(other, dtype=x.dtype, device=x.device)
        return torch.maximum(x, other)
    return _tensor_reduce_max(x, dim=dim, keepdim=keepdim)


def _ops_min(
    x: torch.Tensor,
    other: torch.Tensor | float | int | None = None,
    *,
    dim: int | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    if other is not None:
        if not isinstance(other, torch.Tensor):
            other = torch.as_tensor(other, dtype=x.dtype, device=x.device)
        return torch.minimum(x, other)
    return _tensor_reduce_min(x, dim=dim, keepdim=keepdim)


def _ops_norm(
    x: torch.Tensor,
    p: float | int = 2,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    if "ord" in kwargs and kwargs["ord"] is not None:
        p = kwargs["ord"]
    if "axis" in kwargs and kwargs["axis"] is not None:
        dim = kwargs["axis"]
    if "dim" in kwargs and kwargs["dim"] is not None:
        dim = kwargs["dim"]
    if "keepdims" in kwargs and kwargs["keepdims"] is not None:
        keepdim = bool(kwargs["keepdims"])
    if "keepdim" in kwargs and kwargs["keepdim"] is not None:
        keepdim = bool(kwargs["keepdim"])
    kwargs: Dict[str, Any] = {"p": p}
    if dim is not None:
        kwargs["dim"] = dim
        kwargs["keepdim"] = bool(keepdim)
    return torch.norm(x, **kwargs)


def _ops_maximum(a: torch.Tensor, b: torch.Tensor | float | int) -> torch.Tensor:
    if not isinstance(b, torch.Tensor):
        b = torch.as_tensor(b, dtype=a.dtype, device=a.device)
    return torch.maximum(a, b)


def _ops_minimum(a: torch.Tensor, b: torch.Tensor | float | int) -> torch.Tensor:
    if not isinstance(b, torch.Tensor):
        b = torch.as_tensor(b, dtype=a.dtype, device=a.device)
    return torch.minimum(a, b)


def _build_operator_table() -> Dict[str, Callable[..., torch.Tensor]]:
    return {
        "logsigmoid": F.logsigmoid,
        "softplus": F.softplus,
        "sigmoid": torch.sigmoid,
        "exp": torch.exp,
        "log": torch.log,
        "abs": torch.abs,
        "tanh": torch.tanh,
        "relu": F.relu,
        "sum": torch.sum,
        "mean": torch.mean,
        "stack": torch.stack,
        "cat": torch.cat,
        "ones_like": torch.ones_like,
        "zeros_like": torch.zeros_like,
        "sqrt": lambda x: torch.sqrt(torch.clamp(x, min=1e-8)),
        "pow": torch.pow,
        "add": torch.add,
        "sub": torch.sub,
        "mul": torch.mul,
        "div": lambda a, b: torch.div(a, torch.clamp(b, min=1e-8)),
        "neg": torch.neg,
        "clamp": lambda x, min=-10.0, max=10.0: torch.clamp(x, min=min, max=max),
        "normalize": _safe_normalize,
        "zscore": _safe_zscore,
        "rank_gap": _rank_gap,
        "max": _ops_max,
        "min": _ops_min,
        "maximum": _ops_maximum,
        "minimum": _ops_minimum,
        "sign": torch.sign,
        "norm": _ops_norm,
    }


class _OpsAccessor:
    def __init__(self, table: Dict[str, Callable[..., torch.Tensor]]) -> None:
        self._table = dict(table)

    def __getattr__(self, name: str) -> Callable[..., torch.Tensor]:
        try:
            return self._table[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, name: str) -> Callable[..., torch.Tensor]:
        return self._table[name]


def parse_free_loss_from_text(text: str) -> FreeLossIR:
    obj = _extract_json_object(text)
    return ir_from_json(obj)


def compile_free_loss(ir: FreeLossIR, *, operator_whitelist: Sequence[str] | None = None) -> CompiledFreeLoss:

    code_str = (ir.code or "").strip()
    if not code_str:
        raise CompileError("Loss IR must provide generated_loss code.")
    returns = str(ir.implementation_hint.returns or "").strip().lower()
    expects = {str(key) for key in (ir.implementation_hint.expects or [])}
    if returns not in {"per_pair", "pairwise", "one_per_pair"}:
        raise CompileError("Pairwise loss implementation_hint.returns must be 'per_pair'.")
    if "weight" in expects or "instance_idx" in expects:
        raise CompileError("Pairwise loss programs must not read framework aggregation fields.")
    if code_str:
        _validate_user_code(code_str, elementwise_pair=True)

        ops_table = _build_operator_table()
        if operator_whitelist:
            ops_table = {k: v for k, v in ops_table.items() if k in operator_whitelist}
        ops_accessor = _OpsAccessor(ops_table)

        safe_globals: Dict[str, Any] = {
            '__builtins__': {'float': float, 'int': int, 'bool': bool, 'min': min, 'max': max, 'abs': abs, 'len': len},
            'ops': ops_accessor,
        }
        local_ns: Dict[str, Any] = {}
        try:
            exec(code_str, safe_globals, local_ns)
        except Exception as exc:
            raise CompileError(f"Failed to exec loss code from IR: {exc}") from exc

        fn = local_ns.get("generated_loss")
        if not callable(fn):
            raise CompileError(
                "Loss code did not define a callable 'generated_loss(batch, model_output, extra)'."
            )

        def loss_fn(
            batch: Mapping[str, Any],
            model_output: Mapping[str, torch.Tensor],
            extra: Mapping[str, Any] | None,
        ) -> torch.Tensor:
            merged_extra: Dict[str, Any] = {'hyperparams': dict(ir.hyperparams), 'ops': ops_accessor, 'operators': ops_accessor}
            if extra:
                merged_extra.update({key: value for key, value in extra.items() if key not in {'torch', 'F', 'torch.nn.functional'}})
            merged_model_output: Dict[str, torch.Tensor] = dict(model_output or {})
            for key in ("log_prob_w", "log_prob_l"):
                value = batch.get(key) if isinstance(batch, Mapping) else None
                if key not in merged_model_output and isinstance(value, torch.Tensor):
                    merged_model_output[key] = value
            return fn(batch, merged_model_output, merged_extra)

    return CompiledFreeLoss(ir=ir, loss_fn=loss_fn)
