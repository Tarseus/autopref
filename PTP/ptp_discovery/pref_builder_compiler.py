from __future__ import annotations
import ast
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence
import torch
from fitness.free_loss_fidelity import PrefBatch
from .free_loss_compiler import _OpsAccessor, _build_operator_table
from .pref_builder_ir import PreferenceBuilderIR

BuildFn = Callable[[Mapping[str, torch.Tensor], Optional[Mapping[str, Any]]], PrefBatch]

class PreferenceBuilderCompileError(Exception):
    pass

@dataclass
class CompiledPreferenceBuilder:
    ir: PreferenceBuilderIR
    build_fn: BuildFn

class _BuilderCodeValidator(ast.NodeVisitor):
    _ALLOWED_NAMES = {'float', 'int', 'bool', 'len', 'min', 'max', 'abs', 'PrefBatch'}
    _FORBIDDEN_NODES = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Lambda, ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.With, ast.AsyncWith, ast.Try, ast.Raise, ast.Delete, ast.Global, ast.Nonlocal, ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr, ast.AugAssign, ast.AnnAssign)

    def __init__(self, allowed_ops: Sequence[str]) -> None:
        super().__init__()
        self.allowed_ops = {str(name) for name in allowed_ops}

    def visit_Module(self, node: ast.Module) -> None:
        if len(node.body) != 1 or not isinstance(node.body[0], ast.FunctionDef):
            raise PreferenceBuilderCompileError('Builder code must contain only generated_builder.')
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name != 'generated_builder' or node.decorator_list:
            raise PreferenceBuilderCompileError('Builder code must define only generated_builder.')
        args = [arg.arg for arg in node.args.args]
        if args != ['feature_cache', 'extra'] or node.args.vararg or node.args.kwarg or node.args.kwonlyargs:
            raise PreferenceBuilderCompileError('generated_builder must accept exactly feature_cache and extra.')
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith('__') or node.id in {'torch', 'F', 'eval', 'exec', 'compile', 'open', 'getattr', 'setattr', 'delattr', 'globals', 'locals', 'vars', '__import__'}:
            raise PreferenceBuilderCompileError(f'Builder code uses forbidden name {node.id!r}.')
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith('_'):
            raise PreferenceBuilderCompileError('Builder code cannot access private attributes.')
        if isinstance(node.value, ast.Name) and node.value.id in {'ops', 'operators'}:
            if node.attr not in self.allowed_ops:
                raise PreferenceBuilderCompileError(f'Builder operator {node.attr!r} is not allowed.')
        elif node.attr not in {'get', 'nonzero'}:
            raise PreferenceBuilderCompileError(f'Builder attribute {node.attr!r} is not allowed.')
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id not in self._ALLOWED_NAMES:
            raise PreferenceBuilderCompileError(f'Builder call {node.func.id!r} is not allowed.')
        if not isinstance(node.func, (ast.Name, ast.Attribute)):
            raise PreferenceBuilderCompileError('Builder code uses an unsupported callable.')
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        def valid_target(target: ast.AST) -> bool:
            if isinstance(target, ast.Name):
                return target.id not in {'feature_cache', 'extra', 'ops', 'operators', 'PrefBatch'}
            if isinstance(target, (ast.Tuple, ast.List)):
                return all(valid_target(item) for item in target.elts)
            return False
        if not all(valid_target(target) for target in node.targets):
            raise PreferenceBuilderCompileError('Builder assignments may target only local names.')
        self.generic_visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, self._FORBIDDEN_NODES):
            raise PreferenceBuilderCompileError(f'Builder syntax {type(node).__name__} is not allowed.')
        super().generic_visit(node)

def _validate_builder_code(code: str, allowed_ops: Sequence[str]) -> None:
    try:
        tree = ast.parse(code, mode='exec')
    except SyntaxError as exc:
        raise PreferenceBuilderCompileError(f'Builder code has syntax error: {exc}') from exc
    _BuilderCodeValidator(allowed_ops).visit(tree)

def validate_pref_batch(pref_batch: PrefBatch, feature_cache: Mapping[str, torch.Tensor]) -> None:
    if not isinstance(pref_batch, PrefBatch):
        raise ValueError(f'pref_batch must be PrefBatch; got type={type(pref_batch)}')
    objective = feature_cache.get('objective')
    log_prob = feature_cache.get('log_prob')
    if not isinstance(objective, torch.Tensor) or not isinstance(log_prob, torch.Tensor):
        raise ValueError('feature_cache must contain objective and log_prob tensors')
    if objective.ndim != 2 or log_prob.shape != objective.shape:
        raise ValueError('objective and log_prob must have matching (B,K) shapes')
    b_idx, winner_idx, loser_idx = pref_batch.pair_idx
    for name, tensor in (('b_idx', b_idx), ('winner_idx', winner_idx), ('loser_idx', loser_idx)):
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.long or tensor.ndim != 1:
            raise ValueError(f'pair_idx {name} must be a one-dimensional long tensor')
    if not (b_idx.numel() == winner_idx.numel() == loser_idx.numel()):
        raise ValueError('pair_idx tensors must have the same length')
    batch_size, solution_count = int(objective.shape[0]), int(objective.shape[1])
    if b_idx.numel() > 0:
        if int(b_idx.min().item()) < 0 or int(b_idx.max().item()) >= batch_size:
            raise ValueError('pair_idx batch indices out of range')
        if int(winner_idx.min().item()) < 0 or int(winner_idx.max().item()) >= solution_count or int(loser_idx.min().item()) < 0 or int(loser_idx.max().item()) >= solution_count:
            raise ValueError('pair_idx solution indices out of range')
    weight = pref_batch.weight
    if weight is not None:
        if not isinstance(weight, torch.Tensor) or weight.ndim != 1 or weight.numel() != b_idx.numel():
            raise ValueError('weight must be one-dimensional and match pair_idx')
        if not torch.isfinite(weight).all().item():
            raise ValueError('weight must be finite')

def compile_preference_builder(ir: PreferenceBuilderIR, *, operator_whitelist: Sequence[str] | None=None) -> CompiledPreferenceBuilder:
    code = (ir.code or '').strip()
    if not code:
        raise PreferenceBuilderCompileError('PreferenceBuilderIR.code is empty.')
    table = _build_operator_table()
    if operator_whitelist:
        table = {name: value for name, value in table.items() if name in operator_whitelist}
    _validate_builder_code(code, tuple(table))
    accessor = _OpsAccessor(table)
    safe_globals: Dict[str, Any] = {'__builtins__': {'float': float, 'int': int, 'bool': bool, 'len': len, 'min': min, 'max': max, 'abs': abs}, 'ops': accessor, 'operators': accessor, 'PrefBatch': PrefBatch}
    local_ns: Dict[str, Any] = {}
    try:
        exec(code, safe_globals, local_ns)
    except Exception as exc:
        raise PreferenceBuilderCompileError(f'Failed to exec builder code from IR: {exc}') from exc
    fn = local_ns.get('generated_builder')
    if not callable(fn):
        raise PreferenceBuilderCompileError('Builder code did not define generated_builder.')
    def build_fn(feature_cache: Mapping[str, torch.Tensor], extra: Mapping[str, Any] | None) -> PrefBatch:
        merged_extra: Dict[str, Any] = {'hyperparams': dict(ir.hyperparams), 'ops': accessor, 'operators': accessor}
        if extra:
            merged_extra.update({key: value for key, value in extra.items() if key not in {'torch', 'F', 'torch.nn.functional'}})
        out = fn(feature_cache, merged_extra)
        validate_pref_batch(out, feature_cache)
        return out
    return CompiledPreferenceBuilder(ir=ir, build_fn=build_fn)
