
from .pref_builder_ir import PreferenceBuilderIR, PreferenceBuilderImplementationHint

__all__ = [
    "PreferenceBuilderIR",
    "PreferenceBuilderImplementationHint",
    "CompiledPreferenceBuilder",
    "PreferenceBuilderCompileError",
    "compile_preference_builder",
    "validate_pref_batch",
]


def __getattr__(name: str):
    if name in {
        "CompiledPreferenceBuilder",
        "PreferenceBuilderCompileError",
        "compile_preference_builder",
        "validate_pref_batch",
    }:
        from . import pref_builder_compiler as _pbc

        return getattr(_pbc, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")