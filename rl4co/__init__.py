from importlib.metadata import PackageNotFoundError, version as get_version

# The package version is obtained from package metadata when installed (e.g. `pip install -e .`).
# When running from a source checkout without installation, the metadata may be missing.
try:
    __version__ = get_version(__package__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
