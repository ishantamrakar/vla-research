#!/usr/bin/env python
"""
One-time LIBERO config initialization.

LIBERO prompts interactively on first import if ~/.libero/config.yaml
doesn't exist, which hangs non-interactive jobs. Run this once before
training to write the config. All paths point to files bundled inside
the LIBERO package — nothing is downloaded.

Uses importlib.util.find_spec to locate the package WITHOUT importing it,
so the interactive prompt never triggers.

Usage:
    uv run python scripts/setup_libero.py
"""
import importlib.util
import os
import pathlib
import yaml

cfg_dir = pathlib.Path(os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero")))
cfg = cfg_dir / "config.yaml"
cfg.parent.mkdir(parents=True, exist_ok=True)

if cfg.exists():
    print(f"LIBERO config already exists: {cfg}")
else:
    # Locate libero.libero package directory without importing it
    # (importing triggers the interactive prompt at module level)
    spec = importlib.util.find_spec("libero.libero")
    if spec is None:
        raise RuntimeError("libero package not found — run `uv sync` first")
    libero_dir = pathlib.Path(spec.origin).parent

    config = {
        "benchmark_root": str(libero_dir),
        "bddl_files": str(libero_dir / "bddl_files"),
        "init_states": str(libero_dir / "init_files"),
        "datasets": str(libero_dir / "../datasets"),
        "assets": str(libero_dir / "assets"),
    }
    cfg.write_text(yaml.dump(config))
    print(f"LIBERO config written to: {cfg}")
