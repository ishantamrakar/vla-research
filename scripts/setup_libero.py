#!/usr/bin/env python
"""
One-time LIBERO config initialization.

LIBERO prompts interactively on first import if ~/.libero/config.yaml
doesn't exist, which hangs non-interactive jobs. Run this once before
training to write the config. All paths point to files bundled inside
the LIBERO package — nothing is downloaded.

Usage:
    uv run python scripts/setup_libero.py
"""
import os
import pathlib
import yaml

cfg_dir = pathlib.Path(os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero")))
cfg = cfg_dir / "config.yaml"
cfg.parent.mkdir(parents=True, exist_ok=True)

if cfg.exists():
    print(f"LIBERO config already exists: {cfg}")
else:
    from libero.libero import get_default_path_dict
    cfg.write_text(yaml.dump(get_default_path_dict()))
    print(f"LIBERO config written to: {cfg}")
