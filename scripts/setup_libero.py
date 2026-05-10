#!/usr/bin/env python
"""
One-time LIBERO initialization: config file + asset files.

Run before training to avoid two silent failures:
  1. Interactive prompt on first import (missing ~/.libero/config.yaml)
  2. Eval crash when simulator can't find XML scenes (assets not in pip package)

Both are one-time costs — subsequent runs skip everything that's already done.

Usage:
    uv run python scripts/setup_libero.py
"""
import importlib.util
import os
import pathlib
import yaml

# ---- locate the libero.libero package WITHOUT importing it ------------------
# Importing libero.libero triggers the interactive prompt at module level if
# config.yaml doesn't exist yet.
spec = importlib.util.find_spec("libero.libero")
if spec is None:
    raise RuntimeError("libero package not found — run `uv sync` first")
libero_dir = pathlib.Path(spec.origin).parent  # .../site-packages/libero/libero/

# ---- 1. write config.yaml ---------------------------------------------------
# Suppresses "Do you want to specify a custom path?" prompt on first import.
# All paths point to files bundled inside the package (bddl, init_states)
# or downloaded below (assets). Nothing goes to the NFS home directory.
cfg_dir = pathlib.Path(
    os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
)
cfg = cfg_dir / "config.yaml"
cfg.parent.mkdir(parents=True, exist_ok=True)

if cfg.exists():
    print(f"LIBERO config already exists: {cfg}")
else:
    config = {
        "benchmark_root": str(libero_dir),
        "bddl_files":     str(libero_dir / "bddl_files"),
        "init_states":    str(libero_dir / "init_files"),
        "datasets":       str(libero_dir / "../datasets"),
        "assets":         str(libero_dir / "assets"),
    }
    cfg.write_text(yaml.dump(config))
    print(f"LIBERO config written to: {cfg}")

# ---- 2. download assets if missing ------------------------------------------
# The pip package ships an empty assets/ directory. The sim needs XML scene
# files and 3D meshes from lerobot/libero-assets. We download directly into
# the venv's assets/ dir (already on $SCRATCH) so nothing touches NFS home.
assets_dir = libero_dir / "assets"
key_asset = assets_dir / "scenes" / "libero_tabletop_base_style.xml"

if key_asset.exists():
    print(f"LIBERO assets already present: {assets_dir}")
else:
    print("LIBERO assets missing — downloading from lerobot/libero-assets (~10 min)...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="lerobot/libero-assets",
        repo_type="dataset",
        local_dir=str(assets_dir),
    )
    print(f"LIBERO assets downloaded to: {assets_dir}")
