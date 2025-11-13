"""Stage-1 coordinator.
Validates env/paths, initializes logger, constructs cohort, and invokes the threadpool."""

from __future__ import annotations
import argparse, os, time
from pathlib import Path
from typing import Any, Dict, List
import math

from patient_embedding.shared.io import ensure_dir, nonempty
from .threadpool import run_threadpool_stage

def run() -> None:
    out_dir = Path(os.environ['NARRATIVES_DIR'])
    ensure_dir(out_dir)

    # Perform a pre-flight check BEFORE loading all data.
    num_patients = int(os.environ['NUM_PATIENTS'])
    
    # Count existing, non-empty .md files in the output directory.
    existing_md_files = [p for p in out_dir.glob("*.md") if nonempty(p)]
    num_existing = len(existing_md_files)

    print(f"Date:   {time.asctime()}", flush=True)
    print(f"Outdir: {out_dir}", flush=True)
    print(f"[Stage1] Found {num_existing} existing narratives. Requesting {max(0, num_patients - num_existing)} more...")

    # If the work is already done, exit immediately.
    if num_existing >= num_patients:
        print(f"[Stage1] Sufficient narratives already exist. Skipping generation.")
        print("[Stage1] complete.", flush=True)
        return

    print("=== Stage 1: Forge Narratives ===", flush=True)
    print(f"Outdir: {out_dir}", flush=True)

    # Create and store the narratives
    run_threadpool_stage()

    print("[Stage1] complete.", flush=True)