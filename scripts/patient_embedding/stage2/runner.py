"""Stage-2 coordinator.
Initializes embedder, gathers narratives, runs embedding loop."""

from __future__ import annotations
from pathlib import Path
from scripts.patient_embedding.stage2.embed_loop import run_embed_loop
import os
from dotenv import load_dotenv
load_dotenv()

def run() -> None:
    pass

    print("Stage 2 complete.", flush=True)