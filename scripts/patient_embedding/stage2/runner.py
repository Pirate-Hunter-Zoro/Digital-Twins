"""Stage-2 coordinator.
Initializes embedder, gathers narratives, runs embedding loop."""

from __future__ import annotations
import argparse, os, time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import math

from common.models.patient_embedder import PatientEmbedder

from patient_embedding.shared.io import read_text
from .embed_loop import run_embed_loop
import os
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[3]

def run() -> None:
    # Grab the available narratives
    narratives_dir = Path(os.environ['NARRATIVES_DIR'])
    
    # Grab the relevant patient IDs
    artifacts_path = PROJECT_ROOT / "artifacts"
    num_patients = int(os.environ['NUM_PATIENTS'])
    sampled_ids_path = artifacts_path / f"{num_patients}_patients/sampled_patient_ids.txt"
    with open(sampled_ids_path, 'r') as f:
        # Ignore empty lines
        cohort_ids = [line.strip() for line in f if line.strip()]
        
    ids_to_narrative = {}
    for id in cohort_ids:
        narrative_path = narratives_dir / f"{id}.md"
        if not narrative_path.exists():
            raise ValueError(f"Missing narrative for patient {id}")
        ids_to_narrative[id] = narrative_path
                
    items = zip(ids_to_narrative.keys(), ids_to_narrative.values())
    run_embed_loop(list(items))

    print("Stage 2 complete.", flush=True)