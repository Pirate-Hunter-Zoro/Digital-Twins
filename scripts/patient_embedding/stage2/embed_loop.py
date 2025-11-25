"""Per-patient embedding loop.
Embed each narrative, write .npy + audit .embed.txt."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import multiprocessing

import numpy as np
from common.models.patient_embedder import PatientEmbedder
from patient_embedding.shared.io import read_text
from patient_embedding.shared.narrative_parsing import parse_narrative_sections
from patient_embedding.shared.io import write_npy

RECORD_EVERY = 1
VECTOR_DIR = Path(os.environ['VECTORS_DIR'])
FULL_VEC_DIR = VECTOR_DIR / "full"
SUMMARY_VEC_DIR = VECTOR_DIR / "summary"
MEDICATIONS_VEC_DIR = VECTOR_DIR / "medications"
DIAGNOSES_VEC_DIR = VECTOR_DIR / "diagnoses"

def vectors_exist(pid: str) -> bool:
    vec_paths = [FULL_VEC_DIR / f"full_{pid}.npy", SUMMARY_VEC_DIR / f"summary_{pid}.npy", MEDICATIONS_VEC_DIR / f"medications_{pid}.npy", DIAGNOSES_VEC_DIR / f"diagnoses_{pid}.npy"]
    for vec_path in vec_paths:
        if (not vec_path.exists()) or (vec_path.stat().st_size == 0):
            return False
    return True


def _embed_pair(embedder: PatientEmbedder, pids_with_narratives: List[Tuple[str, Path]]):
    # Filter by only non-existant vectors
    indices = [i for i in range(len(pids_with_narratives)) if not vectors_exist(pids_with_narratives[i][0])]
    narratives = [read_text(pids_with_narratives[i][1]) for i in indices]
    sections = parse_narrative_sections(narratives)
    summaries = [section['summary'] for section in sections]
    medications = [section['medications'] for section in sections]
    diagnoses = [section['diagnoses'] for section in sections]
    
    vec_paths_full = [FULL_VEC_DIR / f"full_{pids_with_narratives[i][0]}.npy" for i in indices]
    vec_paths_summary = [SUMMARY_VEC_DIR / f"summary_{pids_with_narratives[i][0]}.npy" for i in indices]
    vec_paths_medications = [MEDICATIONS_VEC_DIR / f"medications_{pids_with_narratives[i][0]}.npy" for i in indices]
    vec_paths_diagnoses = [DIAGNOSES_VEC_DIR / f"diagnoses_{pids_with_narratives[i][0]}.npy" for i in indices]
    
    vecs_full = embedder.vectorize(narratives)
    vecs_summary = embedder.vectorize(summaries)
    vecs_medications = embedder.vectorize(medications)
    vecs_diagnoses = embedder.vectorize(diagnoses)
    for vec_path, vec in zip(vec_paths_full, vecs_full):
        write_npy(vec_path, vec)
    for vec_path, vec in zip(vec_paths_summary, vecs_summary):
        write_npy(vec_path, vec)
    for vec_path, vec in zip(vec_paths_medications, vecs_medications):
        write_npy(vec_path, vec)
    for vec_path, vec in zip(vec_paths_diagnoses, vecs_diagnoses):
        write_npy(vec_path, vec)    
        
def run_embed_loop(items: List[Tuple[str, Path]]):
    print("Loading Embedding Model...", flush=True)
    embedder = PatientEmbedder()
    batch_size = int(os.environ['EMBEDDER_BATCH_SIZE'])
    # Create the batches
    num_batches = int(len(items) / batch_size)
    if num_batches * batch_size < len(items):
        num_batches += 1
    batch_starts = [i*batch_size for i in range(num_batches)] # inclusive
    batch_ends = [min(batch_start + batch_size, len(items)) for batch_start in batch_starts] # non-inclusive
    batches = [items[batch_start: batch_end] for batch_start, batch_end in zip(batch_starts, batch_ends)]
    for i, batch in  enumerate(batches):
        _embed_pair(embedder, batch)
        if ((i+1) % RECORD_EVERY) == 0:
            print(f"Embedded {i+1} patient batches of size {batch_size}", flush=True)