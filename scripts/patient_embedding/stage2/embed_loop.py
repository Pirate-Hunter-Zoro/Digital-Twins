"""Per-patient embedding loop.
Embed each narrative, write .npy + audit .embed.txt."""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple
from scripts.common.models.string_embedder import StringEmbedder
from patient_embedding.shared.io import read_text
from patient_embedding.shared.io import write_npy

RECORD_EVERY = 1
VECTOR_DIR = Path(os.environ['VECTORS_DIR'])

def vectors_exist(pid: str) -> bool:
    vec_path = VECTOR_DIR / f"{pid}.npy"
    return vec_path.exists() and vec_path.stat().st_size == 0

def _embed_pair(embedder: StringEmbedder, pids_with_narratives: List[Tuple[str, Path]]):
    # Filter by only non-existant vectors
    indices = [i for i in range(len(pids_with_narratives)) if not vectors_exist(pids_with_narratives[i][0])]
    narratives = [read_text(pids_with_narratives[i][1]) for i in indices]
    
    vec_paths = [VECTOR_DIR / f"{pids_with_narratives[i][0]}.npy" for i in indices]
    
    vecs = embedder.vectorize(narratives)
    for vec_path, vec in zip(vec_paths, vecs):
        write_npy(vec_path, vec)
        
def run_embed_loop(items: List[Tuple[str, Path]]):
    print("Loading Embedding Model...", flush=True)
    embedder = StringEmbedder()
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