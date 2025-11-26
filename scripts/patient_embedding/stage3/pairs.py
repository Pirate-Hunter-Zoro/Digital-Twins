"""Pair building strategy"""

from __future__ import annotations
import os
import random
from typing import Callable, List, Tuple
from itertools import combinations
from pathlib import Path
from patient_embedding.shared.plots import histogram
from dotenv import load_dotenv

load_dotenv()

def pair_id(a: str, b: str) -> str:
    """Creates a canonical, order-independent ID for a pair."""
    # Sort the patient IDs to ensure (a,b) and (b,a) produce the same key.
    sorted_pair = tuple(sorted((str(a), str(b))))
    return f"{sorted_pair[0]}:{sorted_pair[1]}"

def find_closest_pair(target_cos_val: float, pairs_by_similarity: list[tuple[str,str,float,float,float]]) -> tuple[str,str,float]:
    """Search for the closest pair which is not yet picked.
    Note that the list is sorted so while we can get a slight performance boost with binary search.
    BUT since we must remove whatever element we find, this will still be a linear operation so don't get too excited..."""
    left = 0
    right = len(pairs_by_similarity)
    while left < right:
        mid = (left + right) // 2
        if pairs_by_similarity[mid][2] < target_cos_val:
            # Look right
            left = mid+1
        else:
            # Note that the odds of equality are essentially zero with floats so just assume this means greater
            # Look left
            right = mid
    
    record_spot = None
    record = float('inf')
    for spot in [left-1,left,left+1]:
        if spot >= 0 and spot < len(pairs_by_similarity):
            dist = abs(pairs_by_similarity[spot][2]-target_cos_val)
            if dist < record:
                record = dist
                record_spot = spot
    record_pair = pairs_by_similarity[record_spot]
    pairs_by_similarity.pop(record_spot)
    return record_pair

PROJECT_ROOT = Path(__file__).resolve().parents[3]

def build_pairs(
    rnd: random.Random, 
    cos_func: Callable[[str, str], float]
) -> List[Tuple[str, str, float]]:
    """
    Primary pair-building function.
    """
    artifacts_path = PROJECT_ROOT / "artifacts"
    artifacts_path.mkdir(parents=True, exist_ok=True)
    num_patients = int(os.environ['NUM_PATIENTS'])
    sampled_ids_path = artifacts_path / f"{num_patients}_patients/sampled_patient_ids.txt"
    with open(sampled_ids_path, 'r') as f:
        patient_ids = [line.strip() for line in f.readlines()]
        all_possible_pairs = list(combinations(patient_ids, 2))
        print(f"[Stage3] Calculating cosine similarity for {len(all_possible_pairs)} unique pairs...")

        pairs_by_similarity = []
        for i, (a, b) in enumerate(all_possible_pairs):
            if i > 0 and i % 5000 == 0:
                print(f"[Stage3] ...scanned {i} / {len(all_possible_pairs)} pairs")
            pairs_by_similarity.append((a, b, cos_func(a,b)))
        
        # Create histogram of all cosine values
        similarities = [pair[2] for pair in pairs_by_similarity]
        histogram(similarities, f"Cosine Similarity of All Pairs ({str(cos_func)})", Path(os.environ['ANALYSIS_DIR']) / f"{str(cos_func)}_cos_sim_all_pairs")
        
        pairs_by_similarity.sort(key = lambda x: x[2])
        lowest = pairs_by_similarity[0][2]
        highest = pairs_by_similarity[-1][2]
        selected_pairs = []
        for _ in range(min(int(os.environ['NUM_PAIRS']),len(pairs_by_similarity))):
            v = rnd.uniform(lowest,highest)
            pair = find_closest_pair(v, pairs_by_similarity)
            selected_pairs.append(pair)

        return selected_pairs