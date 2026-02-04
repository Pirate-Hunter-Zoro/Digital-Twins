import sqlite3
import os
import random

from scripts.digital_twins.predictions.trd_predictor import TRDPredictor
from scripts.digital_twins.neighbors.retriever import Retriever

from dotenv import load_dotenv
load_dotenv()

def main():
    retriever = Retriever()
    pair_ids = set()
    all_hash_ids = retriever.ids
    all_vectors = retriever.vectors
    pairs = []
    for _ in range(int(os.environ['NUM_PAIRS_SANITY_CHECK'])):
        pair_indices = random.sample(range(len(all_hash_ids)), 2)
        pair_indices = (pair_indices[0], pair_indices[1]) if pair_indices[0] < pair_indices[1] else (pair_indices[1], pair_indices[0])
        while pair_indices in pair_ids:
            pair_indices = random.sample(range(len(all_hash_ids)), 2)
            pair_indices = (pair_indices[0], pair_indices[1]) if pair_indices[0] < pair_indices[1] else (pair_indices[1], pair_indices[0])  
        pair_ids.add(pair_indices)
        first_idx = pair_indices[0]
        second_idx = pair_indices[1]
        pairs.append(((all_hash_ids[first_idx], all_vectors[first_idx]), (all_hash_ids[second_idx], all_vectors[second_idx])))