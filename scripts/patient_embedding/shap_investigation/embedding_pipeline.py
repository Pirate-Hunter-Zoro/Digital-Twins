from scripts.patient_embedding.shap_investigation.data_parser import parse_test_narratives
from scripts.common.models.patient_embedder import PatientEmbedder
from scripts.patient_embedding.shared.io import write_npy
from typing import Dict
import numpy as np
import pickle
import os
import multiprocessing
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CACHE_PATH = Path(os.environ['SHAP_DIR']) / "vector_components.pkl"
VECTORS_PATH = Path(os.environ['VECTORS_DIR'])
FULL_VEC_DIR = VECTORS_PATH / "full"
SUMMARY_VEC_DIR = VECTORS_PATH / "summary"
MEDICATIONS_VEC_DIR = VECTORS_PATH / "medications"
DIAGNOSES_VEC_DIR = VECTORS_PATH / "diagnoses"

def embed_batch(batch_ids: list[str], vector_components: Dict[str, Dict[str, np.array]], string_components: Dict[str, Dict[str, str]], embedder: PatientEmbedder):
    batch_ids_missing_vector = []
    for id in batch_ids:
        vector_paths = [VECTORS_PATH / f"{label}" / f"{label}_{id}" for label in string_components[id].keys()]
        for vector_path in vector_paths:
            if not vector_path.exists():
                batch_ids_missing_vector.append(id)
            else:
                vector_components[id]['full'] = np.load(FULL_VEC_DIR / f'full_{id}.npy')
                vector_components[id]['summary'] = np.load(SUMMARY_VEC_DIR / f'summary_{id}.npy')
                vector_components[id]['medications'] = np.load(MEDICATIONS_VEC_DIR / f'medications_{id}.npy')
                vector_components[id]['diagnoses'] = np.load(DIAGNOSES_VEC_DIR / f'diagnoses_{id}.npy') 
    
    if len(batch_ids_missing_vector) > 0:
        full_texts_missing = [string_components[id]['full'] for id in batch_ids_missing_vector]
        summarys_missing = [string_components[id]['summary'] for id in batch_ids_missing_vector]
        medications_lists_missing = [string_components[id]['medications'] for id in batch_ids_missing_vector]
        diagnoses_lists_missing = [string_components[id]['diagnoses'] for id in batch_ids_missing_vector]
        
        full_text_vectors_missing = embedder.vectorize(full_texts_missing)
        summarys_vectors_missing = embedder.vectorize(summarys_missing)
        medications_lists_vectors_missing = embedder.vectorize(medications_lists_missing)
        diagnoses_lists_vectors_missing = embedder.vectorize(diagnoses_lists_missing)
    
        for i,id in enumerate(batch_ids_missing_vector):
            # Store the vectors in a dictionary as well as a file
            vector_components[id]['full'] = full_text_vectors_missing[i]
            write_npy(FULL_VEC_DIR / f'full_{id}.npy', full_text_vectors_missing[i])
            vector_components[id]['summary'] = summarys_vectors_missing[i]
            write_npy(SUMMARY_VEC_DIR / f'summary_{id}.npy', summarys_vectors_missing[i])
            vector_components[id]['medications'] = medications_lists_vectors_missing[i]
            write_npy(MEDICATIONS_VEC_DIR / f'medications_{id}.npy', medications_lists_vectors_missing[i])
            vector_components[id]['diagnoses'] = diagnoses_lists_vectors_missing[i]
            write_npy(DIAGNOSES_VEC_DIR / f'diagnoses_{id}.npy', diagnoses_lists_vectors_missing[i])

def forge_test_vectors(batch_size: int=4) -> Dict[str, Dict[str, np.array]]:
    # Convert all strings into vectors
    
    if CACHE_PATH.exists():
        # Already did the work
        with open(CACHE_PATH, 'rb') as f:
            return pickle.load(f)
    
    embedder = PatientEmbedder()
    string_components = parse_test_narratives()
    ids = list(string_components.keys())
    left = 0
    right = min(batch_size-1, len(ids)-1)
    batches_by_id = []
    while left < len(ids):
        batch_ids = ids[left:right+1]
        batches_by_id.append(batch_ids)
        left = right + 1
        right = min(left + batch_size - 1, len(ids)-1)
    
    vector_components = {id: {} for id in ids}
    for i, batch_ids in enumerate(batches_by_id):
        # Modifies the dictionary in place
        embed_batch(batch_ids=batch_ids, vector_components=vector_components, string_components=string_components, embedder=embedder)
        print(f"Embedded {i+1} out of {len(batches_by_id)} batches...")
    
    print(f"Saving vectors to {CACHE_PATH}...")
    os.makedirs(CACHE_PATH.parent, exist_ok=True)
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(vector_components, f)
        
    return vector_components