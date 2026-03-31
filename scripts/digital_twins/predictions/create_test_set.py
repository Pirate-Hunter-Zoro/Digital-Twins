import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

from scripts.digital_twins.predictions.trd_predictor import TRDPredictor

test_ids_path = Path(os.environ['ANALYSIS_DIR']) / 'test_patient_ids.txt'

def create_train_test_split() -> tuple[set[str], set[str]]:
    """Split the patient population pool into a train and test set stratified by TRD status

    Returns:
        tuple[set[str], set[str]]: train ids, test ids
    """
    all_vector_paths = Path(os.environ['DETERMINISTIC_VECTORS_DIR']).glob("*.npy")
    all_patient_ids = [p.stem for p in all_vector_paths]
    predictor = TRDPredictor()
    if (not test_ids_path.exists()) or (int(os.environ['SCRUB_DETERMINISTIC_VECTORS']) == 1):
        # Need to create stratified train/test split
        train_ids, test_ids = train_test_split(all_patient_ids, test_size=0.2, stratify=[predictor.get_trd_status(id) for id in all_patient_ids], random_state=int(os.environ['SEED']))
        with open(test_ids_path, 'w') as f:
            f.write("\n".join(test_ids))
        train_ids = set(train_ids)
        test_ids = set(test_ids)
        
    # Load in the test IDs to implicitly find the train ones
    with open(test_ids_path, 'r') as f:
        test_ids = set(f.read().splitlines())
        train_ids = set([id for id in all_patient_ids if id not in test_ids])
        
    train_trd_count = len([id for id in train_ids if predictor.get_trd_status(id)==1])
    test_trd_count = len([id for id in test_ids if predictor.get_trd_status(id)==1])
    print(f"Found {len(all_patient_ids)} total patients.\n\
Split into train set of size {len(train_ids)} and test set of size {len(test_ids)}.\n\
Train set TRD count: {train_trd_count}\n\
Test set TRD count: {test_trd_count}", flush=True)

    return (train_ids, test_ids)