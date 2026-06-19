import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd

from scripts.embedder_investigation.predictions.trd_predictor import TRDPredictor

test_ids_path = Path(os.environ['ANALYSIS_DIR']) / 'test_patient_ids.txt'

def create_train_test_split() -> tuple[set[str], set[str]]:
    """Split the patient population pool into a train and test set stratified by TRD status

    Returns:
        tuple[set[str], set[str]]: train ids, test ids
    """
    # Read in just the index - no columns
    patient_df = pd.read_parquet(Path(os.environ['FEATURE_DATAFRAME_PATH']), columns=[])
    all_patient_ids = patient_df.index.tolist()
    predictor = TRDPredictor()
    if (not test_ids_path.exists()) or (int(os.environ['SCRUB_FEATURE_VECTORS']) == 1):
        # Need to create stratified train/test split
        train_ids, test_ids = train_test_split(all_patient_ids, test_size=0.2, stratify=[predictor.get_trd_status(id) for id in all_patient_ids], random_state=int(os.environ['SEED']))
        with open(test_ids_path, 'w') as f:
            f.write("\n".join(test_ids))
        
    # Load in the test IDs to implicitly find the train ones
    with open(test_ids_path, 'r') as f:
        # Intersect with all valid patients just in case an old invalid patient from a stale file shows up in the test_ids
        test_ids = set(f.read().splitlines()) & set(all_patient_ids)
        train_ids = set([id for id in all_patient_ids if id not in test_ids])
        
    train_trd_count = len([id for id in train_ids if predictor.get_trd_status(id)==1])
    test_trd_count = len([id for id in test_ids if predictor.get_trd_status(id)==1])
    print(f"Found {len(all_patient_ids)} total patients.\n\
Split into train set of size {len(train_ids)} and test set of size {len(test_ids)}.\n\
Train set TRD count: {train_trd_count}\n\
Test set TRD count: {test_trd_count}", flush=True)

    return (train_ids, test_ids)