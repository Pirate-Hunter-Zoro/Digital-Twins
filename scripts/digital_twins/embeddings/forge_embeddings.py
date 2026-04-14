import os
from tqdm import tqdm
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from scripts.models.patient_embedder import PatientEmbedder

RECORD_EVERY = 30

def forge():
    narratives_dir = Path(os.environ['DETERMINISTIC_NARRATIVES_DIR'])
    if not narratives_dir.exists():
        raise ValueError(f"Missing directory: {str(narratives_dir)}...", flush=True)
    else:
        embedder = PatientEmbedder()
        # Embed all of our deterministic narratives
        narrative_files = narratives_dir.glob("*.md")
        batch_size = int(os.environ['EMBEDDER_BATCH_SIZE'])
        current_narrative_batch = []
        current_patient_id_batch = []
        batches_embedded = 0
        # Use tqdm to view progress
        for narrative_file in tqdm(narrative_files):
            with open(narrative_file, 'r') as f:
                if len(current_narrative_batch) == batch_size:
                    embedder.embed((current_patient_id_batch, current_narrative_batch))
                    batches_embedded += 1
                    if batches_embedded % RECORD_EVERY == 0:
                        print(f"Finished embedding {batches_embedded} batches of size {batch_size}...", flush=True)
                    current_narrative_batch = []
                    current_patient_id_batch = []
                current_narrative_batch.append(f.read())
                current_patient_id_batch.append(narrative_file.stem)
        if len(current_narrative_batch) > 0:
            # One last leftover batch
            embedder.embed((current_patient_id_batch, current_narrative_batch))
            
if __name__=="__main__":
    forge()