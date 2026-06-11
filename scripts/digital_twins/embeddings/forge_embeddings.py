import os
from tqdm import tqdm
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from scripts.models.patient_embedder import PatientEmbedder

RECORD_EVERY = 30

def forge():
    narratives_dir = Path(os.environ['NARRATIVES_DIR'])
    if not narratives_dir.exists():
        raise ValueError(f"Missing directory: {str(narratives_dir)}...", flush=True)
    else:
        embedder = PatientEmbedder()
        embedder_valid_ids = set(embedder.narrative_chronological_lengths.keys())
        # Embed all of our narratives
        narrative_files = list(narratives_dir.glob("*.md"))
        for p in narrative_files:
            if p.stem not in embedder_valid_ids:
                p.unlink(missing_ok=True)
        narrative_valid_ids = set([p.stem for p in narrative_files])
        narrative_valid_ids = narrative_valid_ids & embedder_valid_ids
        narrative_files = [p for p in narrative_files if p.stem in narrative_valid_ids]
        for deleted_id in embedder.purge_orphans(narrative_valid_ids):
            print(f"Deleted orphan patient {deleted_id} from embedding database...", flush=True)
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