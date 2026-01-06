import os
from tqdm import tqdm
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from scripts.models.string_embedder import StringEmbedder

RECORD_EVERY = 30

def forge():
    narratives_dir = Path(os.environ['DETERMINISTIC_NARRATIVES_DIR'])
    if not narratives_dir.exists():
        raise ValueError(f"Missing directory: {str(narratives_dir)}...", flush=True)
    else:
        embedder = StringEmbedder()
        # Embed all of our deterministic narratives
        narrative_files = narratives_dir.glob("*.md")
        batch_size = int(os.environ['EMBEDDER_BATCH_SIZE'])
        current_batch = []
        batches_embedded = 0
        # Use tqdm to view progress
        for narrative_file in tqdm(narrative_files):
            if len(current_batch) < batch_size:
                with open(narrative_file, 'r') as f:
                    current_batch.append(f.read())
            else:
                embedder.vectorize(current_batch)
                batches_embedded += 1
                if batches_embedded % RECORD_EVERY == 0:
                    print(f"Finished embedding {batches_embedded} batches of size {batch_size}...", flush=True)
                current_batch = []
        if len(current_batch) > 0:
            # One last leftover batch
            embedder.vectorize(current_batch)