import os
import sys
import json
import logging
from tqdm import tqdm
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from scripts.models.string_embedder import StringEmbedder

def forge():
    narratives_dir = Path(os.environ['DETERMINISTIC_NARRATIVES_DIR'])
    if not narratives_dir.exists():
        raise ValueError(f"Missing directory: {str(narratives_dir)}...")
    else:
        embedder = StringEmbedder()
        # Embed all of our deterministic narratives
        narrative_files = narratives_dir.glob("*.md")
        batch_size = int(os.environ['EMBEDDER_BATCH_SIZE'])
        current_batch = []
        # Use tqdm to view progress
        for narrative_file in tqdm(narrative_files):
            if len(current_batch) < batch_size:
                with open(narrative_file, 'r') as f:
                    current_batch.append(f.read())
            else:
                embedder.vectorize(current_batch)
                current_batch = []
        if len(current_batch) > 0:
            # One last leftover batch
            embedder.vectorize(current_batch)