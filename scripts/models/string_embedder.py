import os
from typing import List
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3
import torch

from dotenv import load_dotenv
load_dotenv()

from scripts.shared.utils import generate_string_id

class StringEmbedder:
    """
    A simplified class that uses the sentence-transformers library to embed text.
    It handles all the underlying complexity.
    """
   
    def __init__(self):
        """
        Loads a SentenceTransformer model from a local path.
        The library automatically handles device placement.
        """
        full_model_path = Path(os.environ['EMBEDDER_MODEL_PATH'])

        if not os.path.isdir(full_model_path):
            raise FileNotFoundError(f"Model directory not found: {full_model_path}")

        model_name = os.environ['EMBEDDER_MODEL_NAME']
        print(f"[PatientEmbedder] Loading SentenceTransformer model '{model_name}'.")

        # The library handles everything: loading the model, tokenizer, and pooling configuration.
        self.model = SentenceTransformer(
            model_name_or_path = str(full_model_path),
            device = os.environ['EMBEDDER_DEVICE'] if torch.cuda.is_available() else 'cpu',
            trust_remote_code=True
        )
        self.vectors_path = Path(os.environ['VECTORS_DIR'])
        os.makedirs(self.vectors_path, exist_ok=True)
        
        # Make connection to database
        self.connection = sqlite3.connect(self.vectors_path / 'vectors.db')
        self.connection.execute('''
CREATE TABLE IF NOT EXISTS vectors (
    id TEXT PRIMARY KEY,
    vector BLOB,
    text TEXT,
    length INTEGER
);
''')
        
        # Whether vectors are to be scrubbed and recomputing
        self.scrub_vectors = int(os.environ['SCRUB_VECTORS']) == 1

    def vectorize(self, strings: list[str]) -> List[np.array]:
        """
        Generates normalized vector embeddings for a batch of texts using a simple .encode() call.
        
        :param strings: List of strings to embed
        :type strings: list[str]
        :return: Resulting vectors
        :rtype: List
        """
        cursor = self.connection.cursor()
        vectors = [None for _ in strings]
        to_compute = []
        to_compute_indices = []
        for i, string in enumerate(strings):
            id = generate_string_id(text=string)
            cursor.execute("SELECT vector FROM vectors WHERE id=?", (id,))
            # See if we already have this string vectorized (and we're not scrubbing)
            row = cursor.fetchone()
            if row == None or self.scrub_vectors:
                # Need to recompute vector
                to_compute_indices.append(i)
                to_compute.append(string)
            else:
                vectors[i] = np.frombuffer(row[0], dtype=np.float32)
                
        if len(to_compute) > 0:
            missing_vectors = self.model.encode(
                to_compute,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True,
                batch_size=int(os.environ['EMBEDDER_BATCH_SIZE'])
            )
            new_records = []
            for i, missing_vector in zip(to_compute_indices, missing_vectors):
                vectors[i] = missing_vector
                id = generate_string_id(text=strings[i])
                vector_bytes = missing_vector.tobytes()
                new_records.append((id, vector_bytes, strings[i], len(strings[i])))
                
            self.connection.executemany(
                sql='''
INSERT OR REPLACE INTO vectors (id, vector, text, length) VALUES (?, ?, ?, ?)
''',
                parameters = new_records
            )
            self.connection.commit()
                
        return [vec.astype(np.float32) for vec in vectors]