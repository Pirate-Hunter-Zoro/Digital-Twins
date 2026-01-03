import os
from typing import List
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from scripts.shared.utils import generate_string_id
import torch

from dotenv import load_dotenv
load_dotenv()

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
            raise FileNotFoundError(f"Pathetic. Model directory not found: {full_model_path}")

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
        # The encode method handles tokenization, inference, and pooling.
        vectors = [None for _ in strings]
        to_compute = []
        to_compute_indices = []
        for i, string in enumerate(strings):
            id = generate_string_id(text=string)
            vectors_store_path = self.vectors_path / f"vector_{id}.npy"
            if vectors_store_path.exists() and not self.scrub_vectors:
                vectors[i] = np.load(vectors_store_path)
            else:
                to_compute.append(string)
                to_compute_indices.append(i)
                
        if len(to_compute) > 0:
            missing_vectors = self.model.encode(
                to_compute,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True,
                batch_size=int(os.environ['EMBEDDER_BATCH_SIZE'])
            )
            for i, missing_vector in zip(to_compute_indices, missing_vectors):
                vectors[i] = missing_vector
                id = generate_string_id(text=strings[i])
                vectors_store_path = self.vectors_path / f"vector_{id}.npy"
                np.save(vectors_store_path, missing_vector)
                string_store_path = self.vectors_path / f"string_{id}.txt"
                with open(string_store_path, 'w') as f:
                    f.write(strings[i])
                
        return [vec.astype(np.float32) for vec in vectors]