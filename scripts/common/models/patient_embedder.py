import os
from typing import Dict, List
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import torch

load_dotenv()

class PatientEmbedder:
    """
    A simplified class that uses the sentence-transformers library to embed text.
    It handles all the underlying complexity.
    """
    # The home of your models remains unchanged.
    BASE_MODEL_DIR = "/media/studies/ehr_study/data-EHR-prepped/Mikey-Digital-Twins/models/"

    def __init__(self):
        """
        Loads a SentenceTransformer model from a local path.
        The library automatically handles device placement.

        Args:
            model_name: The name of the model directory under BASE_MODEL_DIR.
        """
        full_model_path = Path(os.environ['EMBEDDER_MODEL_PATH'])

        if not os.path.isdir(full_model_path):
            raise FileNotFoundError(f"Pathetic. Model directory not found: {full_model_path}")

        model_name = os.environ['EMBEDDER_MODEL_NAME']
        print(f"[PatientEmbedder] Loading SentenceTransformer model '{model_name}'.")

        # The library handles everything: loading the model, tokenizer, and pooling configuration.
        self.model = SentenceTransformer(
            model_name_or_path = str(full_model_path),
            device = os.environ['EMBEDDER_DEVICE'] if torch.cuda.is_available() else 'cpu'
        )

    def vectorize(self, narratives: list[str]) -> List[np.array]:
        """
        Generates normalized vector embeddings for a batch of texts using a simple .encode() call.
        """
        # The encode method handles tokenization, inference, and pooling.
        # normalize_embeddings=True is the same as the manual normalization you were doing.
        vectors = self.model.encode(
            narratives,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        return [vec.astype(np.float32) for vec in vectors]