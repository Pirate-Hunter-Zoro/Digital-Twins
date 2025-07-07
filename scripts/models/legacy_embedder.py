import os
import numpy as np
from pathlib import Path
from gensim.models import KeyedVectors


class LegacyEmbedder:
    def __init__(self, model_name: str):
        """
        Load GloVe, Word2Vec, or FastText from local models directory.
        Expects files to already be present in ../models/
        Supported model_name values:
            - 'glove.6B.100d.txt'
            - 'glove.6B.200d.txt'
            - 'glove.6B.300d.txt'
            - 'word2vec-google-news-300.model'
            - 'fasttext-wiki-news-subwords-300.model'
        """
        self.model_name = model_name
        self.vector_size = None
        self.model = self._load_model()

    def _load_model(self):
        models_dir = Path(__file__).resolve().parents[3] / "models"
        model_path = models_dir / self.model_name

        print(f"📦 Loading local model from {model_path}")
        if self.model_name.endswith(".model"):
            # Binary Gensim format (FastText or Word2Vec)
            model = KeyedVectors.load(str(model_path), mmap='r')
        elif self.model_name.endswith(".txt"):
            # GloVe format in .txt (converted to word2vec format)
            model = KeyedVectors.load_word2vec_format(str(model_path), binary=False)
        else:
            raise ValueError(f"❌ Unsupported model file type: {self.model_name}")

        self.vector_size = model.vector_size
        print(f"✅ Model loaded with vector size {self.vector_size}")
        return model

    def encode(self, text: str) -> np.ndarray:
        tokens = text.strip().lower().split()
        vectors = [self.model[token] for token in tokens if token in self.model]

        if not vectors:
            return np.zeros(self.vector_size)

        return np.mean(vectors, axis=0)

    def __call__(self, text: str) -> np.ndarray:
        return self.encode(text)
