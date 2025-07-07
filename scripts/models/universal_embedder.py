# scripts/models/universal_embedder.py

import os
from pathlib import Path

from scripts.models.transformer_embedder import TransformerEmbedder
from scripts.models.legacy_embedder import LegacyEmbedder

class UniversalEmbedder:
    def __init__(self, model_name_or_path: str):
        self.model_name = model_name_or_path
        self.embedder = None
        self.backend = None

        try:
            self.embedder = TransformerEmbedder(model_name_or_path)
            self.backend = "transformers"
            print(f"✅ Loaded TransformerEmbedder: {model_name_or_path}")
        except Exception as e:
            print(f"⚠️ TransformerEmbedder failed: {e}")
            try:
                self.embedder = LegacyEmbedder(model_name_or_path)
                self.backend = "legacy"
                print(f"✅ Fallback to LegacyEmbedder: {model_name_or_path}")
            except Exception as fallback_error:
                print(f"❌ LegacyEmbedder also failed: {fallback_error}")
                raise RuntimeError(f"Could not load any embedder for {model_name_or_path}")

    def encode(self, text: str):
        if not self.embedder:
            raise RuntimeError("Embedder not initialized.")
        return self.embedder.encode(text)
