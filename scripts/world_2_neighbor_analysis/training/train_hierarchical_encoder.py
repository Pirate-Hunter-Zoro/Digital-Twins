# scripts/world_2_neighbor_analysis/training/train_hierarchical_encoder.py

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader

# --- Dynamic sys.path adjustment! ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.common.config import setup_config, get_global_config
from scripts.common.data_loading.load_patient_data import load_patient_data
from scripts.common.utils import clean_term, get_visit_term_lists
from scripts.common.models.hierarchical_encoder import HierarchicalPatientEncoder

class PatientReadmissionPredictor(nn.Module):
    def __init__(self, term_embedding_dim, visit_hidden_dim, patient_hidden_dim):
        super().__init__()
        self.encoder = HierarchicalPatientEncoder(
            term_embedding_dim, visit_hidden_dim, patient_hidden_dim
        )
        self.classifier = nn.Linear(patient_hidden_dim, 1)

    def forward(self, patient_trajectory):
        patient_vector = self.encoder(patient_trajectory)
        if patient_vector is None:
            return None
        return self.classifier(patient_vector)

class PatientTrajectoryDataset(Dataset):
    def __init__(self, patient_data, term_vectorizer, device):
        self.trajectories = []
        self.labels = []
        self.term_vectorizer = term_vectorizer
        self.device = device
        self._prepare_data(patient_data)

    def _prepare_data(self, patient_data):
        print("...Preparing data for the Training Gymnasium...")
        config = get_global_config()
        
        for patient in patient_data:
            if len(patient['visits']) < config.num_visits + 1:
                continue

            history_term_lists = get_visit_term_lists(patient['visits'], config.num_visits)
            
            for end_idx, trajectory_lists in history_term_lists.items():
                if end_idx + 1 < len(patient['visits']):
                    trajectory_tensors = []
                    for visit_terms in trajectory_lists:
                        if visit_terms:
                            trajectory_tensors.append(
                                self.term_vectorizer.encode(visit_terms, convert_to_tensor=True, device=self.device)
                            )
                    
                    # If, after all that, we have no tensors, this history is empty!
                    # So we skip it and don't create a label for it!
                    if not trajectory_tensors:
                        continue
                    # --------------------------------

                    self.trajectories.append(trajectory_tensors)
                    
                    history_end_date = pd.to_datetime(patient['visits'][end_idx]['StartVisit'])
                    next_visit_date = pd.to_datetime(patient['visits'][end_idx + 1]['StartVisit'])
                    days_diff = (next_visit_date - history_end_date).days
                    label = 1.0 if days_diff <= 30 else 0.0
                    self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.trajectories[idx], torch.tensor(self.labels[idx], dtype=torch.float32)

def main():
    print("🏋️‍♀️ Welcome to the Hierarchical Encoder Training Gymnasium! 🏋️‍♀️")

    # --- Configuration ---
    VECTORIZER_METHOD = "allenai/scibert_scivocab_uncased"
    NUM_VISITS = 6
    EPOCHS = 3
    LEARNING_RATE = 1e-4
    
    # --- ✨THE FIRST PART OF THE FIX! ✨ ---
    # We detect if a GPU is available and set our device!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🛠️ Using device: {device}")
    
    setup_config("visit_sequence", VECTORIZER_METHOD, "cosine", NUM_VISITS, 0, 0)

    output_dir = project_root / "data" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    trained_model_path = output_dir / "hierarchical_encoder_trained.pth"

    if trained_model_path.exists():
        print(f"✅ A trained model already exists. Skipping training!")
        return

    print("Loading tools and patient data...")
    term_vectorizer = SentenceTransformer(f"/media/scratch/mferguson/models/{VECTORIZER_METHOD.replace('/', '-')}")
    all_patient_data = load_patient_data()

    print("Assembling the dataset...")
    dataset = PatientTrajectoryDataset(all_patient_data, term_vectorizer, device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print("Building the new model...")
    model = PatientReadmissionPredictor(
        term_embedding_dim=term_vectorizer.get_sentence_embedding_dimension(),
        visit_hidden_dim=128,
        patient_hidden_dim=256
    )
    # --- ✨THE SECOND PART OF THE FIX! ✨ ---
    # We move our entire model to the GPU!
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"\n💪 Starting training for {EPOCHS} epochs! 💪")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (trajectory, label) in enumerate(dataloader):
            optimizer.zero_grad()
            
            label = label.to(device)
            output = model(trajectory[0]) 

            if output is None: continue

            # --- THE MAGNIFICENT, PRECISE FIX! ---
            # We now squeeze only the last dimension to match the label's shape!
            loss = loss_fn(output.squeeze(-1), label)
            # ------------------------------------
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS}, Step {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

        print(f"🌟 Epoch {epoch+1} complete! Average Loss: {total_loss / len(dataloader):.4f} 🌟")

    print(f"\n💾 Training complete! Saving the magnificent trained encoder to {trained_model_path}")
    torch.save(model.encoder.state_dict(), trained_model_path)
    print("✅ Done!")

if __name__ == "__main__":
    main()