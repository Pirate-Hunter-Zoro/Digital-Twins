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
project_root = current_script_dir.parents[3] # Adjust path based on new location
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Import all our magnificent inventions! ---
from scripts.common.config import setup_config, get_global_config
from scripts.common.data_loading.load_patient_data import load_patient_data
from scripts.common.utils import clean_term
# NOTE: We're copying the Encoder class here, but it should live in a common models file!
from scripts.world_2_neighbor_analysis.compute_nearest_neighbors import HierarchicalPatientEncoder, get_visit_term_lists

# --- A NEW INVENTION! The Full Prediction Model! ---
# It's our encoder with a little "predictor" head attached!
class PatientReadmissionPredictor(nn.Module):
    def __init__(self, term_embedding_dim, visit_hidden_dim, patient_hidden_dim):
        super().__init__()
        self.encoder = HierarchicalPatientEncoder(
            term_embedding_dim, visit_hidden_dim, patient_hidden_dim
        )
        # This little gear makes the final prediction!
        self.classifier = nn.Linear(patient_hidden_dim, 1)

    def forward(self, patient_trajectory):
        patient_vector = self.encoder(patient_trajectory)
        if patient_vector is None:
            return None
        # The output is a "logit", a raw score before the sigmoid function.
        return self.classifier(patient_vector)

# --- A NEW DATASET MACHINE! To feed our model! ---
class PatientTrajectoryDataset(Dataset):
    def __init__(self, patient_data, term_vectorizer):
        self.trajectories = []
        self.labels = []
        self.term_vectorizer = term_vectorizer
        self._prepare_data(patient_data)

    def _prepare_data(self, patient_data):
        print("...Preparing data for the Training Gymnasium...")
        config = get_global_config()
        
        for patient in patient_data:
            if len(patient['visits']) < config.num_visits + 1:
                continue

            history_term_lists = get_visit_term_lists(patient['visits'])
            
            for end_idx, trajectory_lists in history_term_lists.items():
                # The label is based on the NEXT visit after the history window
                if end_idx + 1 < len(patient['visits']):
                    # --- SIMULATING A LABEL! ---
                    # Let's create a dummy label for 30-day readmission
                    history_end_date = pd.to_datetime(patient['visits'][end_idx]['StartVisit'])
                    next_visit_date = pd.to_datetime(patient['visits'][end_idx + 1]['StartVisit'])
                    days_diff = (next_visit_date - history_end_date).days
                    label = 1.0 if days_diff <= 30 else 0.0
                    
                    self.labels.append(label)
                    
                    # Convert terms to embeddings right here!
                    trajectory_tensors = []
                    for visit_terms in trajectory_lists:
                        if visit_terms:
                            trajectory_tensors.append(
                                self.term_vectorizer.encode(visit_terms, convert_to_tensor=True)
                            )
                    self.trajectories.append(trajectory_tensors)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # We need to pad the trajectories so they are all the same length in a batch
        # This is a complex step, so for this blueprint, we'll use a batch size of 1
        return self.trajectories[idx], torch.tensor(self.labels[idx], dtype=torch.float32)


def main():
    print("🏋️‍♀️ Welcome to the Hierarchical Encoder Training Gymnasium! 🏋️‍♀️")

    # --- Configuration ---
    VECTORIZER_METHOD = "allenai/scibert_scivocab_uncased"
    NUM_VISITS = 6
    EPOCHS = 3
    LEARNING_RATE = 1e-4
    
    setup_config("visit_sequence", VECTORIZER_METHOD, "cosine", NUM_VISITS, 0, 0)

    # --- Path to save our newly trained, super-smart model! ---
    output_dir = project_root / "data" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    trained_model_path = output_dir / "hierarchical_encoder_trained.pth"

    # --- Check if we've already trained it! ---
    if trained_model_path.exists():
        print(f"✅ Hooray! A trained model already exists at {trained_model_path}. No need to train again!")
        return

    # --- Load Tools & Data ---
    print("Loading tools and patient data...")
    term_vectorizer = SentenceTransformer(f"/media/scratch/mferguson/models/{VECTORIZER_METHOD.replace('/', '-')}")
    all_patient_data = load_patient_data()

    # --- Prepare Dataset and DataLoader ---
    # For simplicity in this blueprint, we'll use a batch size of 1
    # A real implementation would need a custom "collate_fn" to handle padding!
    print("Assembling the dataset... this may take a moment.")
    dataset = PatientTrajectoryDataset(all_patient_data, term_vectorizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # --- Initialize The Model, Optimizer, and Loss Function ---
    print("Building the new model and setting up the training equipment...")
    model = PatientReadmissionPredictor(
        term_embedding_dim=term_vectorizer.get_sentence_embedding_dimension(),
        visit_hidden_dim=128,
        patient_hidden_dim=256
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss() # Perfect for binary (0 or 1) classification!

    # --- THE TRAINING LOOP! LET'S GET STRONG! ---
    print(f"\n💪 Starting training for {EPOCHS} epochs! 💪")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (trajectory, label) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # The dataloader gives us a batch of 1, so we get the first item
            output = model(trajectory[0]) 

            if output is None: continue

            loss = loss_fn(output.squeeze(), label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS}, Step {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

        print(f"🌟 Epoch {epoch+1} complete! Average Loss: {total_loss / len(dataloader):.4f} 🌟")

    # --- Save the weights of our newly trained champion model! ---
    print(f"\n💾 Training complete! Saving the magnificent trained encoder to {trained_model_path}")
    # We only save the weights of the ENCODER part, not the final classifier head!
    torch.save(model.encoder.state_dict(), trained_model_path)
    print("✅ Done!")

if __name__ == "__main__":
    main()