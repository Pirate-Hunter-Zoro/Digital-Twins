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
from sklearn.model_selection import train_test_split # For our data split!
from torch.nn.utils import clip_grad_norm_ # For our safety rail!

# --- Dynamic sys.path adjustment ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.common.config import setup_config, get_global_config
from scripts.common.data_loading.load_patient_data import load_patient_data
from scripts.common.utils import clean_term, get_visit_term_lists
from scripts.common.models.hierarchical_encoder import HierarchicalPatientEncoder

# --- The Full Prediction Model ---
class PatientReadmissionPredictor(nn.Module):
    def __init__(self, term_embedding_dim, visit_hidden_dim, patient_hidden_dim):
        super().__init__()
        self.encoder = HierarchicalPatientEncoder(term_embedding_dim, visit_hidden_dim, patient_hidden_dim)
        self.classifier = nn.Linear(patient_hidden_dim, 1)

    def forward(self, patient_trajectory):
        patient_vector = self.encoder(patient_trajectory)
        if patient_vector is None: return None
        return self.classifier(patient_vector)

# --- The Data-Feeder Bot ---
class PatientTrajectoryDataset(Dataset):
    def __init__(self, patient_data_subset, term_vectorizer, device):
        self.trajectories = []
        self.labels = []
        self.term_vectorizer = term_vectorizer
        self.device = device
        self._prepare_data(patient_data_subset)

    def _prepare_data(self, patient_data):
        print("...Preparing data subset...")
        config = get_global_config()
        positive_labels = 0
        
        for patient in patient_data:
            if len(patient['visits']) < config.num_visits + 1: continue

            history_term_lists = get_visit_term_lists(patient['visits'], config.num_visits)
            
            for end_idx, trajectory_lists in history_term_lists.items():
                if end_idx + 1 < len(patient['visits']):
                    trajectory_tensors = []
                    for visit_terms in trajectory_lists:
                        if visit_terms:
                            trajectory_tensors.append(
                                self.term_vectorizer.encode(visit_terms, convert_to_tensor=True, device=self.device)
                            )
                    
                    if not trajectory_tensors: continue

                    self.trajectories.append(trajectory_tensors)
                    
                    history_end_date = pd.to_datetime(patient['visits'][end_idx]['StartVisit'])
                    next_visit_date = pd.to_datetime(patient['visits'][end_idx + 1]['StartVisit'])
                    label = 1.0 if (next_visit_date - history_end_date).days <= 30 else 0.0
                    if label == 1.0: positive_labels += 1
                    self.labels.append(label)
        
        print(f"✅ Subset prepared! Found {len(self.labels)} total samples.")
        print(f"   - Positive Samples (Readmitted): {positive_labels} ({(positive_labels/len(self.labels)*100):.2f}%)")
        print(f"   - Negative Samples (Not Readmitted): {len(self.labels) - positive_labels}")

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.trajectories[idx], torch.tensor(self.labels[idx], dtype=torch.float32)

# --- The New Accuracy Calculator Bot! ---
def calculate_accuracy(model, dataloader, device):
    model.eval() # Set model to evaluation mode!
    correct = 0
    total = 0
    with torch.no_grad():
        for trajectories, labels in dataloader:
            labels = labels.to(device)
            output = model(trajectories[0])
            if output is None: continue
            
            # Turn logits into probabilities and then into 0 or 1 predictions!
            preds = torch.sigmoid(output.squeeze(-1)) >= 0.5
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return (correct / total) * 100 if total > 0 else 0

def main():
    print("🏋️‍♀️ Welcome to the Upgraded Hierarchical Encoder Training Gymnasium! 🏋️‍♀️")

    # --- Configuration ---
    VECTORIZER_METHOD = "allenai/scibert_scivocab_uncased"
    NUM_VISITS = 6
    EPOCHS = 5 # Let's train for a bit longer!
    LEARNING_RATE = 1e-5 # ✨ UPGRADE 1: Lower Learning Rate!
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🛠️ Using device: {device}")
    
    setup_config("visit_sentence", VECTORIZER_METHOD, "cosine", NUM_VISITS, 5000, 5)

    output_dir = project_root / "data" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    trained_model_path = output_dir / "hierarchical_encoder_trained.pth"

    if trained_model_path.exists():
        print(f"✅ A trained model already exists at {trained_model_path}. Skipping training!")
        return

    print("Loading tools and patient data...")
    term_vectorizer = SentenceTransformer(f"/media/scratch/mferguson/models/{VECTORIZER_METHOD.replace('/', '-')}")
    all_patient_data = load_patient_data()

    # --- ✨ UPGRADE 2: Train/Test Split! ---
    print("\n--- Splitting data into training and testing sets... ---")
    train_data, test_data = train_test_split(all_patient_data, test_size=0.2, random_state=42)
    
    print("Assembling training dataset...")
    train_dataset = PatientTrajectoryDataset(train_data, term_vectorizer, device)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    print("\nAssembling testing dataset...")
    test_dataset = PatientTrajectoryDataset(test_data, term_vectorizer, device)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("\nBuilding the new model...")
    model = PatientReadmissionPredictor(
        term_embedding_dim=term_vectorizer.get_sentence_embedding_dimension(),
        visit_hidden_dim=128, patient_hidden_dim=256
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    # --- ✨ UPGRADE 3: Learning Rate Scheduler! ---
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)

    best_test_accuracy = 0.0
    epochs_without_improvement = 0

    print(f"\n💪 Starting training for {EPOCHS} epochs! 💪")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i, (trajectory, label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            label = label.to(device)
            output = model(trajectory[0]) 
            if output is None: continue

            loss = loss_fn(output.squeeze(-1), label)
            loss.backward()
            # --- ✨ UPGRADE 4: Gradient Clipping! ---
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        train_acc = calculate_accuracy(model, train_dataloader, device)
        test_acc = calculate_accuracy(model, test_dataloader, device)
        
        print(f"🌟 Epoch {epoch+1} complete! Avg Loss: {total_loss / len(train_dataloader):.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% 🌟")
        
        # --- ✨ Checkpoint and Early Stopping Logic! ✨ ---
        if test_acc > best_test_accuracy:
            print(f"🚀 New best test accuracy! {test_acc:.2f}% > {best_test_accuracy:.2f}%. Saving champion model!")
            best_test_accuracy = test_acc
            torch.save(model.encoder.state_dict(), trained_model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"🤔 Test accuracy did not improve. Patience counter: {epochs_without_improvement}/{PATIENCE}")

        if epochs_without_improvement >= PATIENCE:
            print(f"🚫 Early stopping triggered after {epoch+1} epochs! The champion has been crowned!")
            break
            
        scheduler.step()

    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()