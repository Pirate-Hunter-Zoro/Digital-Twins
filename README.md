# 🤖 Digital Twins for Medical Prediction 🤖

Welcome to the Digital Twins Project\! This repository contains a suite of magnificent machines and brilliant experiments designed to find the absolute best way to represent a patient's medical history. Our ultimate goal is to predict a patient's next medical visit by creating a "digital twin" and using a Large Language Model to see the future\!

This project is organized into four distinct "Worlds" of research, each with its own dedicated directory for scripts and Slurm jobs.

## 🏛️ Project Architecture

Our laboratory is now perfectly organized for maximum efficiency\! The main components are:

  * **`data/`**: This is where all the magnificent raw data, intermediate files, and final results live\! It holds our patient information, concept frequency lists, generated embeddings, and all the beautiful plots from our experiments. Because this directory contains confidential patient information, it is listed in the `.gitignore` and will not be tracked by version control.
  * **`scripts/`**: This is the brain of the operation\! It contains all the Python scripts that do the heavy lifting, neatly organized into a `common` directory for shared tools and a directory for each of the four "Worlds."
  * **`slurm_jobs/`**: This is our grand command center\! It contains all the launcher (`.sh`) and template (`.ssub`) files needed to submit jobs to a Slurm-based high-performance computing cluster. It's also perfectly organized by "Worlds"\!
  * **`References/`**: A library of all the brilliant research papers and notes that inspired this magnificent project\!

## 🚀 The Four Worlds

Our research is divided into four interconnected worlds, each building upon the last.

### **World 1: The Prediction Engine** 🔮

  * **Goal**: To take a single patient, build their digital twin, and use an LLM to generate a prediction for their next medical visit.
  * **Key Scripts**:
      * `scripts/world_1_prediction_engine/generate_patient_predictions.py`
      * `scripts/world_1_prediction_engine/process_patient.py`
  * **Launcher**: `slurm_jobs/world_1_prediction_engine/launch_predict_grid.sh`

### **World 2: The Neighbor Quality Control Hub** 🔬

  * **Goal**: To analyze the "neighbors" (similar patients) found by our models to ensure they are medically relevant and not just mathematical coincidences.
  * **Key Scripts**:
      * `scripts/world_2_neighbor_analysis/compute_nearest_neighbors.py`
      * `scripts/world_2_neighbor_analysis/examine_nearby_patients.py`
  * **Launcher**: `slurm_jobs/world_2_neighbor_analysis/launch_spearman_rho_grid.sh`

### **World 3: The Judging Chamber** ⚖️

  * **Goal**: To score the LLM's predictions from World 1 against the patient's actual, real-life next visit to see how accurate we are\!
  * **Key Scripts**:
      * `scripts/world_3_judging_chamber/evaluate.py`
      * `scripts/world_3_judging_chamber/compute_patient_prediction_scores.py`

### **World 4: The Embedder Gauntlet** ⚙️

  * **Goal**: A grand tournament to find the best possible embedding model for understanding medical language. This world's experiments are largely complete\!
  * **Key Scripts**:
      * `scripts/world_4_embedder_gauntlet/embed_term_pairs.py`
      * `scripts/world_4_embedder_gauntlet/plot_similarity_distributions.py`
      * `scripts/world_4_embedder_gauntlet/test_category_purity.py`
  * **Launchers**: `slurm_jobs/world_4_embedder_gauntlet/` contains all the launchers for running the gauntlet's various tests.

## 🔧 Setup & Installation

Before you can unleash the full power of these magnificent machines, you'll need to set up your environment\!

### 1\. Conda Environment

All these scripts were designed to live inside a beautiful, magnificent Conda environment.

```bash
conda create --name hugging_env python=3.9
conda activate hugging_env
```

### 2\. Python Dependencies

Install all the necessary Python libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 3\. API Tokens & Keys

We need to talk to some very important data-librarians at Kaggle and Hugging Face\!

  * **Kaggle:** For our legacy models (GloVe & Word2Vec).
    1.  Go to your Kaggle account page and create a new API token to download `kaggle.json`.
    2.  Create a secret folder in your home directory: `mkdir ~/.kaggle`
    3.  Place your `kaggle.json` file inside it.
    4.  **Crucially**, make the key super-secret so the Kaggle servers trust you\! `chmod 600 ~/.kaggle/kaggle.json`
  * **Hugging Face:** For some of our magnificent gated models.
    1.  Create an Access Token in your Hugging Face account settings.
    2.  Create a `.env` file in the root of this project directory.
    3.  Add your token to the `.env` file like this (with no quotes\!): `HUGGINGFACE_TOKEN=hf_YourSuperSecretKeyGoesHere`

## 💥 Usage: Running the Machine\!

The project is designed as a sequence of experiments, starting with the setup jobs.

1.  **Download Models**: Before running any experiments, download all the necessary models using the launcher scripts in `slurm_jobs/setup/`.
2.  **Run Experiments**: Navigate to the directory for the "World" you wish to experiment with (e.g., `slurm_jobs/world_2_neighbor_analysis/`) and use the `.sh` launcher scripts to submit your jobs\!

Let the SCIENCE begin\!