# 🤖 Digital Twins for Medical Prediction 🤖

Welcome to the Digital Twins Project\! This repository contains a suite of magnificent machines and brilliant experiments designed to find the absolute best way to represent a patient's medical history. Our ultimate goal is to predict a patient's next medical visit by creating a "digital twin" and using a Large Language Model to see the future\!

This project is organized into four distinct "Worlds" of research, each with its own dedicated directory for scripts and Slurm jobs.

## 🏛️ Project Architecture

Our laboratory is now perfectly organized for maximum efficiency\! The main components are:

  * **`data/`**: This is where all the magnificent raw data, intermediate files, and final results live\! It holds our patient information, concept frequency lists, and the outputs of our experiments. Key subdirectories include:
      * `embeddings/`: Stores the raw cosine similarity scores from the main gauntlet.
      * `baseline_scores/`: Stores the results of our new, super-smart baseline significance score\!
      * Because this directory may contain confidential patient information, it is listed in the `.gitignore`.
  * **`scripts/`**: This is the brain of the operation\! It contains all the Python scripts that do the heavy lifting, neatly organized into a `common` directory for shared tools and a directory for each of the four "Worlds."
  * **`slurm_jobs/`**: This is our grand command center\! It contains all the launcher (`.sh`) and template (`.ssub`) files needed to submit jobs to a Slurm-based high-performance computing cluster. It's also perfectly organized by "Worlds"\!
  * **`References/`**: A library of all the brilliant research papers and notes that inspired this magnificent project\!

## 🚀 The Four Worlds

Our research is divided into four interconnected worlds, each building upon the last. The official workflow is **World 4 ➡️ World 2 ➡️ World 1 ➡️ World 3**.

### **World 1: The Prediction Engine** 🔮

  * **Goal**: To take a single patient, build their digital twin, and use an LLM to generate a prediction for their next medical visit.
  * **Key Scripts**: `scripts/world_1_prediction_engine/`
  * **Launcher**: `slurm_jobs/world_1_prediction_engine/launch_predict_grid.sh`

### **World 2: The Neighbor Quality Control Hub** 🔬

  * **Goal**: To analyze the "neighbors" (similar patients) found by our models to ensure they are medically relevant.
  * **Key Scripts**: `scripts/world_2_neighbor_analysis/`
  * **Launcher**: `slurm_jobs/world_2_neighbor_analysis/launch_nn_grid.sh`

### **World 3: The Judging Chamber** ⚖️

  * **Goal**: To score the LLM's predictions from World 1 against the patient's actual, real-life next visit. This is also where we can calculate term importance across documents.
  * **Key Scripts**:
      * `scripts/world_3_judging_chamber/evaluate.py`
      * `scripts/world_3_judging_chamber/generate_idf_registry.py`
  * **Launcher**: `slurm_jobs/world_3_judging_chamber/launch_generate_idf_registry.sh`

### **World 4: The Embedder Gauntlet** ⚙️

  * **Goal**: A grand tournament to find the best possible embedding model for understanding medical language. This world's experiments are largely complete\!
  * **Key Scripts**:
      * `scripts/world_4_embedder_gauntlet/embed_term_pairs.py`: Calculates cosine similarity.
      * `scripts/world_4_embedder_gauntlet/plot_similarity_distributions.py`: Plots the raw similarity scores.
      * `scripts/world_4_embedder_gauntlet/test_category_purity.py`: The final test for our champion model.
  * **New\! Baseline Score Analysis**: A secondary analysis to see how much better our models are than random chance\!
      * `scripts/world_4_embedder_gauntlet/compute_baseline_term_matching.py`: Calculates our new significance score.
      * `scripts/world_4_embedder_gauntlet/plot_baseline_distributions.py`: Plots the new baseline scores.
  * **Launchers**: `slurm_jobs/world_4_embedder_gauntlet/` contains all the launchers for running the gauntlet's various tests, including the new `launch_baseline_analysis.sh`\!

## 🔧 Setup & Installation

Before you can unleash the full power of these magnificent machines, you'll need to set up your environment\!

### 1\. Conda Environment

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

  * **Kaggle:** For our legacy models (GloVe & Word2Vec).
    1.  Go to your Kaggle account and create an API token to download `kaggle.json`.
    2.  Place it in a `~/.kaggle/` directory.
    3.  Make the key super-secret: `chmod 600 ~/.kaggle/kaggle.json`
  * **Hugging Face:** For some of our magnificent gated models.
    1.  Create an Access Token in your Hugging Face account settings.
    2.  Create a `.env` file in the root of this project and add your token: `HUGGINGFACE_TOKEN=hf_YourSuperSecretKeyGoesHere`

## 💥 Usage: Running the Machine\!

The project is designed as a sequence of experiments.

1.  **Setup**: Run the model download scripts in `slurm_jobs/setup/`.
2.  **World 4 - Main Gauntlet**: Run the main similarity gauntlets in `slurm_jobs/world_4_embedder_gauntlet/` to generate the raw cosine similarity scores.
3.  **World 4 - Baseline Analysis (Optional)**: After the main gauntlet is done, run `./slurm_jobs/world_4_embedder_gauntlet/launch_baseline_analysis.sh` to calculate and plot our new significance scores\!
4.  **Continue the Workflow**: Proceed with the experiments for World 2, then World 1, and finally World 3\!

Let the SCIENCE begin\!