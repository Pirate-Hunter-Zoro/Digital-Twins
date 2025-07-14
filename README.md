# 🤖 Digital Twins for Medical Prediction 🤖

Welcome to the Digital Twins Project! This repository contains a suite of magnificent machines and brilliant experiments designed to find the absolute best way to represent a patient's medical history. Our ultimate goal is to predict a patient's next medical visit by creating a "digital twin" and using a Large Language Model to see the future!

This project is organized into four distinct "Worlds" of research, each with its own dedicated directory for scripts and Slurm jobs.

## 🏛️ Project Architecture

Our laboratory is now perfectly organized for maximum efficiency! The main components are:

* **`data/`**: This is where all the magnificent raw data, intermediate files, and final results live! It holds our patient information, concept frequency lists, and the outputs of our experiments. Because this directory may contain confidential patient information, it is listed in the `.gitignore`. The subdirectories are now dynamically named based on the parameters of each experiment for ultimate organization!
* **`scripts/`**: This is the brain of the operation! It contains all the Python scripts that do the heavy lifting, neatly organized into a `common` directory for shared tools and a directory for each of the four "Worlds."
* **`slurm_jobs/`**: This is our grand command center! It contains all the launcher (`.sh`) and template (`.ssub`) files needed to submit jobs to a Slurm-based high-performance computing cluster. It's also perfectly organized by "Worlds"!
* **`References/`**: A library of all the brilliant research papers and notes that inspired this magnificent project!

## 🚀 The Four Worlds

Our research is divided into four interconnected worlds, each building upon the last. The official workflow is **World 4 ➡️ World 2 ➡️ World 1 ➡️ World 3**.

### **World 1: The Prediction Engine** 🔮

* **Goal**: To take a single patient, build their digital twin, and use an LLM to generate a prediction for their next medical visit.
* **Status**: On Hold

### **World 2: The Neighbor Quality Control Hub** 🔬

* **Goal**: To analyze the "neighbors" (similar patients) found by our models to ensure they are medically relevant. We do this by calculating two metrics: the Mahalanobis distance of a patient to its neighbors and an LLM-generated relevance score. We then calculate the Spearman-rho correlation between these two metrics.
* **Key Scripts**:
    * `scripts/world_2_neighbor_analysis/compute_nearest_neighbors.py`: The Neighbor Discovery Machine! It uses a sentence-transformer model to generate vector embeddings for all patient visit sequences and finds the nearest neighbors for each. It now uses a sliding window to generate more data!
    * `scripts/world_2_neighbor_analysis/examine_and_correlate_neighbors.py`: The Quality Control-inator! This script takes the neighbors, calculates the Mahalanobis distance and LLM relevance scores, and generates a beautiful graph of the final correlation.
* **Launcher**: `slurm_jobs/world_2_neighbor_analysis/launch_world2_grid.sh`
* **Status**: **ACTIVE!** The grand experiment is currently running on the cluster!

### **World 3: The Judging Chamber** ⚖️

* **Goal**: To score the LLM's predictions from World 1 against the patient's actual, real-life next visit.
* **Key Scripts**: `scripts/world_3_judging_chamber/`
* **Status**: On Hold

### **World 4: The Embedder Gauntlet** ⚙️

* **Goal**: A grand tournament to find the best embedding models for understanding medical language.
* **Key Scripts**: `scripts/world_4_embedder_gauntlet/`
* **Status**: Phase 1 (Pair Generation and Tournaments) is complete! The champion model is **`allenai/scibert_scivocab_uncased`**.

## 🔧 Setup & Installation

Before you can unleash the full power of these magnificent machines, you'll need to set up your environment!

### 1\. Conda Environment

```bash
conda create --name dt_env python=3.9
conda activate dt_env
````

### 2\. Python Dependencies

Install all the necessary Python libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 3\. API Tokens & Keys

  * **Hugging Face:** For some of our magnificent gated models.
    1.  Create an Access Token in your Hugging Face account settings.
    2.  Create a `.env` file in the root of this project and add your token: `HUGGINGFACE_TOKEN=hf_YourSuperSecretKeyGoesHere`

## 💥 Usage: Running the Machine\!

The project is designed as a sequence of experiments.

1.  **Setup**: Run the model download scripts in `slurm_jobs/setup/`. This will download all the necessary sentence-transformer models to your scratch directory.
2.  **World 2 - Run Analysis**: Run the neighbor quality analysis by executing the launcher script: `bash slurm_jobs/world_2_neighbor_analysis/launch_world2_grid.sh`. This script will first run `compute_nearest_neighbors.py` to generate the data, followed by `examine_and_correlate_neighbors.py` to analyze it and produce our beautiful graphs\!
3.  **Continue the Workflow**: Once we have the results from World 2, we'll proceed with the experiments for World 1, and finally World 3\!

Let the SCIENCE begin\!
