# 🤖 Digital Twins for Medical Prediction 🤖

Welcome to the Digital Twins Project\! This repository contains a suite of magnificent machines and brilliant experiments designed to find the absolute best way to represent a patient's medical history. Our ultimate goal is to predict a patient's next medical visit by creating a "digital twin" and using a Large Language Model to see the future\!

## 🏛️ Project Architecture

Our laboratory is now perfectly organized for maximum efficiency\! The main components are:

  * **`data/`**: This is where all the magnificent raw data, intermediate files, and final results live\! It holds our patient information, concept frequency lists, and the outputs of our experiments. Because this directory may contain confidential patient information, it is listed in the `.gitignore`. The subdirectories are now dynamically named based on the parameters of each experiment for ultimate organization\!
  * **`scripts/`**: This is the brain of the operation\! It contains all the Python scripts that do the heavy lifting, neatly organized into a `common` directory for shared tools and a directory for each of the four "Worlds."
  * **`slurm_jobs/`**: This is our grand command center\! It contains all the launcher (`.sh`) and template (`.ssub`) files needed to submit jobs to a Slurm-based high-performance computing cluster. It's also perfectly organized by "Worlds"\!
  * **`References/`**: A library of all the brilliant research papers and notes that inspired this magnificent project\!

## 🚀 The Four Worlds

Our research is divided into four interconnected worlds, each building upon the last. The official workflow is **World 4 ➡️ World 2 ➡️ World 1 ➡️ World 3**.

### **World 1: The Prediction Engine** 🔮

  * **Goal**: To take a single patient, build their digital twin, and use an LLM to generate a prediction for their next medical visit.
  * **Status**: On Hold

### **World 2: The Similarity Metric Validation Hub** 🔬

  * **Goal**: To rigorously compare different patient vectorization methods (e.g., our custom GRU model vs. a large Transformer) and different similarity metrics (e.g., Cosine Similarity, Euclidean Distance, LLM Semantic Similarity) to see if they agree on which patients are similar or dissimilar.
  * **Key Scripts**:
      * `scripts/common/models/training/train_gru_embedder.py`: The Training Gymnasium\! This script trains our custom `HierarchicalPatientEncoder` model on the extrinsic task of predicting 30-day readmission.
      * `scripts/world_2_neighbor_analysis/vectorizers/run_vectorization.py`: The Vectorization Factory\! A scalable, modular script that takes patient data and an embedder type (`gru` or `transformer`) and creates the final patient vector files.
      * `compute_distance_metrics.py`: The Universal Metric-Calculating Behemoth\! This machine takes the vector files and computes all pairwise metrics (cosine similarity, euclidean distance, LLM score) for every possible pair.
      * `plot_metrics.py`: The Correlation Matrix Megaplotter\! This is our data-art machine\! It reads the metrics file and generates beautiful scatter plots and heatmaps to visualize the relationships between the different similarity metrics.
  * **Launcher**: `slurm_jobs/world_2_neighbor_analysis/launch_world2_pipeline_grid.sh`
  * **Status**: **ACTIVE\!** We have designed a new, automated, multi-stage pipeline to vectorize the data, compute all pairwise metrics, and then generate a full suite of analytical plots.

### **World 3: The Judging Chamber** ⚖️

  * **Goal**: To score the LLM's predictions from World 1 against the patient's actual, real-life next visit.
  * **Status**: On Hold

### **World 4: The Embedder Gauntlet** ⚙️

  * **Goal**: A grand tournament to find the best base embedding models for understanding medical language. The winner of this gauntlet is used as the foundational term embedder in our World 2 GRU model.
  * **Launcher**: `slurm_jobs/world_4_embedder_gauntlet/launch_full_gauntlet.sh`
  * **Status**: Ready for launch\!

## 🔧 Setup & Installation

Before you can unleash the full power of these magnificent machines, you'll need to set up your environment\!

### 1\. Conda Environment

```bash
conda create --name dt_env python=3.9
conda activate dt_env
```

### 2\. Python Dependencies

Install all the necessary Python libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## 💥 Usage: Running the Machine\!

The project is designed as a sequence of experiments that build on each other.

1.  **Setup**: Run the model download scripts in `slurm_jobs/setup/`. This will download all the necessary sentence-transformer and generative models to your scratch directory.
2.  **World 4 - Run Embedder Gauntlet**: Run the grand tournament to determine the best base model for encoding medical terms.
    ```bash
    bash slurm_jobs/world_4_embedder_gauntlet/launch_full_gauntlet.sh
    ```
3.  **World 2 - Run Full Analysis Pipeline**: Once you have a champion vectorizer from World 4, you can run the entire World 2 pipeline with a single command. This script will automatically handle the new, magnificent three-stage process in the correct order for each embedder type we want to test:
    1.  **Vectorize** all patient histories using the specified embedder (`gru` or `transformer`).
    2.  **Compute Metrics** for every single pair of patients (cosine, euclidean, and LLM score).
    3.  **Analyze & Plot** a random sample of the metrics, generating the final scatter plots and heat maps.
    <!-- end list -->
    ```bash
    bash slurm_jobs/world_2_neighbor_analysis/launch_world2_pipeline_grid.sh
    ```
4.  **Continue the Workflow**: Once we have a robust method for creating and validating patient vectors, we can proceed with the experiments for World 1, and finally World 3\!

Let the SCIENCE begin\!
