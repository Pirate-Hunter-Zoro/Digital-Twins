# 🤖 Digital Twins for Medical Prediction 🤖

Welcome to the Digital Twins Project\! This repository contains a suite of magnificent machines and brilliant experiments designed to find the absolute best way to represent a patient's medical history. Our ultimate goal is to predict a patient's next medical visit by creating a "digital twin" and using a Large Language Model to see the future\!

This project is organized into four distinct "Worlds" of research, each with its own dedicated directory for scripts and Slurm jobs.

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

### **World 2: The Neighbor Quality Control Hub** 🔬

  * **Goal**: To build, train, and validate a powerful, custom hierarchical embedding model that understands patient trajectories over time. The ultimate goal is to use this trained model to find both the nearest (most similar) and farthest (most dissimilar) patients to test our similarity metrics comprehensively.
  * **Key Scripts**:
      * `scripts/world_2_neighbor_analysis/training/train_hierarchical_encoder.py`: The Training Gymnasium\! This script builds and trains our new, custom `HierarchicalPatientEncoder` model on the extrinsic task of predicting 30-day readmission. This is a crucial first step.
      * `scripts/world_2_neighbor_analysis/compute_nearest_neighbors.py`: The Flexible Behemoth\! After the encoder is trained, this machine generates smart, time-aware vectors for all patient histories. It then calculates the similarity to every other patient and saves the **entire ranked list** of neighbors, from nearest to farthest, for maximum flexibility.
      * `scripts/world_2_neighbor_analysis/examine_and_correlate_neighbors.py`: The Quality Control-inator\! This script loads the full ranked lists of neighbors and intelligently analyzes the **k-nearest** vs. the **k-farthest** pairs. It compares their vector similarity to an LLM-judged clinical similarity score to see if our metrics work across the full spectrum of patient relationships.
      * `scripts/world_2_neighbor_analysis/plot_individual_patient_correlations.py`: The Multi-Scope Inspector\! This machine creates a grid of plots to visualize the correlation results for a few individual patients, helping us understand if patterns are consistent or patient-specific.
  * **Launcher**: `slurm_jobs/world_2_neighbor_analysis/launch_world2_pipeline_grid.sh`
  * **Status**: **ACTIVE\!** We have designed a new, automated, multi-stage pipeline to train our custom model, compute all neighbors, and perform multiple layers of analysis.

### **World 3: The Judging Chamber** ⚖️

  * **Goal**: To score the LLM's predictions from World 1 against the patient's actual, real-life next visit.
  * **Key Scripts**: `scripts/world_3_judging_chamber/`
  * **Status**: On Hold

### **World 4: The Embedder Gauntlet** ⚙️

  * **Goal**: A grand tournament to find the best base embedding models for understanding medical language. The winner of this gauntlet is used as the foundational term embedder in our World 2 hierarchical model.
  * **Key Scripts**: `scripts/world_4_embedder_gauntlet/`
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
3.  **World 2 - Run Full Analysis Pipeline**: Once you have a champion vectorizer from World 4, you can run the entire World 2 pipeline with a single command. This script will automatically handle the multi-stage process in the correct order:
    1.  Train the custom `HierarchicalPatientEncoder`.
    2.  Use the trained encoder to compute and save the full ranked list of neighbors for every patient.
    3.  Examine the k-nearest and k-farthest neighbors and generate correlation plots.
    4.  Generate detailed plots for individual patient neighborhoods.
    <!-- end list -->
    ```bash
    bash slurm_jobs/world_2_neighbor_analysis/launch_world2_pipeline_grid.sh
    ```
4.  **Continue the Workflow**: Once we have a robust method for finding neighbors, we can proceed with the experiments for World 1, and finally World 3\!

Let the SCIENCE begin\!
