# 🤖 Digital Twins for Medical Prediction 🤖

Welcome to the Digital Twins Project\! This repository contains a suite of magnificent machines and brilliant experiments designed to find the absolute best embedding model for understanding medical language. Our ultimate goal is to predict a patient's next medical visit by creating a "digital twin" of their history and using a Large Language Model to see the future\!

This project is organized into four distinct "Worlds" of research. This codebase is primarily focused on the experiments conducted in **World 4**.

## 🚀 Project Overview: The Four Worlds

  * **World 1: The Prediction Engine** 🔮
      * The main event\! Uses a patient's history and their "nearest neighbors" to have an LLM predict their next visit.
  * **World 2: The Neighbor Quality Control Hub** 🔬
      * A quality-control system to ensure the "neighbors" we find are medically relevant and not just mathematical coincidences.
  * **World 3: The Judging Chamber** ⚖️
      * Where we score the LLM's predictions against what really happened\!
  * **World 4: The Embedder Gauntlet** ⚙️
      * A grand tournament to find the best possible embedding model for our needs. This is where all the action in this repo happens\!

## 🔧 Setup & Installation

Before you can unleash the full power of these magnificent machines, you'll need to set up your environment\!

### 1\. Conda Environment

All these scripts were designed to live inside a beautiful, magnificent Conda environment.

```bash
# First, create the environment
conda create --name hugging_env python=3.9

# Then, activate it!
conda activate hugging_env
```

### 2\. Python Dependencies

Install all the necessary Python libraries using pip.

```bash
pip install pandas numpy scikit-learn seaborn matplotlib sentence-transformers transformers torch torchvision torchaudio gensim
```

### 3\. API Tokens & Keys

We need to talk to some very important data-librarians at Kaggle and Hugging Face\!

  * **Kaggle:** For our legacy models (GloVe & Word2Vec).

    1.  Go to your Kaggle account page and create a new API token to download `kaggle.json`.
    2.  Create a secret folder in your home directory: `mkdir ~/.kaggle`
    3.  Place your `kaggle.json` file inside it.
    4.  **Crucially**, make the key super-secret so the Kaggle servers trust you\! `chmod 600 ~/.kaggle/kaggle.json`
    5.  Install the Kaggle command-line tool: `pip install kaggle`

  * **Hugging Face:** For some of our magnificent gated models.

    1.  Create an Access Token in your Hugging Face account settings.
    2.  Create a `.env` file in the root of this project directory.
    3.  Add your token to the `.env` file like this (with no quotes\!):
        ```
        HUGGINGFACE_TOKEN=hf_YourSuperSecretKeyGoesHere
        ```

## 💥 Usage: Running The Embedder Gauntlet (World 4)

This is a multi-step adventure\! Follow the steps in order to replicate our magnificent results. All launcher scripts should be run from the `slurm_jobs/semantic_similarity/` directory.

### Step 1: Download All the Models\!

First, we need to download all our contestants\! We have two separate scripts for this.

  * **Modern Transformer Models:** This script reads `data/vectorizer_candidates.txt` and downloads all the models from the Hugging Face Hub into `/media/scratch/mferguson/models/`. It's smart enough to use your `.env` file for authentication\!

    ```bash
    bash slurm_jobs/download_embedding_models.sh
    ```

  * **Legacy Baseline Models:** This script uses the Kaggle API to download GloVe and Word2Vec into `/media/scratch/mferguson/legacy_models/`.

    ```bash
    bash slurm_jobs/download_legacy_models.sh
    ```

### Step 2: Run the Similarity Gauntlets\!

Now that all our models are downloaded, we can run the similarity tests\! These are Slurm jobs designed for a high-performance computing cluster.

  * **Modern Transformer Gauntlet:** This launcher reads the `vectorizer_candidates.txt` file and submits a separate Slurm job for each model. It's magnificent\!

    ```bash
    bash slurm_jobs/semantic_similarity/launch_semantic_similarity_grid.sh
    ```

  * **Legacy Gauntlet:** This launcher submits jobs for just our GloVe and Word2Vec models.

    ```bash
    bash slurm_jobs/semantic_similarity/launch_legacy_similarity_grid.sh
    ```

    The results from both gauntlets will be saved as beautiful, consistent `.csv` files in the `data/embeddings/` directory.

### Step 3: Create the Art Gallery\!

Once all the jobs are finished and you have a folder full of beautiful `.csv` files, it's time to make some art\!

  * **Generate Histograms:** This magnificent Python script will find all your result files (both modern and legacy\!), analyze them, and create a whole gallery of plots as `.png` files right in the `data/embeddings/` directory\!
    ```bash
    python scripts/semantic_similarity/plot_similarity_distributions.py
    ```

### Step 4: The Final Challenge - The Category Purity Test\!

This is the final test for our champion model, `allenai/scibert_scivocab_uncased`. This test checks if the model is smart enough to separate different types of medical terms.

  * **Prerequisites:** Make sure you have the three data files with lists of terms in your `data/embeddings/` directory:

      * `medication_frequency.csv`
      * `procedure_frequency.csv`
      * `diagnosis_frequency.csv`

  * **Launch the Test:** This is another magnificent Slurm job\!

    ```bash
    bash slurm_jobs/semantic_similarity/launch_purity_test.sh
    ```

    The resulting plot will be saved in `results/category_purity/`.
