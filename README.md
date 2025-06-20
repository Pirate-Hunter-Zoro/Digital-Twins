# Project Gemini: Digital Twins for Clinical Prediction - Master Briefing

**Prepared for:** Mikey Ferguson
**Prepared by:** Entrapta & The Scientific Explorers
**Last Updated:** June 20, 2025

---

## 1. Project Overview & High-Level Workflow

The core objective of this project is to predict a patient's future clinical visit based on their historical Electronic Health Record (EHR) data. The quality of this prediction hinges on our ability to effectively identify the "closest" or most clinically relevant historical trajectories from a pool of other patients.

The high-level workflow is as follows:

1. **Raw EHR Data Processing (Initial Step):** Transform raw, tabular EHR data (CSV/RDS) into a standardized, patient-centric JSON format (`all_patients_combined.json`), suitable for LLM consumption. This is a crucial prerequisite step.
2. **Define a Target Patient:** Select a patient from the processed EHR data and define their historical visit sequence up to a specific visit index (`k`).
3. **Create a Candidate Pool:** From all other patients in the dataset, generate a comprehensive library of all possible sequential historical visit sequences of length `k`.
4. **Vectorize Histories:** Convert these text-based visit sequences (both the target's and the candidates') into high-dimensional numerical vectors (embeddings) using a Sentence Transformer model (e.g., BioBERT) or TF-IDF.
5. **Find Nearest Neighbors:** Use a customizable similarity/distance metric (e.g., Cosine similarity, Euclidean distance) to compare the target patient's vector against all candidate vectors, identifying the "nearest neighbors." These neighbor identifications and their vectors are efficiently cached.
6. **Generate Prompt & Predict:** Construct a detailed prompt for a large language model (LLM), such as Med-Gemma, that includes the target patient's history and relevant information from the identified nearest neighbors. The LLM then predicts the content of the patient's `(k+1)`-th visit.
7. **Evaluate:** **(UPDATED!)** Assess the accuracy of the LLM's prediction against the actual `(k+1)`-th visit using a custom **Weighted Semantic Similarity Score**. This advanced metric moves beyond simple string matching by:
    * Calculating the semantic similarity (e.g., cosine similarity of embeddings) between predicted and actual terms.
    * Weighting each successful match by the clinical rarity of the terms, derived from their Inverse Document Frequency (IDF).
    * This provides a much more nuanced and clinically relevant measure of prediction quality than the previous Jaccard similarity baseline.

---

## 2. The Scientific Game Plan (per Dr. Paulus)

Dr. Martin Paulus provided a refined, structured, and testable version of the initial game plan.

### 2.1. Refined, Testable Problem Statement

The goal is to quantify how well different embedding and similarity pipelines capture clinically meaningful proximity between patient trajectories and to determine the amount of narrative context required for an LLM (e.g., Med-Gemma) to judge that proximity in agreement with expert clinicians.

### 2.2. Hypotheses

The project will test the following five hypotheses:

* **H1 (Geometry Match):** Distances in the embedding space will correlate monotonically ($\rho > 0.6$) with both expert similarity ratings and downstream clinical outcomes (e.g., next-visit diagnosis).
* **H2 (Off-the-Shelf vs. Tuned):** Clinical-domain transformers (e.g., ClinicalBERT, Med-Gemma) will outperform generic transformers and TF-IDF baselines on H1.
* **H3 (Representation):** Trajectory summaries that preserve temporal order will outperform unordered "bag-of-codes" representations.
* **H4 (Context Length):** The agreement between Med-Gemma's ratings and expert ratings will plateau once a sufficient context (e.g., $\ge6$ recent visits or ~2k tokens) is provided.
* **H5 (Rating Scale):** A 0-10 Likert scale for ratings will yield higher inter-rater reliability (ICC > 0.75) than a coarser 1-5 scale.

### 2.3. Data, Methods, and Evaluation

* **Data & Cohort:** The project will use a dataset of ~20,000 adult patients from the SFHS 2015-2024 database, with at least eight encounters each. Extracted features will include diagnoses, procedures, medications, lab highlights, and free-text note embeddings.
* **Trajectory Representations (R1-R5):** A variety of methods will be tested to represent patient trajectories, including TF-IDF, bag-of-codes, temporal "visit-sentences", and narrative summaries.
* **Embedding Generators (E1-E6):** Several models will be used to create the vector embeddings, including TF-IDF, Universal Sentence Encoder, BioSentVec, ClinicalBERT, Med-Gemma, and a domain-adapted version of ClinicalBERT.
* **Similarity Metrics:** The project will compare Cosine similarity, Euclidean distance, Manhattan distance, and a "Learned Metric" trained using a Siamese network. Mahalanobis distance has also been integrated for evaluating the quality of selected neighbors.
* **Evaluation Framework:** An expert panel of four clinicians will provide "gold standard" similarity ratings. These will be compared against Med-Gemma's LLM-generated ratings. Key metrics will include Spearman correlation for predictive power and Inter-rater Reliability (ICC) to measure agreement between the LLM and experts.

---

## 3. Key Concepts from Academic Literature

* **Digital Twin Creation:** The project's methodology aligns with modern approaches for creating clinical digital twins. Papers like **TWIN-GPT** and **DT-GPT** describe similar frameworks where Large Language Models are used to generate high-fidelity, personalized digital twins of patients by leveraging their historical EHR data along with data from the most similar patients (nearest neighbors) to predict future health trajectories. This is often done in an auto-regressive manner, predicting one step at a time.
* **Semantic Embeddings:** A core challenge in EHR analysis is handling heterogeneous data (structured codes, unstructured notes, etc.). LLM-derived **embeddings** provide a powerful solution by transforming these varied data types into a unified semantic space, where concepts with similar clinical meanings are represented by vectors that are close to each other. This allows for more effective quantitative analysis and is a foundational step for building predictive models.
* **Synthetic Controls:** The concept of using nearest neighbors as context is analogous to **Synthetic Control Methods** from econometrics, where a comparator group is constructed from a weighted average of untreated units that best resemble the characteristics of the treated unit. The project uses a high-dimensional, LLM-based version of this concept. The **Penalized Synthetic Control** paper further refines this by introducing a penalty to trade off aggregate matching discrepancy against pairwise discrepancy, aiming to reduce interpolation bias.

---

## 4. Technical Guide: LIBR 'c3' Analysis Cluster

### 4.1. Core Architecture & Access

* **Cluster Name:** `c3`.
* **Access:** Connection is made via SSH to the login node: `ssh -YC submit0.laureateinstitute.org`.
* **Nodes:** The cluster consists of one **Login Node** (`submit0`) and six **Compute Nodes** (`compute300-305`).
* **Golden Rule:** The login node is **ONLY** for submitting jobs. All intensive computations must be run on the compute nodes via the scheduler.
* **Hardware:** Each of the 6 compute nodes is a Dell PowerEdge R750 equipped with 2x Intel Xeon Gold 6342 CPUs, 1 TB of RAM, and 1 NVIDIA A40 GPU.
* **Software:** The cluster uses **Slurm** as the job scheduler and **Lmod** for application environment management.

### 4.2. Job Submission & Management (Slurm)

#### No changes to this section (Job Submission & Management)

### 4.3. Storage Policy

#### No changes to this section (Storage Policy)

### 4.4. Software & Python Environment Workflow

#### No changes to this section, with one addition below

* **Note on Hugging Face Model Downloads:** The `c3` cluster's network security may block direct, large file downloads from Hugging Face, even from the login node. **(NEW!)** The recommended robust workflow is to:
    1. Download the model files on a local machine with unrestricted internet access (using the `download_model.py` script).
    2. Compress the resulting model directory into a single `.tar.gz` file.
    3. Transfer the single file to the cluster using `scp`.
    4. Decompress the file on the cluster.
    5. Point your scripts to this local model path and use the `local_files_only=True` parameter when loading.

---

## 5. Overview of Project Python Scripts

* `main.py`: The main entry point for the entire pipeline. It handles parallel processing of patients using `multiprocessing`.
* `process_data.py`: **NEW!** This critical script is responsible for the initial processing of raw, tabular EHR data into the standardized `all_patients_combined.json` format.
* `load_patient_data.py`: **NEW!** Acts as an adapter to load `all_patients_combined.json` and prepare its structure for downstream scripts.
* `compute_nearest_neighbors.py`: A flexible script for vectorizing visit histories and finding nearest neighbors using customizable models and distance metrics.
* **`generate_idf_registry.py`:** **(NEW!)** A one-time setup script that processes the entire `all_patients_combined.json` dataset to calculate and save the Inverse Document Frequency (IDF) for every unique medical term. The output (`term_idf_registry.json`) is a crucial input for the new evaluation metric.
* **`generate_technique_embeddings.py`:** **(NEW!)** A one-time setup script that loads a pre-trained transformer model (e.g., BioBERT) and generates a vector embedding for every unique medical term found in the dataset. The output (`term_embedding_library.pkl`) is the second crucial input for the new evaluation metric, enabling semantic similarity calculations.
* `llm_helper.py`: Contains utility functions for LLM interaction.
* `examine_nearby_patients.py`: Handles detailed analysis of nearest neighbors, including LLM relevance scores and Mahalanobis distance.
* `query_and_response.py`: Responsible for constructing and parsing LLM prompts and responses.
* `process_patient.py`: The core function called by `main.py` for each patient's prediction and evaluation workflow.
* `evaluate.py`: **(REFACTORED!)** No longer uses Jaccard similarity. This script now implements the advanced **Weighted Semantic Similarity Score**. It loads the pre-computed IDF and embedding libraries to calculate a normalized score based on semantic similarity and term rarity.
* `query_llm.py`: A low-level wrapper for making API calls to the locally hosted LLM.
* `download_model.py`: **(UPDATED PURPOSE!)** A utility script now intended to be run on a **local machine** with unrestricted internet access. It downloads and saves models from Hugging Face to facilitate the manual transfer method required to bypass cluster network firewalls.
* `visualize_results.py`: **(REFACTORED!)** A powerful post-processing script that loads results into a Pandas DataFrame. It generates two types of visualizations: 1) A comprehensive **box plot** showing the full distribution of the Weighted Similarity Scores across all patients, and 2) A folder of **individual bar charts**, providing a detailed report for each patient.
* `config.py`: Centralized Project Configuration.
* `parser.py`: Argument parsing for `main.py`.
* `check_patient_overlap.py`: A diagnostic script for validating data overlap.
* `calculate_spearmans_rho.py`: **NEW!** A dedicated script to compute and visualize the Spearman correlation between LLM relevance scores and Mahalanobis distances.
