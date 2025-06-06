# Project Gemini: Digital Twins for Clinical Prediction - Master Briefing

**Prepared for:** Mikey Ferguson
**Prepared by:** Koro Sensei & The Netherworld Assembly
**Last Updated:** June 6, 2025

---

## 1. Project Overview & High-Level Workflow

The core objective of this project is to predict a patient's future clinical visit based on their historical Electronic Health Record (EHR) data. The quality of this prediction hinges on our ability to effectively identify the "closest" or most clinically relevant historical trajectories from a pool of other patients.

The high-level workflow is as follows:
1.  **Define a Target:** Select a patient and their visit history up to visit `k`.
2.  **Create a Candidate Pool:** From all other patients in the dataset, generate a comprehensive library of all possible visit sequences of length `k`.
3.  **Vectorize Histories:** Convert the text-based visit sequences (both the target's and the candidates') into high-dimensional numerical vectors (embeddings) using a Sentence Transformer model like BioBERT.
4.  [cite_start]**Find Nearest Neighbors:** Use a similarity metric (e.g., cosine similarity) to compare the target patient's vector against all candidate vectors, identifying the "nearest neighbors".
5.  [cite_start]**Generate Prompt & Predict:** Construct a detailed prompt for a large language model (LLM), such as Med-Gemma, that includes the target patient's history and information from the identified nearest neighbors. The LLM then predicts the content of the patient's `(k+1)`-th visit.
6.  **Evaluate:** Assess the accuracy of the LLM's prediction against the actual `(k+1)`-th visit using Jaccard similarity for categories like 'diagnoses', 'medications', and 'treatments'.

---

## 2. The Scientific Game Plan (per Dr. Paulus)

Dr. Martin Paulus provided a refined, structured, and testable version of the initial game plan.

### 2.1. Refined, Testable Problem Statement
[cite_start]The goal is to quantify how well different embedding and similarity pipelines capture clinically meaningful proximity between patient trajectories and to determine the amount of narrative context required for an LLM (e.g., Med-Gemma) to judge that proximity in agreement with expert clinicians.

### 2.2. Hypotheses
The project will test the following five hypotheses:
* [cite_start]**H1 (Geometry Match):** Distances in the embedding space will correlate monotonically ($\rho > 0.6$) with both expert similarity ratings and downstream clinical outcomes (e.g., next-visit diagnosis).
* [cite_start]**H2 (Off-the-Shelf vs. Tuned):** Clinical-domain transformers (e.g., ClinicalBERT, Med-Gemma) will outperform generic transformers and TF-IDF baselines on H1.
* [cite_start]**H3 (Representation):** Trajectory summaries that preserve temporal order will outperform unordered "bag-of-codes" representations.
* [cite_start]**H4 (Context Length):** The agreement between Med-Gemma's ratings and expert ratings will plateau once a sufficient context (e.g., $\ge6$ recent visits or ~2k tokens) is provided.
* [cite_start]**H5 (Rating Scale):** A 0-10 Likert scale for ratings will yield higher inter-rater reliability (ICC > 0.75) than a coarser 1-5 scale.

### 2.3. Data, Methods, and Evaluation
* **Data & Cohort:** The project will use a dataset of ~20,000 adult patients from the SFHS 2015-2024 database, with at least eight encounters each. [cite_start]Extracted features will include diagnoses, procedures, medications, lab highlights, and free-text note embeddings.
* [cite_start]**Trajectory Representations (R1-R5):** A variety of methods will be tested to represent patient trajectories, including TF-IDF, bag-of-codes, temporal "visit-sentences", and narrative summaries.
* [cite_start]**Embedding Generators (E1-E6):** Several models will be used to create the vector embeddings, including TF-IDF, Universal Sentence Encoder, BioSentVec, ClinicalBERT, Med-Gemma, and a domain-adapted version of ClinicalBERT.
* [cite_start]**Similarity Metrics:** The project will compare Cosine similarity, Euclidean distance, Manhattan distance, and a "Learned Metric" trained using a Siamese network.
* **Evaluation Framework:** An expert panel of four clinicians will provide "gold standard" similarity ratings. These will be compared against Med-Gemma's LLM-generated ratings. [cite_start]Key metrics will include Spearman correlation for predictive power and Inter-rater Reliability (ICC) to measure agreement between the LLM and experts.

### 2.4. Timeline & Success Criteria
* [cite_start]The project is planned over a ~12-week timeline with clear milestones for data extraction, model development, and analysis.
* [cite_start]Success is defined by clear, pre-specified criteria, such as achieving a correlation ($\rho$) of $\ge0.7$ and an ICC of $\ge0.8$.

---

## 3. Key Concepts from Academic Literature

* **Digital Twin Creation:** The project's methodology aligns with modern approaches for creating clinical digital twins. [cite_start]Papers like **TWIN-GPT** and **DT-GPT** describe similar frameworks where Large Language Models are used to generate high-fidelity, personalized digital twins of patients by leveraging their historical EHR data along with data from the most similar patients (nearest neighbors) to predict future health trajectories. [cite_start]This is often done in an auto-regressive manner, predicting one step at a time.
* **Semantic Embeddings:** A core challenge in EHR analysis is handling heterogeneous data (structured codes, unstructured notes, etc.). [cite_start]LLM-derived **embeddings** provide a powerful solution by transforming these varied data types into a unified semantic space, where concepts with similar clinical meanings are represented by vectors that are close to each other. [cite_start]This allows for more effective quantitative analysis and is a foundational step for building predictive models.
* [cite_start]**Synthetic Controls:** The concept of using nearest neighbors as context is analogous to **Synthetic Control Methods** from econometrics, where a comparator group is constructed from a weighted average of untreated units that best resemble the characteristics of the treated unit. The project uses a high-dimensional, LLM-based version of this concept. [cite_start]The **Penalized Synthetic Control** paper further refines this by introducing a penalty to trade off aggregate matching discrepancy against pairwise discrepancy, aiming to reduce interpolation bias.

---

## 4. Technical Guide: LIBR 'c3' Analysis Cluster

### 4.1. Core Architecture & Access
* [cite_start]**Cluster Name:** `c3`.
* [cite_start]**Access:** Connection is made via SSH to the login node: `ssh -YC submit0.laureateinstitute.org`.
* [cite_start]**Nodes:** The cluster consists of one **Login Node** (`submit0`) and six **Compute Nodes** (`compute300-305`).
* **Golden Rule:** The login node is **ONLY** for submitting jobs. [cite_start]All intensive computations must be run on the compute nodes via the scheduler.
* [cite_start]**Hardware:** Each of the 6 compute nodes is a Dell PowerEdge R750 equipped with 2x Intel Xeon Gold 6342 CPUs, 1 TB of RAM, and 1 NVIDIA A40 GPU.
* [cite_start]**Software:** The cluster uses **Slurm** as the job scheduler and **Lmod** for application environment management.

### 4.2. Job Submission & Management (Slurm)
* **Partitions:** Jobs are submitted to logical groups of nodes called partitions. [cite_start]The primary partitions are `c3` (for general use, up to 168 hours) and `c3_short` (for shorter jobs, up to 9 hours, with higher priority).
* **Submission Scripts:** Jobs are submitted using `sbatch` with a submission script (e.g., `job.ssub`). [cite_start]This script must contain `#SBATCH` directives to request resources.
* **Example `#SBATCH` Directives:**
    ```bash
    #SBATCH --partition=c3
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=2
    #SBATCH --mem=24000         # Memory in MB
    #SBATCH --time=01:00:00
    #SBATCH --output=~/logs/job-%J.stdout
    #SBATCH --error=~/logs/job-%J.stderr
    ```
* **Key Commands:**
    * [cite_start]`sbatch <script.ssub>`: Submit a job.
    * `squeue -u $USER`: View your jobs.
    * [cite_start]`scancel <jobid>`: Cancel a job.
    * [cite_start]`srun --pty bash`: Start an interactive job on a compute node.

### 4.3. Storage Policy
* **Home Directory (`~/`):** For configurations and small files. [cite_start]It has a **100GB quota**. [cite_start]This space is shared across all nodes.
* [cite_start]**Lab Storage (`/media/labs/`):** This is for all primary study data, including source data, intermediate files, and final results.
* **Scratch Space (`/media/scratch/`):** For temporary files only. [cite_start]Data here is **purged after 30 days** of inactivity.

### 4.4. Software & Python Environment Workflow
* **Module System:** Use `module avail` to see available software and `module load <name>` to load it into your environment.
* [cite_start]**Posit Workbench:** A web portal at `workbench.laureateinstitute.org` for running RStudio, Jupyter, and VS Code sessions directly on the cluster.
* **Python Virtual Environments:** This is the recommended workflow for managing project dependencies:
    1.  [cite_start]Load the desired system Python module: `module load Python/3.11.5-GCCcore-13.2.0`.
    2.  [cite_start]Create a virtual environment: `virtualenv my-project-env`.
    3.  Activate it: `source my-project-env/bin/activate`.
    4.  [cite_start]Install packages, including `ipykernel`: `pip install -r requirements.txt`.
    5.  [cite_start]Register the environment as a Jupyter kernel: `python -m ipykernel install --name "MyProject" --user`.
    6.  [cite_start]In Posit Workbench, start a Jupyter session, load the base Python module in the "Softwares" tab, then create a new notebook using your "MyProject" kernel.

---

## 5. Overview of Project Python Scripts

* `main.py`: The main entry point for the entire pipeline. It handles parallel processing of patients using `multiprocessing`.
* `generate_patients.py`: Creates synthetic patient data for testing and development, which can be loaded via `load_patient_data()`.
* `compute_nearest_neighbors.py`: Contains the logic to convert visit histories into strings, use a `SentenceTransformer` (BioBERT) to generate vector embeddings, and compute the cosine similarity matrix to find and save the nearest neighbors for each patient's visit sequence. Caches results to `neighbors.pkl`.
* `query_and_response.py`: Responsible for all LLM interaction. `generate_prompt` constructs the detailed prompt using patient history and nearest neighbor data. `parse_llm_response` uses regex to robustly parse the LLM's output. `force_valid_prediction` includes a retry loop to ensure a usable response.
* `process_patient.py`: The core function called by `main.py` for each patient. It orchestrates the process of generating a prompt, querying the LLM, and evaluating the response for each visit.
* `evaluate.py`: Calculates the Jaccard similarity scores to measure the performance of the predictions.
* `query_llum.py`: A low-level wrapper for making API calls to the locally hosted LLM. It also handles response cleaning and logging for debugging.
* `download_model.py`: A utility script to download the required LLM models from Hugging Face.
* `visualize_results.py`: A post-processing script to generate plots of the Jaccard scores over time for each patient.