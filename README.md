# TODO List for Digital Twins Project

---

## PHASE 3: Agentic Workflow Redesign (Inspired by otto-SR)

This is the next major evolution of our project. Instead of a single, monolithic prompt-building process, we will re-architect the system into a pipeline of specialized agents.

- **[ ] Upgrade Module 1: Build the "Neighbor-Selection Protocol" (Screener Agent)**
  - [ ] Define a detailed schema for inclusion/exclusion criteria for what constitutes a "good" neighbor patient. This goes beyond simple vector similarity.
  - [ ] Implement a "Screener Agent" that takes a target patient and the protocol, and searches the entire patient database to return a high-quality list of neighbor IDs.

- **[ ] Upgrade Module 2: Build the "Data-Extraction Engine" (Extractor Agent)**
  - [ ] Create a dedicated "Extractor Agent" that can process a patient's full visit history.
  - [ ] Instead of creating a long narrative string, this agent's job is to pull out specific, structured key-value pairs (e.g., `diagnosis: "Hypertension"`, `medication: "Lisinopril"`).
  - [ ] The goal is to create a concise, data-rich, and token-efficient representation of the patient's history.

- **[ ] The Grand Redesign: Implement the "Agentic Assembly Line"**
  - [ ] **Agent 1 (Screener):** Takes target patient -> outputs perfect neighbor list.
  - [ ] **Agent 2 (Extractor):** Takes target + neighbors -> outputs structured, token-efficient data for all of them.
  - [ ] **Agent 3 (Synthesizer):** Takes the structured data from the Extractor -> builds the final, optimized prompt for the prediction model.
  - [ ] Refactor `main.py` and `process_patient.py` to orchestrate this new three-agent pipeline.

---

## PHASE 2: Performance & Evaluation

- **[ ] Analyze results from the "token fix" experiments.**
  - [ ] Did reducing `max_tokens` in `query_llm.py` solve the issue?
  - [ ] Did truncating the `history_section` in `query_and_response.py` prevent overloads for patients with long histories?
  - [ ] How did reducing `--num_neighbors` affect performance and accuracy?
- **[x] Visualize Results:** Enhance the `visualize_results.py` script to generate more insightful plots.
  - [x] Box plots for score distributions.
  - [x] Individual report plots for each patient.
- **[x] Calculate Spearman's Rho:** Create a script to measure the correlation between LLM relevance scores and Mahalanobis distance to see if they align.

---

## PHASE 1: Initial Implementation & Data Processing

- **[x] Fix Token Overload Errors:** Investigate and resolve the `BadRequestError` related to exceeding the model's maximum context length.
- **[x] Implement Weighted Scoring:** Replace the Jaccard similarity score with a more sophisticated weighted score using IDF and semantic similarity (cosine similarity on embeddings).
- **[x] Compute Nearest Neighbors:** Develop a script to find the nearest neighbors for each patient's visit sequence.
- **[x] Generate Term Embeddings & IDF:** Create scripts to pre-calculate and save term embeddings (using BioBERT) and IDF scores for all medical terms in the dataset.
- **[x] Process Raw Data:** Convert the raw CSV/SQLite data into a structured JSON format that's easier to work with.
