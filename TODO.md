## 🧠 Digital Twin Project – Updated TODOs & Research Leads

---

### ✅ 1. Look into Sentence Similarity  
**Be wary of all nearby neighbors coming from same patient - concatenate contiguous sequences of visits coming from same patient - have system search for X number of patients; not X number of windows - keep polling until you reach desired number of patients and concatenate if you've already seen the patient
**Goal:** Improve patient similarity with sentence embeddings.  
**Augmented Insight:**  
- Test biomedical-specific models (`BioBERT`, `PubMedBERT`, `BioMistral`)  
- Evaluate LLM-derived embeddings vs `sentence-transformers`.

---

### 💉 2. Convert ICD9 and TempCodes into Descriptions  
**Goal:** Improve interpretability by mapping codes.  
**Augmented Insight:**  
- Map to descriptions and embed those for semantic similarity.
- Use these in prompts and nearest-neighbor comparisons.

---

### 🪙 3. Weighted Nearest Neighbors (WNN)  
**Goal:** Let LLM know neighbor similarity.  
**Augmented Insight:**  
- Prompt with weighted textual summaries or embed composite profiles.

---

### ⚡ 4. Use Normed Distance for Fast Local Search  
**Goal:** Optimize lookup speed.  
**Augmented Insight:**  
- Distill large LLM embeddings for fast local FAISS lookups.
- Use hybrid: fast shortlist + LLM refinement.

---

### 🔁 5. Look for Any Similar Sequence of Visits  
**Goal:** Sliding window across visit subsequences.  
**Augmented Insight:**  
- Embed subsequences as “visit narratives” for RAG or prediction.

---

### 📚 6. LangChain / RAG Integration  
**Goal:** Use RAG for visit prediction.  
**Augmented Insight:**  
- Use LangChain + semantic retrievers.
- Prompt LLMs with retrieved sequences to generate plausible next visits or outcomes.

---

### 🔋 7. Use More Powerful LLM  
**Goal:** Improve generation quality.  
**Augmented Insight:**  
- Add `BioMistral`, `ClinicalGPT`, `Mixtral` to your benchmark set.
- Test zero-shot capabilities and lab forecast quality.

---

## 🆕 New TODOs from GDR and DT-GPT Papers

---

### 🧬 8. Forecast Lab Trajectories (DT-GPT Style)  
**Goal:** Predict future lab values from past visit data.  
**Steps:**  
- Format input as chronological visit text.  
- Fine-tune LLM on NSCLC or MIMIC-style data.  
- Measure MAE and correlation.

---

### 🗣️ 9. Chatbot Interpretability Interface  
**Goal:** Allow natural language queries about predictions.  
**Steps:**  
- Add prompts like "Which features influenced this prediction?"  
- Use template-based summarization or embedding similarity for variable salience.

---

### 📊 10. Zero-Shot Variable Forecasting  
**Goal:** Test generalization by predicting unseen variables.  
**Steps:**  
- Train on a subset of labs.  
- Prompt to predict a different lab not seen during training.  
- Compare to LightGBM.

---

### ⚖️ 11. Bias and Fairness Audits of Synthetic Controls  
**Goal:** Detect and mitigate demographic bias.  
**Steps:**  
- Measure Statistical Parity Difference (SPD) across groups.  
- Adjust generation or post-process to rebalance.

