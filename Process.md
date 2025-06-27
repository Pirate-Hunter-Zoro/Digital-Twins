# 🔍 World One: Evaluation Pipeline for LLM Visit Predictions

## 🧠 Objective

Evaluate how well an LLM-predicted patient visit matches the actual next visit, by comparing their **terms** using semantic similarity and importance (IDF).

---

## 📦 Inputs (per patient visit)

- 🗃️ **Predicted terms**: From LLM output
- 🏥 **Actual terms**: From real patient visit
- 🧠 **Term Embeddings**: from `term_embedding_library_mpnet_combined.pkl`
- 📈 **IDF values**: from `term_idf_registry.json`

Categories involved:

- `diagnoses`
- `medications`
- `treatments`

---

## ⚙️ Evaluation Logic per Category

Step 1: Clean and filter terms (must have embeddings)

Step 2: Compute cosine similarity matrix between
actual terms and predicted terms

Step 3: Sort actual terms by descending IDF

Step 4: For each actual term in that order:

```text
┌────────────────────────────────────────────┐
│ • Find the unmatched predicted term        │
│   with the highest cosine similarity       │
│                                            │
│ • Score = similarity * sqrt(idf_actual *   │
│   idf_predicted)                           │
│                                            │
│ • Remove both terms from future matching   │
└────────────────────────────────────────────┘
```

Step 5: Sum all match scores for that category

Step 6: Any leftover unmatched terms contribute 0

Step 7: Repeat for all 3 categories

---

## 🧮 Final Scores

- `category_score = sum(all weighted matches)`
- `overall_score = mean([diagnoses, medications, treatments])`

---

## 📊 Example Match Table

| Actual Term             | IDF  | Matched Prediction        | IDF  | Cosine Sim | Weighted Score           |
|-------------------------|------|----------------------------|------|-------------|---------------------------|
| Pneumonia               | 4.2  | Bacterial lung infection   | 3.8  | 0.72        | 0.72 × √(4.2×3.8) ≈ 2.88 |
| Albuterol Sulfate       | 5.0  | Albuterol                  | 5.5  | 0.95        | 0.95 × √(5.0×5.5) ≈ 4.68 |
| Mammogram Screening     | 6.1  | —                          | —    | —           | 0.00                     |

---

## ✅ Output

A dict like:

```python
{
  "diagnoses": 4.2,
  "medications": 3.1,
  "treatments": 1.9,
  "overall": 3.07
}
```

---

## 🧠 Why This Works

- 🧠 Uses semantic meaning (via embeddings) to match terms that might not be literal string matches
- 🔍 Prioritizes **important** terms (via IDF)
- 🎯 Greedy matching ensures each predicted/actual term is only used once

---

## 🗂️ Script Reference

This logic lives in:
📄 `evaluate.py → evaluate_prediction_by_category(...)`
