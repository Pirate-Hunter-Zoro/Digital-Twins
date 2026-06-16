<!--
Supplementary material for the TRD prediction manuscript (writeup/manuscript.md).
Convert alongside the main text at submission: pandoc supplement.md -o supplement.docx.
-->

# Supplement S1. LLM clinical-similarity judge

## S1.1 Task

In the neighbor-weighted prediction arm, the LLM weighting strategy used a
large language model (`MedGemma-27B`) as a clinical-similarity judge. For each
anchor–neighbor pair the judge received the two patients' deterministic
narratives and returned a structured JSON score. Only the integer
`overall_similarity` field (0–100), rescaled to the unit interval, was used as
the neighbor weight; the per-dimension sub-scores and free-text lists were
retained for auditing only and did not enter the prediction. The judge was run
with cached results (each unordered pair scored once and stored), so repeated
runs are deterministic with respect to the cache.

## S1.2 System prompt (verbatim)

```
You are a clinical similarity scorer for major depressive disorder (MDD).
Compare two patient narratives that contain only structured EHR data from a fixed baseline window.

Judge similarity ONLY on factors that affect antidepressant response/tolerability.
Treat "Missing" as unknown (neutral). Treat "Absent" as truly absent. Do not infer beyond text.

Weight these dimensions when scoring (sum=100):
1. Baseline symptom phenotype (PHQ-9 subitems): 25 points
2. Psychiatric comorbidity & threat/anxiety/trauma & SUD/suicidality: 20 points
3. Medical/metabolic & pain/NSAIDs: 20 points
4. Treatment exposure & medication burden (polypharmacy, prior adequate trials, 3y complexity): 20 points
5. Social/functional access (no-show, marital/employment/SDOH): 10 points
6. Safety flags (contraindications): 5 points

Return JSON only.
Do not repeat items in lists.
Do NOT provide explanations, reasoning, or calculations. Output raw JSON only.
```

## S1.3 User prompt and output schema (verbatim)

The two narratives are injected at `{narrative_a}` and `{narrative_b}`.

```
INDEX PATIENT:
{narrative_a}

CANDIDATE PATIENT:
{narrative_b}

Limit "top_similarity_drivers" and "key_mismatches" to at most 5 items each. Return this JSON:
{
    "overall_similarity": 0-100,
    "phenotype": 0-100,
    "psych_comorbidity": 0-100,
    "metabolic_pain": 0-100,
    "treatment_burden": 0-100,
    "social_functional": 0-100,
    "safety": 0-100,
    "top_similarity_drivers": ["driver 1", "...up to 5"],
    "key_mismatches": ["mismatch 1", "...up to 5"]
}

Scoring rules:
- Same band (e.g., NSAID 0 vs 1) = small penalty; different band (0 vs >=2) = large penalty.
- Reward matches on polypharmacy, pain/NSAIDs, anxiety/fatigue most.
- If a dimension has 'Missing' data, strictly output the value 0 for that score.
```

## S1.4 Worked examples

The examples below are the extremes of the judged distribution across the 8B
encoder's neighborhoods: the five lowest- and five highest-scoring anchor–
neighbor pairs, extracted by `scripts/digital_twins/neighbors/llm_similarity_audit.py`
and provided in full at `test_data/llm_judge_low_{1..5}.txt` and
`test_data/llm_judge_high_{1..5}.txt`. Two of each are reproduced here in full;
the remaining responses are shown as JSON only.

### Lowest-similarity example (score 25) — dissimilar patients

> **Narrative A** — MDD (Recurrent, Severe); Female, 54, Married, smoker;
> Anxiety, SUD, PTSD present, suicidality flagged; thyroid + hyperlipidemia;
> 0 prior adequate AD trials; 2 active meds (citalopram, venlafaxine); 1 psych
> inpatient day; epilepsy and uncontrolled HTN present.
>
> **Narrative B** — MDD (Single Episode, Unspecified); Female, 76, Single,
> smoker; no psychiatric comorbidity, no suicidality; no medical comorbidity;
> 0 prior adequate AD trials; 0 active meds; no safety flags.

```json
{
    "overall_similarity": 25,
    "phenotype": 0, "psych_comorbidity": 25, "metabolic_pain": 0,
    "treatment_burden": 25, "social_functional": 0, "safety": 0,
    "top_similarity_drivers": [
        "Both patients are female smokers.",
        "Both patients have 0 prior adequate AD trials.",
        "Both patients have 0 benzodiazepine days (2y).",
        "Both patients have 0 psych inpatient days (2y).",
        "Both patients have 0 ED psych visits (2y)."
    ],
    "key_mismatches": [
        "Index patient has severe recurrent MDD vs candidate single episode.",
        "Index patient has anxiety, SUD, PTSD vs candidate has none.",
        "Index patient has epilepsy and uncontrolled HTN vs candidate has none."
    ]
}
```

### Highest-similarity example (score 98) — near-duplicate patients

> **Narrative A** — MDD (Single Episode, Unspecified); Female, 71, Divorced,
> never-smoker; Anxiety present, no SUD/suicidality; chronic pain +
> hyperlipidemia; 0 prior adequate AD trials; 1 active med (fluoxetine);
> uncontrolled HTN.
>
> **Narrative B** — Identical on every clinical field; differs only in age
> (67 vs 71) and recorded religion (Christian vs Baptist).

```json
{
    "overall_similarity": 98,
    "phenotype": 100, "psych_comorbidity": 100, "metabolic_pain": 100,
    "treatment_burden": 100, "social_functional": 100, "safety": 100,
    "top_similarity_drivers": [
        "Identical baseline symptom phenotype (PHQ-9 subitems)",
        "Identical psychiatric comorbidity (Anxiety present, SUD/Suicidality/Trauma absent)",
        "Identical medical comorbidity (Chronic Pain present, Diabetes/Thyroid absent)",
        "Identical treatment exposure (No prior adequate AD trials)",
        "Identical medication burden (1 active med, no NSAIDs)"
    ],
    "key_mismatches": [
        "Age difference (71 vs 67)",
        "Marital Status difference (Divorced vs Divorced)",
        "Religion difference (Baptist vs Christian)"
    ]
}
```

### Additional responses (JSON only)

Second-lowest pair (score 25):

```json
{
    "overall_similarity": 25,
    "phenotype": 0, "psych_comorbidity": 25, "metabolic_pain": 100,
    "treatment_burden": 0, "social_functional": 0, "safety": 100,
    "top_similarity_drivers": ["Absent NSAID burden", "Absent Chronic Pain", "Absent Epilepsy", "Absent Uncontrolled HTN", "Absent Diabetes"],
    "key_mismatches": ["Presence of Anxiety/SUD/PTSD/Adjustment Disorder in Index", "Active polypharmacy (2 meds) in Index", "Different MDD severity/episode type"]
}
```

Second-highest pair (score 98):

```json
{
    "overall_similarity": 98,
    "phenotype": 0, "psych_comorbidity": 100, "metabolic_pain": 100,
    "treatment_burden": 100, "social_functional": 100, "safety": 100,
    "top_similarity_drivers": ["No psychiatric comorbidity", "No substance use disorder", "No chronic pain", "No NSAID burden", "No prior adequate antidepressant trials"],
    "key_mismatches": ["Age difference (76 vs 60)", "Missing BMI", "Missing BP"]
}
```

## S1.5 Interpretation and caveats

The judge's **overall** similarity behaved sensibly at the extremes — near-
duplicate patients scored 98, clinically dissimilar patients scored 25 — and
the population of ~523,000 cached judgements spans roughly 35–98 (modal
45–65), so the score is graded rather than degenerate.

The **per-dimension sub-scores and free-text rationales are not reliable** and
should be read as illustrative only:

- The `overall_similarity` is a holistic judgment, **not an aggregation of the
  six sub-scores**: the second-lowest pair scores `metabolic_pain` 100 and
  `safety` 100 yet an `overall` of 25, and the second-highest pair scores
  `phenotype` 0 yet an `overall` of 98.
- The judge **confabulates rationale**: the highest pair cites "identical
  PHQ-9 subitems" although no PHQ-9 subitems appear in the narrative, and lists
  "Divorced vs Divorced" — two identical values — as a key mismatch. The
  second-highest pair lists "Missing BMI/BP" as mismatches despite the system
  prompt instructing that Missing be treated as neutral.

Because only `overall_similarity` enters the neighbor weighting, these
sub-score artifacts do not propagate into the predictions; they are reported
here for transparency. Consistent with the main-text result, LLM weighting
added only marginal discrimination over cosine weighting, which we attribute to
the weighting strategy being second-order to the retrieval scheme rather than to
any deficiency in the overall similarity score.
