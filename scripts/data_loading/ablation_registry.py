# The semantic-feature ablation slate. One entry per narrative concept that gets
# permuted across donors; read by the narrative builder, the ablated-embedding
# forge, and the ablation runner.
#
# APPEND ONLY. Two things key off a spec's POSITION in this list rather than off
# its id: the SLURM array index that selects which spec a GPU task embeds, and
# the per-spec bootstrap seed in display_ablated_roc_deltas, which is drawn from
# the enumeration index. Inserting or reordering an entry therefore moves the
# already-published confidence intervals of every spec after it. Donor pairings
# are safe either way -- those are seeded on the spec id (build_pairings).
ABLATIONS = [
    {"id": "permute_race",          "display": "Race / ethnicity",          "bundle": "sociodemographics", "key": "Race_Ethnicity", "strategy": "permute_field"},
    {"id": "permute_psych_history", "display": "Psychiatric history",        "bundle": "psych_history",                              "strategy": "permute_section"},
    {"id": "permute_med_burden",    "display": "Medication burden",          "bundle": "medication_burden",                          "strategy": "permute_section"},
    {"id": "permute_treatment_contraindications", "display": "Treatment contraindications", "bundle": "treatment_contraindications",  "strategy": "permute_section"},
    {"id": "permute_sdoh",          "display": "Social determinants (SDOH)", "bundle": "sociodemographics", "key": "SDOH",           "strategy": "permute_field"},
    # Added 2026-08-21, and appended for the reason above. This section was left
    # off the original slate on the mistaken grounds that it restates the
    # outcome label. It does not: the label counts antidepressant treatments in
    # the POST-anchor year, whereas this section reports prior adequate trials,
    # benzodiazepine days, hypnotics and augmentation recorded at or before the
    # anchor. Permuting it measures how much of the embedding's discrimination
    # rests on prior antidepressant exposure, which is expected to be the
    # largest delta on the slate -- a reason to have the number, not a reason to
    # omit it.
    {"id": "permute_treatment_exposure", "display": "Treatment exposure",    "bundle": "treatment_exposure",                         "strategy": "permute_section"},
]
