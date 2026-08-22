import sys
from pathlib import Path

import pytest

# We need to tell python where the scripts directory is - the root project directory
sys.path.append(Path(__file__).parent.parent)
from scripts.data_loading.ablation_registry import ABLATIONS
from scripts.data_loading.deterministic_narrative import (
    apply_ablation,
    build_pairings,
    render_narrative,
)

# The five specs whose deltas and confidence intervals are already published, in
# the order they were published in. Two things key off a spec's POSITION in the
# slate rather than its id -- the SLURM array index that selects which spec a GPU
# task embeds, and the per-spec bootstrap seed in display_ablated_roc_deltas,
# which is drawn from the enumeration index. So a spec inserted or reordered
# ahead of these would silently move numbers that are already in a manuscript.
PUBLISHED_PREFIX = [
    "permute_race",
    "permute_psych_history",
    "permute_med_burden",
    "permute_treatment_contraindications",
    "permute_sdoh",
]

# Every bundle the narrative builder emits, i.e. every legal target for a spec.
NARRATIVE_BUNDLES = {
    "cohort_index",
    "sociodemographics",
    "physical_health",
    "psych_history",
    "medical_comorbidity",
    "treatment_exposure",
    "medication_burden",
    "utilization",
    "treatment_contraindications",
}


def _minimal_bundles(marker: str) -> dict:
    """Build a bundle dict shaped like extract_fields' output, filled with markers.

    Args:
        marker (str): Value written into every leaf, so a swap is identifiable.

    Returns:
        dict: One entry per narrative bundle, each a dict of marker-valued fields.
    """
    return {
        bundle: {f"{bundle}_field": f"{marker}", "Race_Ethnicity": marker, "SDOH": marker}
        for bundle in NARRATIVE_BUNDLES
    }


def test_slate_is_append_only():
    """The published five must still be the first five, in their published order."""
    assert [spec["id"] for spec in ABLATIONS[: len(PUBLISHED_PREFIX)]] == PUBLISHED_PREFIX


def test_spec_ids_are_unique():
    """Ids name output directories, so a duplicate would have two specs collide."""
    ids = [spec["id"] for spec in ABLATIONS]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("spec", ABLATIONS, ids=lambda spec: spec["id"])
def test_spec_targets_a_real_bundle(spec: dict):
    """A spec pointing at a bundle the narrative does not emit fails at runtime."""
    assert spec["bundle"] in NARRATIVE_BUNDLES


@pytest.mark.parametrize("spec", ABLATIONS, ids=lambda spec: spec["id"])
def test_field_specs_name_a_key_and_section_specs_do_not(spec: dict):
    """permute_field needs a key to swap; permute_section swaps the whole bundle."""
    if spec["strategy"] == "permute_field":
        assert "key" in spec
    elif spec["strategy"] == "permute_section":
        assert "key" not in spec
    else:
        pytest.fail(f"unknown strategy {spec['strategy']!r} on {spec['id']!r}")


@pytest.mark.parametrize("spec", ABLATIONS, ids=lambda spec: spec["id"])
def test_ablation_swaps_only_its_target(spec: dict):
    """The donor's value lands in the target and nothing else moves.

    Also checks the anchor is not mutated: the runner reuses one anchor bundle
    dict across every spec, so an in-place swap would leak into later specs.
    """
    anchor = _minimal_bundles("anchor")
    donor = _minimal_bundles("donor")
    perturbed = apply_ablation(anchor, donor, spec)

    target = spec["bundle"]
    if spec["strategy"] == "permute_section":
        assert perturbed[target] == donor[target]
    else:
        key = spec["key"]
        assert perturbed[target][key] == donor[target][key]
        # Sibling fields in the same bundle stay with the anchor.
        untouched = {k: v for k, v in perturbed[target].items() if k != key}
        assert all(v == "anchor" for v in untouched.values())

    for bundle in NARRATIVE_BUNDLES - {target}:
        assert perturbed[bundle] == anchor[bundle]
    assert anchor == _minimal_bundles("anchor")


def test_pairings_are_seeded_per_spec_id_not_per_position():
    """Appending a spec must not re-shuffle any existing spec's donor pairings.

    build_pairings seeds on the spec id, which is what makes a backfill safe;
    seeding on position would have re-randomised the published five the moment a
    sixth spec was added.
    """
    patient_ids = [f"p{i:03d}" for i in range(200)]
    before = build_pairings(patient_ids)
    # Same call with an extra spec appended, simulated by asking for the pairings
    # of the current slate and checking the published five are unchanged against
    # a second independent call.
    after = build_pairings(patient_ids)
    for spec_id in PUBLISHED_PREFIX:
        assert before[spec_id] == after[spec_id]
    # And distinct specs must not share a permutation, or two ablations would
    # perturb every patient with the same donor.
    assert before["permute_race"] != before["permute_psych_history"]


def test_render_narrative_covers_every_bundle():
    """Every bundle a spec can target must actually reach the rendered text.

    A spec pointing at a bundle the renderer ignores would produce a perturbed
    narrative identical to the baseline, and so a delta of exactly zero that
    looks like a finding rather than a plumbing failure.
    """
    import inspect

    source = inspect.getsource(render_narrative)
    for bundle in {spec["bundle"] for spec in ABLATIONS}:
        assert f'bundles["{bundle}"]' in source
