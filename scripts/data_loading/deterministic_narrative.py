from typing import Dict
import copy
import os
from pathlib import Path
import random

from dotenv import load_dotenv
load_dotenv()

from scripts.data_loading.features import (
    psych_comorbidity, 
    medical_comorbidity, 
    suicidality_flag,
    prior_adequate_trials,
    benzo_days,
    augmentation_flag,
    polypharmacy,
    nsaid_burden,
    psych_utilization,
    hypnotic_burden,
    somatic_treatment_flag,
    psychotherapy_count,
    safety_comorbidity,
    sud_specifics,
    get_sdoh,
    get_vitals_average,
)
from scripts.data_loading.ablation_registry import ABLATIONS

YEARS_BACK = int(os.environ['YEARS_BACK'])
SEED = int(os.environ['SEED'])

# For each information section, each patient has a Dict value for that information section
_DONOR_POOL: Dict[str, Dict[str, Dict]] = {} # Section -> Patient -> Values
# For each section, each patient is paired with another patient for swapping
_PAIRINGS: Dict[str, Dict[str, str]] = {} # Section -> Patient -> Patient

def set_pairings(pairings: Dict[str, Dict[str, str]]):
    global _PAIRINGS
    _PAIRINGS = pairings

def build_pairings(patient_ids: list[str]) -> Dict[str, Dict[str, str]]:
    """Create random pairings out of the given patient IDs for each attribute

    Args:
        patient_ids (list[str]): List of all patient IDs

    Returns:
        Dict[str, Dict[str, str]]: Outer key is spec id, inner key is anchor->donor
    """
    pairings = {}
    for spec in ABLATIONS:
        # Create pairings for this spec
        rng = random.Random(f"{spec['id']}|{SEED}")
        shuffled = list(patient_ids)
        rng.shuffle(shuffled)
        # The permutation created pairs (allowing one patient to be their own donor)
        pairings[spec["id"]] = dict(zip(patient_ids, shuffled))
    
    return pairings

def set_donor_pool(pool: Dict[str, Dict[str, Dict]]) -> None:
    """Register the cohort-wide donor pool in the parent process before forking workers.

    Args:
        pool (Dict[str, Dict[str, Dict]]): Mapping of patient_id -> bundles dict (from extract_fields).
    """
    global _DONOR_POOL
    _DONOR_POOL = pool


def get_bool_str(val: bool) -> str:
    if val:
        return "Present"
    return "Absent"

def extract_fields(sliced_json: Dict) -> Dict[str, Dict]:
    """Obtain values for each section within a json

    Args:
        sliced_json (Dict): Patient json

    Returns:
        Dict[str, Dict]: Outer dict keyed by section slug, inner dict of raw field values
    """
    demographics_of_interests = [
        "Sex",
        "PreferredLanguage",
        "AgeInYears",
        "SexualOrientation",
        "MaritalStatus",
        "Religion",
        "SmokingStatus",
        "Race_Ethnicity",
    ]

    condition = "MDD"
    if sliced_json["mdd_recurrence"] is not None:
        condition += f" ({sliced_json['mdd_recurrence']}, {sliced_json['mdd_severity']})"

    cohort_index_bundle = {
        "condition": condition,
        "anchor_date": sliced_json["anchor_date"],
        "baseline_window_days": -365 * YEARS_BACK,
        "mdd_to_anchor_days": sliced_json["mdd_to_anchor_days"],
        "num_encounters": sliced_json["num_encounters"],
        "mdd_within_window": sliced_json["mdd_to_anchor_days"] <= 365 * YEARS_BACK,
    }

    sociodemographics_bundle = {
        demographic: sliced_json["demographics"].get(demographic, "Missing")
        for demographic in demographics_of_interests
    }
    sociodemographics_bundle["SDOH"] = get_sdoh(patient_dict=sliced_json)

    vitals_avg_dict = get_vitals_average(sliced_json)
    physical_health_bundle = {
        "bmi": vitals_avg_dict["bmi"],
        "bp_sys": vitals_avg_dict["bp_sys"],
        "bp_dias": vitals_avg_dict["bp_dias"],
    }

    psych_history_bundle = {
        "comorbidities": psych_comorbidity(sliced_json),
        "suicide_flag": suicidality_flag(sliced_json),
        "substances": sud_specifics(sliced_json),
    }

    medical_comorbidity_bundle = {
        "comorbidities": medical_comorbidity(sliced_json),
    }

    treatment_exposure_bundle = {
        "adequate_trials": prior_adequate_trials(sliced_json),
        "benzo_days": benzo_days(sliced_json),
        "hypnotics": hypnotic_burden(sliced_json),
        "augmentation": augmentation_flag(sliced_json),
        "somatic": somatic_treatment_flag(sliced_json),
        "psychotherapy_count": psychotherapy_count(sliced_json),
    }

    medication_burden_bundle = {
        "active_meds": polypharmacy(sliced_json),
        "nsaid_ingredients": nsaid_burden(sliced_json),
    }

    in_patient_days, num_emergency = psych_utilization(sliced_json, YEARS_BACK)
    utilization_bundle = {
        "psych_inpatient_days": in_patient_days,
        "ed_psych_visits": num_emergency,
    }

    safety_bundle = {
        "comorbidities": safety_comorbidity(sliced_json),
    }

    return {
        "cohort_index": cohort_index_bundle,
        "sociodemographics": sociodemographics_bundle,
        "physical_health": physical_health_bundle,
        "psych_history": psych_history_bundle,
        "medical_comorbidity": medical_comorbidity_bundle,
        "treatment_exposure": treatment_exposure_bundle,
        "medication_burden": medication_burden_bundle,
        "utilization": utilization_bundle,
        "safety": safety_bundle,
    }
    
def render_narrative(bundles: Dict[str, Dict]) -> str:
    """Given the bundles of a patient, return the new corresponding narrative

    Args:
        bundles (Dict[str, Dict]): All patient information

    Returns:
        str: Resulting new narrative
    """
    demographics_of_interests = [
        "Sex",
        "PreferredLanguage",
        "AgeInYears",
        "SexualOrientation",
        "MaritalStatus",
        "Religion",
        "SmokingStatus",
        "Race_Ethnicity",
    ]

    cohort = bundles["cohort_index"]
    HEADER = f"### COHORT & INDEX\nCondition: {cohort['condition']} | \
Index date: {cohort['anchor_date']} | \
Baseline window: {cohort['baseline_window_days']}...0 days | \
MDD-to-anchor gap: {cohort['mdd_to_anchor_days']} days | \
Encounters in window: {cohort['num_encounters']} | \
MDD within window: {get_bool_str(cohort['mdd_within_window'])}\n"

    socio = bundles["sociodemographics"]
    demographics = [f"{demographic}: {socio[demographic]}" for demographic in demographics_of_interests]
    DEMOGRAPHICS = f"### SOCIODEMOGRAPHICS / ACCESS\n{' | '.join(demographics)}\nSDOH: {' | '.join(socio['SDOH'])}\n"

    vitals = bundles["physical_health"]
    bmi_val = vitals["bmi"]
    bmi_str = f"{bmi_val:.1f}" if isinstance(bmi_val, (int, float)) else "Missing"
    sys_val = vitals["bp_sys"]
    dia_val = vitals["bp_dias"]
    if isinstance(sys_val, (int, float)) and isinstance(dia_val, (int, float)):
        bp_str = f"{sys_val:.0f}/{dia_val:.0f}"
    else:
        bp_str = "Missing"
    VITALS = f"### PHYSICAL HEALTH\nBMI: {bmi_str} | BP (mean): {bp_str}\n"

    psych = bundles["psych_history"]
    psych_comorbidities = [f"{psych_arm}: {get_bool_str(psych['comorbidities'][psych_arm])}" for psych_arm in psych["comorbidities"].keys()]
    substances = ' | '.join([sud_name for sud_name in sorted(list(psych["substances"].keys())) if psych["substances"][sud_name]])
    PSYCH_HISTORY = f"### PSYCH HISTORY\n{' | '.join(psych_comorbidities)}\nSUICIDE FLAG ({YEARS_BACK}y): {get_bool_str(psych['suicide_flag'])}\nSUBSTANCE ABUSE: {substances if len(substances) > 0 else 'None'}\n"

    medcom = bundles["medical_comorbidity"]
    med_comorbidities = [f"{med_arm}: {get_bool_str(medcom['comorbidities'][med_arm])}" for med_arm in medcom["comorbidities"].keys()]
    MED_HISTORY = f"### MEDICAL COMORBIDITY\n{' | '.join(med_comorbidities)}\n"

    treat = bundles["treatment_exposure"]
    adequate_trials_count = treat["adequate_trials"]
    hypnotics_burden_set = treat["hypnotics"]
    TREAT_EXPOSURE = f"### TREATMENT EXPOSURE\nPrior adequate AD trials: {' | '.join([f'{arm}: {adequate_trials_count[arm]}' for arm in adequate_trials_count.keys()]) if len(adequate_trials_count) > 0 else 'Absent'}\n\
Benzodiazepine days (90d): {treat['benzo_days']}\n\
Hypnotics: {' | '.join([hypnotic for hypnotic in sorted(hypnotics_burden_set)]) if len(hypnotics_burden_set) > 0 else 'Absent'}\n\
Augmentation: {get_bool_str(treat['augmentation'])}\n\
Somatic treatments: {get_bool_str(treat['somatic'])} | Psychotherapy visits ({YEARS_BACK}y): {treat['psychotherapy_count'] if treat['psychotherapy_count'] > 0 else 'Absent'}\n"

    medburden = bundles["medication_burden"]
    distinct_ingredients = medburden["active_meds"]
    distinct_nsaid_ingredients = medburden["nsaid_ingredients"]
    MED_BURDEN = f"### MEDICATION BURDEN\nActive meds at baseline: {len(distinct_ingredients)} ({', '.join([ingredient for ingredient in sorted(distinct_ingredients)]) if len(distinct_ingredients) > 0 else 'Absent'})\n\
NSAID burden: {len(distinct_nsaid_ingredients)} ({', '.join([ingredient for ingredient in sorted(distinct_nsaid_ingredients)]) if len(distinct_nsaid_ingredients) > 0 else 'Absent'})\n"

    util = bundles["utilization"]
    UTILIZATION = f"### UTILIZATION\nPsych inpatient days: {util['psych_inpatient_days']} ({YEARS_BACK}y) | ED psych visits: {util['ed_psych_visits']} ({YEARS_BACK}y)\n"

    safety = bundles["safety"]
    SAFETY = "### SAFETY\n"+' | '.join([safety_arm + ": " + get_bool_str(safety['comorbidities'][safety_arm]) for safety_arm in safety['comorbidities'].keys()]) + '\n'

    return "\n".join([HEADER, DEMOGRAPHICS, VITALS, PSYCH_HISTORY, MED_HISTORY, TREAT_EXPOSURE, MED_BURDEN, UTILIZATION, SAFETY])


def apply_ablation(anchor_bundles: Dict[str, Dict], donor_bundles: Dict[str, Dict], spec: Dict) -> Dict[str, Dict]:
    """Return a perturbed copy of anchor_bundles with the donor's value(s) swapped in per spec.

    Args:
        anchor_bundles (Dict[str, Dict]): The anchor patient's full bundle dict.
        donor_bundles (Dict[str, Dict]): A donor patient's full bundle dict.
        spec (Dict): One entry from ABLATIONS. Must carry 'strategy' and 'bundle'; 'permute_field' also requires 'key'.

    Returns:
        Dict[str, Dict]: A deep-copied anchor_bundles with the requested swap applied. Anchor input is never mutated.
    """
    perturbed = copy.deepcopy(anchor_bundles)
    strategy = spec["strategy"]
    bundle_key = spec["bundle"]

    if strategy == "permute_section":
        perturbed[bundle_key] = copy.deepcopy(donor_bundles[bundle_key])
    elif strategy == "permute_field":
        field_key = spec["key"]
        perturbed[bundle_key][field_key] = copy.deepcopy(donor_bundles[bundle_key][field_key])
    else:
        raise ValueError(f"Unknown ablation strategy: {strategy!r}. Spec id: {spec.get('id')!r}.")

    return perturbed


def generate_deterministic_narrative(sliced_json: Dict) -> tuple[str, int]:
    """Parse the sliced patient json to generate a deterministic markdown file output

    Args:
        sliced_json (Dict): Anchor date going back a certain number of years
    Returns:
        tuple[str, int]: Patient id and chronologic length of the patient
    """
    patient_id = sliced_json["patient_id"]
    baseline_dir = Path(os.environ['NARRATIVES_DIR'])
    scrub = int(os.environ['SCRUB_NARRATIVES']) == 1
    patient_info = extract_fields(sliced_json)
    baseline_path = baseline_dir / f"{patient_id}.md"
    if scrub or (not baseline_path.exists()):
        # Must form the narrative
        os.makedirs(baseline_path.parent, exist_ok=True)
        baseline_text = render_narrative(patient_info)
        with open(baseline_path, 'w') as f:
            f.write(baseline_text)
    # Ablation loop
    for spec in ABLATIONS:
        spec_dir = baseline_dir / spec["id"] # e.g. .../narratives/swap_race/
        spec_path = spec_dir / f"{patient_id}.md"
        if scrub or (not spec_path.exists()):
            # Must recreate
            donor_id = _PAIRINGS[spec["id"]][patient_id] # For this spec, we have a bunch of patient pairings - find the patient that gives this patient their ablated value for this spec
            donor_info = _DONOR_POOL[donor_id]
            # Grab this other patient's information for this spec and overwrite this patient's info for that spec for the ablation
            perturbed = apply_ablation(patient_info, donor_info, spec)
            perturbed_text = render_narrative(perturbed)
            os.makedirs(spec_dir, exist_ok=True)
            with open(spec_path, 'w') as f:
                f.write(perturbed_text)
    return (patient_id, sliced_json['days_of_history'])

if __name__=="__main__":
    # Take a sample of produced narratives and put them in the local test_data directory
    all_narratives = list(Path(os.environ['NARRATIVES_DIR']).glob("*.md"))
    for narrative in random.sample(all_narratives, 10):
        with open(narrative, 'r') as f:
            content = f.read()
            new_file = Path("test_data") / narrative.name
            with open(new_file, 'w') as nf:
                nf.write(content)