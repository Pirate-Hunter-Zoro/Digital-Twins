import sys
import os

# --- Dynamic sys.path adjustment for module imports ---
# Get the absolute path to the directory containing the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Calculate the project root.
# This assumes your 'config.py' (or other root-level modules like 'main.py')
# is located two directories up from where this script resides.
# Example: If this script is in 'project_root/scripts/read_data/your_script.py'
# then '..' takes you to 'project_root/scripts/'
# and '..', '..' takes you to 'project_root/'
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

# Add the project root to sys.path if it's not already there.
# Inserting at index 0 makes it the first place Python looks for modules,
# so 'import config' will find it quickly.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- End of sys.path adjustment ---

import random
import json
from scripts.config import get_global_config

diagnosis_pool = [
    "E11",        # Type 2 Diabetes
    "I10",        # Hypertension
    "F33.1",      # Major depressive disorder, recurrent
    "M54.5",      # Low back pain
    "R73.9",      # Hyperglycemia, unspecified
    "J45.909",    # Asthma, unspecified
    "E78.5",      # Hyperlipidemia
    "K21.9",      # GERD
    "N39.0",      # UTI
    "F41.1",      # Generalized anxiety disorder
]

medication_pool = [
    "Metformin",
    "Insulin",
    "Lisinopril",
    "Atorvastatin",
    "Sertraline",
    "Albuterol",
    "Omeprazole",
    "Gabapentin",
    "Hydrochlorothiazide",
    "Bupropion"
]

treatment_pool = [
    "Diet and exercise plan",
    "Cognitive behavioral therapy",
    "Insulin titration follow-up",
    "Smoking cessation counseling",
    "Physical therapy referral",
    "Routine bloodwork",
    "Lifestyle modification discussion",
    "Medication adherence review",
    "Sleep hygiene education",
    "Referral to endocrinology"
]

def generate_visit() -> dict[str, list[str]]:
    """
    Generate a random visit with a diagnoses, medication, and treatment plan.
    """

    diagnoses = random.sample(diagnosis_pool, k=random.randint(1, min(5, len(diagnosis_pool))))  # Two diagnoses for complexity
    medications = random.sample(medication_pool, k=random.randint(1, min(5, len(diagnosis_pool))))
    treatments = random.sample(treatment_pool, k=random.randint(1, min(5, len(treatment_pool))))

    visit = {
        "diagnoses": diagnoses,
        "medications": medications,
        "treatments": treatments
    }

    return visit

total_patients = 0
def generate_patient() -> dict[str, object]:
    """
    Generate a random patient with a unique ID and a list of visits.
    """
    global total_patients
    visits = [generate_visit() for _ in range(get_global_config().num_visits)]
    total_patients += 1
    id = "P" + str(total_patients).zfill(7)
    patient = {
        "patient_id": id,
        "visits": visits
    }
    return patient

def write_and_generate_patients() -> None:
    """
    Generate a list of n random patients.
    """
    patients = [generate_patient() for _ in range(get_global_config().num_patients)]
    with open(f"synthetic_data/patient_data_{get_global_config().num_patients}_{get_global_config().num_visits}.json", "w") as f:
        json.dump(patients, f, indent=4)

def load_patient_data() -> list[dict]:
    """
    Load patient data from the JSON file.
    """
    if get_global_config().use_synthetic_data:
        try:
            with open(f"synthetic_data/patient_data_{get_global_config().num_patients}_{get_global_config().num_visits}.json", "r") as f:
                return json.load(f)
        except:
            write_and_generate_patients()
            return load_patient_data()
    else:
        try:
            with open(f"real_data/patient_data_{get_global_config().num_patients}.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            # TODO - parse the real data files and generate the JSON file for the number of patients
            pass