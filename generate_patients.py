import random
import json

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
def generate_patient(num_visits: int=5) -> dict[str, object]:
    """
    Generate a random patient with a unique ID and a list of visits.
    """
    global total_patients
    visits = [generate_visit() for _ in range(num_visits)]
    total_patients += 1
    id = "P" + str(total_patients).zfill(7)
    patient = {
        "patient_id": id,
        "visits": visits
    }
    return patient

def write_and_generate_patients(n_patients: int=100, num_visits: int=5) -> None:
    """
    Generate a list of n random patients.
    """
    patients = [generate_patient(num_visits=num_visits) for _ in range(n_patients)]
    with open(f"synthetic_data/patient_data_{n_patients}_{num_visits}.json", "w") as f:
        json.dump(patients, f, indent=4)

def load_patient_data(use_synthetic_data: bool=False, num_visits: int=5, num_patients: int=100) -> list[dict]:
    """
    Load patient data from the JSON file.
    """
    if use_synthetic_data:
        try:
            with open(f"synthetic_data/patient_data_{num_patients}_{num_visits}.json", "r") as f:
                return json.load(f)
        except:
            write_and_generate_patients(num_patients=num_patients, num_visits=num_visits)
            return load_patient_data()
    else:
        try:
            with open(f"real_data/patient_data_{num_patients}.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            # TODO - parse the real data files and generate the JSON file for the number of patients
            pass