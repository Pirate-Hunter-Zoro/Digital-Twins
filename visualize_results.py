import json
import pandas as pd

def visualize_results():
    with open('patient_output_results.json', 'r') as file:
        data = json.load(file)

        for patient, visits in data.items():
            df = pd.DataFrame([
                {
                    'visit_idx': v['visit_idx'],
                    'diagnoses_score': v['scores']['diagnoses'],
                    'medications_score': v['scores']['medications'],
                    'treaments_score': v['scores']['treatments'],
                }
                for v in visits
            ])
