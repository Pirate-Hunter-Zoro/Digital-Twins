import json
import pandas as pd
from matplotlib import pyplot as plt
import os

def visualize_results():
    # Create directory for saving plots if it doesn't exist
    if not os.path.exists('jaccard_scores'):
        os.makedirs('jaccard_scores', exist_ok=True)

    with open('patient_output_results.json', 'r') as file:
        data = json.load(file)

        for patient_id, visits in data.items():
            df = pd.DataFrame([
                {
                    'visit_idx': v['visit_idx'],
                    'diagnoses_score': v['scores']['diagnoses'],
                    'medications_score': v['scores']['medications'],
                    'treatments_score': v['scores']['treatments'],
                }
                for v in visits
            ])

            # Sort by visit index
            df = df.sort_values(by='visit_idx')

            # Plot the scores
            plt.figure(figsize=(10, 6))
            plt.plot(df['visit_idx'], df['diagnoses_score'], label='Diagnoses', marker='o')
            plt.plot(df['visit_idx'], df['medications_score'], label='Medications', marker='o')
            plt.plot(df['visit_idx'], df['treatments_score'], label='Treatments', marker='o')

            plt.xlabel('Visit Index')
            plt.ylabel('Jaccard Score')
            plt.title(f'Prediction Scores Over Time for Patient {patient_id}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'jaccard_scores/patient_{patient_id}_scores.png')
            plt.close()

if __name__ == "__main__":
    visualize_results()
    print("Visualization completed and saved in 'jaccard_scores' directory.")