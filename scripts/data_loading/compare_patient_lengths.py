import pandas as pd
from pathlib import Path
import os
import json

from dotenv import load_dotenv
load_dotenv()

def main():
    df = pd.read_csv(Path("test_data/temp/Mikey_length_1_ids_expanded.csv"))
    df = df[df['visit_start_to_ad_index_span_days'] > 0]
    ids = df['PatientEpicId_SH']
    
    first_patient_id = ids.iloc[0]
    anchor_date = df[df['PatientEpicId_SH']==first_patient_id].head(1)['ad_index_date'].iloc[0]
    with open(Path(os.environ['SLICED_PATIENT_JSON_DIR']) / f"{first_patient_id}.json", 'r') as f:
        first_patient_json = json.load(f)
        for encounter in first_patient_json['encounters']:
            print(f"Start Visit: {encounter['details']['start_visit']}\nAnchor Date: {anchor_date}\n\n", flush=True)

if __name__=="__main__":
    main()