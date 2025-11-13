from .embedding_pipeline import forge_test_vectors
from .data_parser import parse_test_narratives
import itertools
import numpy as np
from typing import Tuple
from pathlib import Path
from scripts.patient_embedding.shared.similarity import cosine
from scripts.patient_embedding.shared.prompts import PromptLoader
from scripts.patient_embedding.stage3.pairs import pair_id
from scripts.patient_embedding.shared.prompts import SCORE_PATTERN, EXPLANATION_PATTERN
from scripts.common.models.vllm_client import VllmClient
from scripts.patient_embedding.shared.io import write_text
import os
import re
import json
from dotenv import load_dotenv
load_dotenv()

RECORD_EVERY = 5

def forge_feature_matrix_embedding() -> Tuple[np.array, np.array]:
    vector_components = forge_test_vectors()
    patient_ids = vector_components.keys()
    patient_pairs = itertools.combinations(patient_ids, 2)
    X_features = []
    y_target = []
    
    for (id_a, id_b) in patient_pairs:
        cos_full_text = cosine(vector_components[id_a]['full_text'], vector_components[id_b]['full_text'])
        if np.isnan(cos_full_text):
            cos_full_text = 0
        y_target.append(cos_full_text)
        
        cos_sim_narrative = cosine(vector_components[id_a]['segment_narrative'], vector_components[id_b]['segment_narrative'])
        if np.isnan(cos_sim_narrative):
            cos_sim_narrative = 0
        cos_sim_meds = cosine(vector_components[id_a]['segment_medications'], vector_components[id_b]['segment_medications'])
        if np.isnan(cos_sim_meds):
            cos_sim_meds = 0
        cos_sim_diags = cosine(vector_components[id_a]['segment_diagnoses'], vector_components[id_b]['segment_diagnoses'])
        if np.isnan(cos_sim_diags):
            cos_sim_diags = 0
        X_features.append([cos_sim_narrative, cos_sim_meds, cos_sim_diags])
        
        if len(X_features) % RECORD_EVERY == 0:
            print(f"Computed Cosine Similarity for {len(X_features)} pairs...")
    
    return (np.array(X_features), np.array(y_target))

# Helper method for the judging of the feature matrix
def get_judge_score(pid: str, na: str, nb: str, client: VllmClient, prompt_loader: PromptLoader, segment: str) -> float:
    judge_score_path = Path(os.environ['ANALYSIS_DIR']) / segment / "judge_scores" / f"{pid.replace(':','_')}.json"
    os.makedirs(judge_score_path.parent, exist_ok=True)
    if judge_score_path.exists():
        # The judge already gave us a score
        with open(judge_score_path, 'r') as f:
            judge_output = json.load(f)
            j_score = judge_output['judge_score']
            j_rationale = judge_output['judge_rationale']
    else:
        try:
            # Get the system prompt.
            system_text = prompt_loader.get_judge_system()
            # Render the user prompt.
            user_text = prompt_loader.render_judge_user(narrative_a=na, narrative_b=nb)
            
            messages = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ]
            resp = client.chat(
                messages,
            )
            j_score = None
            j_rationale = None
            
            # Use regex to find the response portions we care about - grouping parentheses around the parts we want to extract
            score_match = re.search(SCORE_PATTERN, resp, re.IGNORECASE | re.DOTALL)
            explanation_match = re.search(EXPLANATION_PATTERN, resp, re.IGNORECASE | re.DOTALL)
            
            if score_match:
                j_score = float(score_match.group(1).strip())
            
            # This will capture everything after 'EXPLANATION:' until the end of the string
            if explanation_match:
                j_rationale = explanation_match.group(1).strip()
            
            # Explicitly check for a non-compliant response
            if j_score is None:
                raise ValueError(f"LLM response did not contain a 'SCORE:' line. Content was: {resp}")
            else:
                # SAVE the judge score - that took a lot of work to produce after all...
                with open(judge_score_path, 'w') as f:
                    json.dump(
                        {
                            "judge_score": j_score,
                            "judge_rationale": j_rationale, 
                        },
                        f,
                        indent=4
                    )
        except Exception as e:
            raise ValueError(f"Exception Occured: {e}...")
        
    return j_score

def forge_feature_matrix_judging() -> Tuple[np.array, np.array]:
    client = VllmClient()
    prompt_loader = PromptLoader()
    string_components = parse_test_narratives()
    patient_ids = string_components.keys()
    patient_pairs = itertools.combinations(patient_ids, 2)
    X_features = []
    y_target = []
    for (id_a, id_b) in patient_pairs:
        pid = pair_id(id_a, id_b)
        judge_full_text_score = get_judge_score(pid, string_components[id_a]['full_text'], string_components[id_b]['full_text'], client, prompt_loader, 'full')
        if np.isnan(judge_full_text_score):
            judge_full_text_score = 0
        y_target.append(judge_full_text_score)
        
        judge_sim_narrative_score = get_judge_score(pid, string_components[id_a]['segment_narrative'], string_components[id_b]['segment_narrative'], client, prompt_loader, 'summary')
        if np.isnan(judge_sim_narrative_score):
            judge_sim_narrative_score = 0
        judge_sim_meds_score = get_judge_score(pid, string_components[id_a]['segment_medications'], string_components[id_b]['segment_medications'], client, prompt_loader, 'medications')
        if np.isnan(judge_sim_meds_score):
            judge_sim_meds_score = 0
        judge_sim_diags_score = get_judge_score(pid, string_components[id_a]['segment_diagnoses'], string_components[id_b]['segment_diagnoses'], client, prompt_loader, 'diagnoses')
        if np.isnan(judge_sim_diags_score):
            judge_sim_diags_score = 0
        X_features.append([judge_sim_narrative_score, judge_sim_meds_score, judge_sim_diags_score])
        
        if len(X_features) % RECORD_EVERY == 0:
            print(f"Queried Judge Similarity for {len(X_features)} pairs...")
    
    return (np.array(X_features), np.array(y_target))