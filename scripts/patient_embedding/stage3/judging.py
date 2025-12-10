"""Parallel LLM judging.
Threaded scoring of narrative pairs via vllm with resume safety and bad-response capture."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterator
import multiprocessing
import re
import json

from common.models.vllm_client import VllmClient

from patient_embedding.shared.prompts import PromptLoader, SCORE_PATTERN, EXPLANATION_PATTERN
from patient_embedding.shared.io import read_text

from dotenv import load_dotenv
load_dotenv()

def init_worker(scr: bool=False):
    global client
    global prompt_loader
    global out_dir
    global scrub
    client = VllmClient()
    prompt_loader = PromptLoader()
    out_dir  = Path(os.environ['ANALYSIS_DIR'])
    scrub = scr
    
def judge_similarity(pid: str, na: str, nb: str) -> Tuple[float, str]:
    # Query generative model to judge the semantic similarity between two text dumps of patient data
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
            raise ValueError(f"LLM response did not contain a 'SCORE:' line for pair id {pid}. Content was: {resp}")
    except Exception as e:
        raise ValueError(f"Error in querying judge for pair id {pid}: {e}")
    
    return (j_score, j_rationale)

def score_pair(patient_pair_tuple: Tuple[str,str,float]) -> Dict[str, Any]:
    # This will compute the judge similarity for the given pair if it does not yet exist
    patient_a_id = patient_pair_tuple[0]
    patient_b_id = patient_pair_tuple[1]
    cosine = patient_pair_tuple[2]
    pid = f"{patient_a_id}:{patient_b_id}"
    
    # The LLM judgement score is stored based on which segments of the narrative we are scoring
    judge_score_path = Path(os.environ['ANALYSIS_DIR']) / "judge_scores" / f"{pid.replace(':','_')}.json"
    if judge_score_path.exists() and not scrub:
        # The judge already gave us a score
        with open(judge_score_path, 'r') as f:
            judge_output = json.load(f)
            j_score = judge_output['judge_score']
            j_rationale = judge_output['judge_rationale']
    else:
        # We need to ask the judge for a score  
        na_path = Path(os.environ['NARRATIVES_DIR']) / f'{patient_a_id}.md'
        nb_path = Path(os.environ['NARRATIVES_DIR']) / f'{patient_b_id}.md'
        
        
        na = read_text(na_path)
        nb = read_text(nb_path)
        
        j_score, j_rationale = judge_similarity(pid, na, nb)
        if j_score != float("nan"):
            # We can save this response
            with open(judge_score_path, 'w') as f:
                json.dump(
                    {
                        "judge_score": j_score,
                        "judge_rationale": j_rationale, 
                    },
                    f,
                    indent=4
                )
       
    return { 
        "patient_a_id": patient_a_id, 
        "patient_b_id": patient_b_id,
        "cosine": cosine,
        "judge_score": j_score, 
        "judge_rationale": j_rationale,
    }

def score_pairs(pairs: List[Tuple[str,str,float]]) -> Iterator[Dict[str, Any]]:
    with multiprocessing.Pool(processes=int(os.environ['NUM_WORKERS_LLM_TASK']), initializer=init_worker, initargs=[True]) as thread_pool:
        yield from thread_pool.imap_unordered(score_pair, pairs)