"""Stage-1 coordinator.
"""

from __future__ import annotations
from multiprocessing import Process
import time

from scripts.patient_embedding.stage1.generator import generate_llm_narratives, generate_deterministic_narratives

def run() -> None:
    print("Running deterministic and LLM narrative generation...", flush=True)
    
    llm_narrative_process = Process(target=generate_llm_narratives)
    deterministic_narrative_process = Process(target=generate_deterministic_narratives)
    
    llm_narrative_process.start()
    deterministic_narrative_process.start()
    
    # Monitor loop
    while llm_narrative_process.is_alive() and deterministic_narrative_process.is_alive():
        time.sleep(1)
        
    # Once here, at least one process has ended - make sure it did not end in error
    exit_codes = [llm_narrative_process.exitcode, deterministic_narrative_process.exitcode]
    terminated = False
    for code in exit_codes:
        if code != None and code != 0:
            # Kill both processes if a single one died
            llm_narrative_process.terminate()
            deterministic_narrative_process.terminate()
            terminated = True
    
    # One of the tasks finished successfully - just wait for the other one
    if llm_narrative_process.is_alive():
        llm_narrative_process.join()
    if deterministic_narrative_process.is_alive():
        deterministic_narrative_process.join()

    if terminated:
        print("[Stage1] FAILED...", flush=True)
    else:
        print("[Stage1] complete.", flush=True)