"""Stage-1 coordinator.
"""

from __future__ import annotations
from multiprocessing import Process

from .generator import generate_llm_narratives, generate_deterministic_narratives

def run() -> None:
    llm_narrative_process = Process(target=generate_llm_narratives)
    deterministic_narrative_process = Process(target=generate_deterministic_narratives)
    
    llm_narrative_process.start()
    deterministic_narrative_process.start()
    
    # Wait until both are done
    llm_narrative_process.join()
    deterministic_narrative_process.join()

    print("[Stage1] complete.", flush=True)