import pandas as pd
from pathlib import Path
import os
import sys
import numpy as np
import asyncio

from scripts.pipeline.predictions.trd_predictor import TRDPredictor
from scripts.pipeline.neighbors.neighbor_scheme import RELEVANT_NEIGHBOR_SCHEMES
from scripts.pipeline.predictions.create_train_test_split import create_train_test_split

from dotenv import load_dotenv
load_dotenv()

RESULTS_DIR = Path(os.environ['RESULTS_DIR'])
LOG_EVERY = 1000
os.makedirs(RESULTS_DIR, exist_ok=True)
test_ids = create_train_test_split()[1]

async def main():
    np.random.seed(int(os.environ['SEED']))
    
    # Establish deterministic order over the different Slurm workers
    sorted_test_ids = sorted(list(test_ids))
    
    # Now break up the patient sample to different workers
    slurm_task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    slurm_task_count = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
    chunk_size = len(test_ids) // slurm_task_count
    num_with_extra = len(test_ids) % slurm_task_count
    
    predictor = TRDPredictor(exclude_ids=test_ids, shard_id=slurm_task_id)
    
    # Grab this worker's chunk of sorted_test_ids
    # start_index inclusive, end_index exclusive
    if slurm_task_id < num_with_extra:
        start_index = slurm_task_id*(chunk_size+1)
        end_index = start_index + chunk_size+1
    else:
        start_index = num_with_extra*(chunk_size+1) + (slurm_task_id - num_with_extra)*chunk_size
        end_index = start_index + chunk_size
    
    chunk_ids = sorted_test_ids[start_index:end_index]

    work_items = []
    for pid in chunk_ids:
        index_narrative = predictor.retriever.get_narrative(pid)
        index_vector = predictor.retriever.get_vector(pid)
        chronological_length = predictor.retriever.get_chronological_length(pid)
        for scheme in RELEVANT_NEIGHBOR_SCHEMES:
            neighbors = predictor.retriever.search(index_vector, scheme)
            for idx, (neighbor_id, cosine_score) in enumerate(neighbors):
                if neighbor_id == pid:
                    # HARD fail - we can't be our own neighbor
                    raise ValueError(f"Patient {pid} was one of their own neighbors...")
                neighbor_narrative = predictor.retriever.get_narrative(neighbor_id)
                record = {
                    "neighbor_scheme": scheme.name,
                    "chronological_length": chronological_length,
                    "anchor_patient_id": pid,
                    "neighbor_patient_id": neighbor_id,
                    "cosine_sim": cosine_score,
                    "neighbor_trd_label": predictor.get_trd_status(neighbor_id),
                    "rank_cosine": idx+1,
                    "llm_sim": float('nan')
                }
                work_items.append((record, index_narrative, neighbor_narrative, pid, neighbor_id))

    # Now that we have all of the work items, we want to throw them at the server LLM_MAX_CONCURRENCY at a time
    if predictor.judge_sims:
        predictor.scorer.start_writer()
        sem = asyncio.Semaphore(int(os.environ['LLM_MAX_CONCURRENCY']))
        async def judge_one(item) -> dict:
            record, index_narrative, neighbor_narrative, pid, neighbor_id = item
            async with sem: # If the maximum amount are already bombarding the server, we wait
                try:
                    judgement = await predictor.scorer.judge_async(index_narrative, neighbor_narrative, pid, neighbor_id)
                    record['llm_sim'] = judgement['overall_similarity']
                except Exception as e:
                    # One unrecoverable pair must not tear down the whole run; leave
                    # llm_sim as NaN and let the resume re-judge it later.
                    print(f"Skipping {pid} vs {neighbor_id} after unrecoverable error: {repr(e)}", file=sys.stderr, flush=True)
                return record

        total = len(work_items)
        done = 0
        coroutines = [judge_one(item) for item in work_items]
        results = []
        try:
            for res in asyncio.as_completed(coroutines):
                results.append(await res)
                done += 1
                if (done % LOG_EVERY) == 0:
                    print(f"Finished {done} judgements out of {total} total...", flush=True)
        finally:
            predictor.scorer.stop_writer()
            await predictor.scorer.client.async_client.aclose()
    else:
        results = [item[0] for item in work_items]
    pd.DataFrame(results).to_csv(RESULTS_DIR / f"neighbor_results_{slurm_task_id}.csv")


if __name__=="__main__":
    asyncio.run(main())