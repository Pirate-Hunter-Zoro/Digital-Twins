import sqlite3
import json
import os
import sys
import time
import queue
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import httpx

from dotenv import load_dotenv
load_dotenv()

from scripts.models.vllm_client import VllmClient
from scripts.shared.prompts import PromptLoader
from scripts.pipeline.neighbors.retriever import Retriever

guided_json = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "overall_similarity", 
        "phenotype", 
        "psych_comorbidity", 
        "metabolic_pain", 
        "treatment_burden",
        "social_functional",
        "safety",
        "top_similarity_drivers",
        "key_mismatches"    
    ],
    "properties": {
        "overall_similarity": {"type": "integer"},
        "phenotype": {"type": "integer"},
        "psych_comorbidity": {"type": "integer"},
        "metabolic_pain": {"type": "integer"},
        "treatment_burden": {"type": "integer"},
        "social_functional": {"type": "integer"},
        "safety": {"type": "integer"},
        "top_similarity_drivers": {"type": "array", "items": {"type": "string"}, "maxItems":5},
        "key_mismatches":  {"type": "array", "items": {"type": "string"}, "maxItems":5},
    }
}

# Writer batching knobs: flush after this many queued rows or this many seconds idle.
WRITE_BATCH_SIZE = 200
WRITE_FLUSH_SECONDS = 5.0


class Scorer:

    def __init__(self, require_client:bool = True, save_time_hist:bool = True, shard_id:int = None):
        """Initialize vllm client and internal database

        Args:
            require_client (bool, optional): Boolean for whether or not to load the VLLM server. Defaults to True.
            save_time_hist (bool, optional): Boolean for whether or not to create the time histogram. Defaults to True.
            shard_id (int, optional): Per-shard judgements DB file. Defaults to None which indicates single canonical file
        """
        if require_client:
            self.client = VllmClient()
        self.prompt_loader = PromptLoader()
        self.retriever = Retriever(save_time_hist=save_time_hist)
        self._init_db(shard_id)
        # Single-writer, batched-commit queue. Judgement writes are pushed here
        # and drained by one dedicated thread, so no SQLite commit ever runs on
        # the asyncio event loop (which previously froze the judging loop) and
        # only one connection ever writes the shard DB.
        self._write_queue: queue.Queue = queue.Queue()
        self._writer_thread: Optional[threading.Thread] = None
        
    def _init_db(self, shard_id: int=None):
        """Initialize database for judgement scores - dependent on shard ID

        Args:
            shard_id (int, optional): Specifies which .db file to create. Defaults to None which implies one canonical file.
        """
        self.canonical_cursor = None
        if shard_id is None:
            vectors_path = Path(os.environ['JUDGEMENTS_DIR']) / "judgements.db"
        else:
            canonical_path = Path(os.environ['JUDGEMENTS_DIR']) / "judgements.db"
            if canonical_path.exists():
                self.canonical_cursor = sqlite3.connect(f"file:{canonical_path}?mode=ro", uri=True, check_same_thread=False).cursor()
            vectors_path = Path(os.environ['JUDGEMENTS_DIR']) / f"judgements_{shard_id}.db"
        os.makedirs(vectors_path.parent, exist_ok=True)
        self._db_path = vectors_path
        # Reader connection (event-loop thread). check_same_thread=False is safe
        # because the writer thread owns its own separate connection; a generous
        # busy_timeout absorbs the brief reader/writer lock overlap instead of
        # raising "database is locked".
        self.connection = sqlite3.connect(vectors_path, timeout=30.0, check_same_thread=False)
        self.connection.execute("PRAGMA busy_timeout=30000")
        self.cursor = self.connection.cursor()
        # Create table for judgements
        self.cursor.execute('''
CREATE TABLE IF NOT EXISTS llm_judgements (
    id_a TEXT,
    id_b TEXT,
    overall_score INTEGER,
    full_response TEXT,
    PRIMARY KEY (id_a, id_b)
);
''')
        self.connection.commit()
        
    def _get_cached_judgement(self, id_a: str, id_b: str) -> Optional[Dict]:
        """Return cached judgement if it exists

        Args:
            id_a (str): ID of first patient
            id_b (str): ID of second patient

        Returns:
            Optional[Dict]: Resulting judgement if it exists (else None)
        """
        pair = (id_a, id_b) if id_a <= id_b else (id_b, id_a)
        self.cursor.execute('''
SELECT full_response FROM llm_judgements WHERE id_a=? AND id_b=?
''', pair)
        row = self.cursor.fetchone()
        if row != None:
            return json.loads(row[0])
        else:
            # Check the canonical cursor (merged .db) as well
            if self.canonical_cursor is not None:
                self.canonical_cursor.execute('''
SELECT full_response FROM llm_judgements WHERE id_a=? AND id_b=?
''', pair)
                row = self.canonical_cursor.fetchone()
                if row != None:
                    return json.loads(row[0])
                else:
                    return None
            else:
                return None
    
    def _cache_judge(self, id_a: str, id_b: str, response_json: Dict):
        """
        Extracts LLM judgement score from json and and saves everything to the database

        :param id_a: String id of first patient
        :type id_a: str
        :param id_b: String id of second patient
        :type id_b: str
        :param response_json: Response from the LLM
        :type response_json: Dict
        """
        pair = (id_a, id_b) if id_a <= id_b else (id_b, id_a)
        score = response_json['overall_similarity']
        # Non-blocking hand-off to the writer thread; never touches SQLite on the
        # event loop.
        self._write_queue.put((pair[0], pair[1], score, json.dumps(response_json, indent=4)))

    def _writer_loop(self):
        """Dedicated single-writer thread: drains the queue and commits in
        batches against its own connection. Being the only writer to the shard
        DB removes the per-row event-loop commits and the lock contention they
        caused."""
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.execute("PRAGMA busy_timeout=30000")
        pending = []

        def _flush():
            if not pending:
                return
            for attempt in range(10):
                try:
                    conn.executemany(
                        "INSERT OR REPLACE INTO llm_judgements (id_a, id_b, overall_score, full_response) VALUES (?, ?, ?, ?)",
                        pending,
                    )
                    conn.commit()
                    pending.clear()
                    return
                except sqlite3.OperationalError as exc:
                    print(f"writer: commit retry {attempt + 1}/10 after {repr(exc)}", file=sys.stderr, flush=True)
                    time.sleep(1.0)
            # Drop rather than wedge the writer forever; dropped pairs are simply
            # re-judged on a later resume (cache miss).
            print(f"writer: DROPPING batch of {len(pending)} judgements after repeated lock failures", file=sys.stderr, flush=True)
            pending.clear()

        while True:
            try:
                item = self._write_queue.get(timeout=WRITE_FLUSH_SECONDS)
            except queue.Empty:
                _flush()
                continue
            if item is None:
                _flush()
                break
            pending.append(item)
            if len(pending) >= WRITE_BATCH_SIZE:
                _flush()
        conn.close()

    def start_writer(self):
        """Start the background judgement writer (idempotent)."""
        if self._writer_thread is None:
            self._writer_thread = threading.Thread(target=self._writer_loop, name="judgement-writer", daemon=True)
            self._writer_thread.start()

    def stop_writer(self):
        """Signal the writer to flush its final batch and wait for it to exit."""
        if self._writer_thread is not None:
            self._write_queue.put(None)
            self._writer_thread.join()
            self._writer_thread = None

    async def judge_async(self, index_narrative: str, candidate_narrative: str, index_id: str, candidate_id: str) -> Dict[str, Any]:
        cached = self._get_cached_judgement(id_a=index_id, id_b=candidate_id)
        if cached != None:
            return cached
        # Otherwise we need to ask the LLM for a judgement
        system_prompt = self.prompt_loader.get_judge_system()
        user_prompt = self.prompt_loader.render_judge_user(narrative_a=index_narrative, narrative_b=candidate_narrative)
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ]
        
        # Try to get a judgement
        attempts = 3
        for _ in range(attempts):
            try:
                response = await self.client.chat_async(messages=messages, guided_json=guided_json)
                response_json = json.loads(response)
                self._cache_judge(id_a=index_id, id_b=candidate_id, response_json=response_json)
                return response_json
            except (httpx.HTTPError, json.JSONDecodeError) as e:
                print(f"{index_id} vs. {candidate_id}: {repr(e)}", file=sys.stderr, flush=True)
            
        # If we made it here, we failed
        raise RuntimeError(f"Failed to judge patients {index_id} and {candidate_id} after {attempts} attempts...")