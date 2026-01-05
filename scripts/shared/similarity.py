"""Vector similarity math.
Cosine distance/utility functions that operate on numpy arrays; safe for NaNs/zeros."""

import os
from pathlib import Path
import numpy as np
import sqlite3

from dotenv import load_dotenv
load_dotenv()

VECTORS_DIR = Path(os.environ['VECTORS_DIR'])
DB_PATH = VECTORS_DIR / 'vectors.db'

def _init_db() -> sqlite3.Connection:
    # Make connection to database
    connection = sqlite3.connect(DB_PATH)
    connection.execute('''
CREATE TABLE IF NOT EXISTS similarities (
    id_a TEXT,
    id_b TEXT,
    score REAL,
    PRIMARY KEY (id_a, id_b)
);
''')
    connection.execute('''
CREATE INDEX IF NOT EXISTS score ON similarities (score);
''')
    return connection

CONNECTION = _init_db()

def _load_vector(vector_id: str) -> np.array:
    cursor = CONNECTION.execute('''SELECT vector FROM vectors WHERE id=?''', (vector_id,))
    row = cursor.fetchone()
    if row is None:
        raise FileNotFoundError(f"Missing vector with id {vector_id}...")
    else:
        return np.frombuffer(row[0], dtype=np.float32)

def cosine(id_a: str, id_b: str) -> float:
    pair = (id_a, id_b) if id_a <= id_b else (id_b, id_a)
    cursor = CONNECTION.execute('''SELECT score FROM similarities WHERE id_a=? AND id_b=?''', pair)
    row = cursor.fetchone()
    if row is None:
        # Must compute and store similarity
        a = _load_vector(vector_id=id_a).astype(np.float64).ravel()
        b = _load_vector(vector_id=id_b).astype(np.float64).ravel()
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0:
            raise ValueError(f"Error - zero vector resulting from string of id {id_a}...")
        elif nb == 0:
            raise ValueError(f"Error - zero vector resulting from string of id {id_b}...")
        else:
            score = float(np.dot(a, b) / (na * nb))
            CONNECTION.execute('''
INSERT OR REPLACE INTO similarities (id_a, id_b, score) VALUES (?, ?, ?)
''',
                (pair[0], pair[1], score)
            )
            CONNECTION.commit()
            return score
    else:
        return row[0]