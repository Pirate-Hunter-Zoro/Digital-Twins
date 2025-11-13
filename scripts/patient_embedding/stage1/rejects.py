"""Reject logger.
Append per-patient errors (id, reason) to stage1_rejects.jsonl."""

from __future__ import annotations
import json
from pathlib import Path

def log_reject(rejects_path: Path, pid: str, err: str) -> None:
    rejects_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rejects_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"patient_id": pid, "error": err}) + "\n")