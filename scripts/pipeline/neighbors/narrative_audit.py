from pathlib import Path
import os
import sqlite3

from dotenv import load_dotenv
load_dotenv()

from scripts.shared.utils import load_trd_set

# Number of example narratives to dump per TRD class.
N_PER_CLASS = 3


def main():
    """Dump a small, reproducible sample of deterministic patient narratives for
    manual inspection / a supplement figure. Reads the narrative text straight
    from the embeddings DB (the narratives are deterministic and identical
    across encoders), labels each by TRD status, and writes one .txt per
    patient to ${RESULTS_DIR}/narrative_audit/. No LLM or vLLM server involved.
    """
    trd = load_trd_set()

    connection = sqlite3.connect(Path(os.environ['EMBEDDINGS_DIR']) / "embeddings.db")
    rows = connection.execute("SELECT patient_id, text FROM embeddings").fetchall()
    connection.close()

    results_dir = Path(os.environ['RESULTS_DIR']) / 'narrative_audit/'
    os.makedirs(results_dir, exist_ok=True)

    # Deterministic, non-cherry-picked sample: the first N_PER_CLASS
    # TRD-positive and N_PER_CLASS TRD-negative patients by sorted patient_id.
    rows.sort(key=lambda r: r[0])
    positive = [r for r in rows if r[0] in trd][:N_PER_CLASS]
    negative = [r for r in rows if r[0] not in trd][:N_PER_CLASS]

    for status, sample in (("TRD_POSITIVE", positive), ("TRD_NEGATIVE", negative)):
        for patient_id, narrative in sample:
            with open(results_dir / f"narrative_{status}_{patient_id}.txt", 'w') as f:
                f.write(f"TRD label: {status}\nPatient hash: {patient_id}\n\n{narrative}")


if __name__ == "__main__":
    main()
