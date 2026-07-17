import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

def main():
    judgements_db = Path(os.environ['JUDGEMENTS_DIR']) / "judgements.db"
    shard_count = int(os.environ['NUM_VLLM_SERVERS'])
    connection = sqlite3.connect(str(judgements_db))
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_judgements (id_a TEXT, id_b TEXT, overall_score INTEGER, full_response TEXT, PRIMARY KEY (id_a, id_b))
        """
    )
    for shard_idx in range(shard_count):
        shard_db_path = Path(os.environ['JUDGEMENTS_DIR']) / f"judgements_{shard_idx}.db"
        if not shard_db_path.exists():
            print(f"Missing expected file: {str(shard_db_path)} for shard {shard_idx}... skipping...")
            continue
        cursor.execute(
        """
        ATTACH DATABASE ? AS shard    
        """,
        (str(shard_db_path),)
        )
        cursor.execute(
        """
        INSERT OR IGNORE INTO llm_judgements SELECT id_a, id_b, overall_score, full_response FROM shard.llm_judgements    
        """
        )
        print(f"Inserted {cursor.rowcount} rows into {str(judgements_db)} from {str(shard_db_path)}...", flush=True)
        connection.commit()
        cursor.execute(
        """
        DETACH DATABASE shard
        """
        )

    # After merging, examine total rows
    cursor.execute(
    """
    SELECT COUNT(*) FROM llm_judgements
    """
    )
    print(f"Observed {cursor.fetchone()[0]} total rows...", flush=True)
    connection.commit()
    connection.close()

if __name__=="__main__":
    main()