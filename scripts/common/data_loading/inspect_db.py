import sqlite3
from pathlib import Path
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
db_file = Path(os.environ['DB_PATH'])
csv_file = Path(os.environ['PERSON_CSV_PATH'])

def inspect_csv_header(file_path: Path):
    header_df = pd.read_csv(file_path, nrows=5)
    col_names = header_df.columns
    print('--- Patient CSV Header and First 5 Rows ---')
    print(col_names)
    print(header_df.head())

if __name__ == "__main__":
    
    # Inspect the DB file
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()
    # Get a layout of the database
    cursor.execute('SELECT name FROM sqlite_master WHERE type="table";')
    results = cursor.fetchall()
    for table in results:
        table_name = table[0]
        cursor.execute(f'PRAGMA table_info({table_name})')
        table_info = cursor.fetchall()
        print(f"{table_name}:")
        print(table_info)
        print("\n\n")
    connection.close()
    
    # Inspect the person csv file
    inspect_csv_header(csv_file)