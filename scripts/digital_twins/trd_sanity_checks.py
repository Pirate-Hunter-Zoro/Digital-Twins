import sqlite3
import os
from pathlib import Path

from scripts.digital_twins.predictions.trd_predictor import TRDPredictor

from dotenv import load_dotenv
load_dotenv()

def main():
    # TODO - load all of vectors
    connection = sqlite3.connect(Path(os.environ['VECTORS_DIR']) / 'vectors.db')
    
    predictor = TRDPredictor()
    
    pass