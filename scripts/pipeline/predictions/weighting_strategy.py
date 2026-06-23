from enum import Enum
import os

from dotenv import load_dotenv
load_dotenv()

class WeightingStrategy(Enum):
    UNIFORM = "UNIFORM"
    COSINE = "COSINE"
    LLM = "LLM"
    COMBINED = "COMBINED"
    
RELEVANT_WEIGHTING_STRATS = [strat for strat in WeightingStrategy]
if int(os.environ['COMPUTE_LLM_SIMILARITY']) == 0:
    RELEVANT_WEIGHTING_STRATS.remove(WeightingStrategy.LLM)
    RELEVANT_WEIGHTING_STRATS.remove(WeightingStrategy.COMBINED)