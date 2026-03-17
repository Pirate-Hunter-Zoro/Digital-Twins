from enum import Enum

class WeightingStrategy(Enum):
    UNIFORM = "UNIFORM"
    COSINE = "COSINE"
    LLM = "LLM"
    COMBINED = "COMBINED"