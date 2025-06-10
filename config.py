# config.py
class ProjectConfig:
    def __init__(self,
                 num_patients: int = 100,
                 num_visits: int = 5,
                 use_synthetic_data: bool = False,
                 vectorizer: str = "sentence_transformer",
                 distance_metric: str = "cosine"):
        self.NUM_PATIENTS = num_patients
        self.NUM_VISITS = num_visits
        self.USE_SYNTHETIC_DATA = use_synthetic_data
        self.VECTORIZER = vectorizer
        self.DISTANCE_METRIC = distance_metric

CONFIG = ProjectConfig()