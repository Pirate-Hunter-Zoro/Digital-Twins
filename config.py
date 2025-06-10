# config.py
class ProjectConfig:
    def __init__(self,
                 num_patients: int = 100,
                 num_visits: int = 5,
                 use_synthetic_data: bool = False,
                 vectorizer_method: str = "sentence_transformer",
                 distance_metric: str = "cosine"):
        self.NUM_PATIENTS = num_patients
        self.NUM_VISITS = num_visits
        self.USE_SYNTHETIC_DATA = use_synthetic_data
        self.VECTORIZER = vectorizer_method
        self.DISTANCE_METRIC = distance_metric

GLOBAL_CONFIG = ProjectConfig()
def setup_config(
        vectorizer_method: str,
        distance_metric: str,
        use_synthetic_data: bool,
        num_visits: int,
        num_patients: int
    ):
    global GLOBAL_CONFIG
    GLOBAL_CONFIG = ProjectConfig(
        vectorizer_method=vectorizer_method,
        distance_metric=distance_metric,
        use_synthetic_data=use_synthetic_data,
        num_visits=num_visits,
        num_patients=num_patients
    )
