class ProjectConfig:
    def __init__(self,
                 num_patients: int = 50,
                 num_visits: int = 10,
                 num_neighbors: int = 5,
                 use_synthetic_data: bool = False,
                 vectorizer_method: str = "sentence_transformer",
                 distance_metric: str = "cosine"):
        self.num_patients = num_patients
        self.num_visits = num_visits
        self.num_neighbors = num_neighbors
        self.use_synthetic_data = use_synthetic_data
        self.vectorizer_method = vectorizer_method
        self.distance_metric = distance_metric

__global_config = None

def setup_config(
        vectorizer_method: str,
        distance_metric: str,
        use_synthetic_data: bool,
        num_visits: int,
        num_patients: int,
        num_neighbors: int,
    ):
    global __global_config
    __global_config = ProjectConfig(
        vectorizer_method=vectorizer_method,
        distance_metric=distance_metric,
        use_synthetic_data=use_synthetic_data,
        num_visits=num_visits,
        num_patients=num_patients,
        num_neighbors=num_neighbors
    )

def get_global_config():
    return __global_config