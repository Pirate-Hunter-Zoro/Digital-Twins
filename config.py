class ProjectConfig:
    def __init__(self,
                 num_patients: int,
                 num_visits: int,
                 num_neighbors: int,
                 use_synthetic_data: bool,
                 vectorizer_method: str,
                 distance_metric: str):
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
    assert __global_config, "Must call 'setup_config()' before 'get_global_config()'"
    return __global_config