# --- Global Config Management ---
class GlobalConfig:
    def __init__(
        self,
        representation_method,
        vectorizer_method,
        distance_metric,
        num_visits,
        num_patients,
        num_neighbors,
        model_name="medgemma",
    ):
        self.representation_method = representation_method
        self.vectorizer_method = vectorizer_method
        self.distance_metric = distance_metric
        self.num_visits = num_visits
        self.num_patients = num_patients
        self.num_neighbors = num_neighbors
        self.model_name = model_name 


# === Global variable for config ===
__global_config = None


def setup_config(
    representation_method,
    vectorizer_method,
    distance_metric,
    num_visits,
    num_patients,
    num_neighbors,
    model_name="medgemma", 
):
    global __global_config
    __global_config = GlobalConfig(
        representation_method=representation_method,
        vectorizer_method=vectorizer_method,
        distance_metric=distance_metric,
        num_visits=num_visits,
        num_patients=num_patients,
        num_neighbors=num_neighbors,
        model_name=model_name, 
    )


def get_global_config():
    assert __global_config, "Must call 'setup_config()' before 'get_global_config()'"
    return __global_config
