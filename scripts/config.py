# --- config.py ---
import sys
import os

# --- Dynamic sys.path adjustment ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of sys.path adjustment ---

class ProjectConfig:
    def __init__(self,
                 num_patients: int,
                 num_visits: int,
                 num_neighbors: int,
                 vectorizer_method: str,
                 distance_metric: str,
                 representation_method: str): # <-- New parameter!
        self.num_patients = num_patients
        self.num_visits = num_visits
        self.num_neighbors = num_neighbors
        self.vectorizer_method = vectorizer_method
        self.distance_metric = distance_metric
        self.representation_method = representation_method # <-- Stored here!

__global_config = None

def setup_config(
        vectorizer_method: str,
        distance_metric: str,
        num_visits: int,
        num_patients: int,
        num_neighbors: int,
        representation_method: str # <-- New parameter!
    ):
    global __global_config
    __global_config = ProjectConfig(
        vectorizer_method=vectorizer_method,
        distance_metric=distance_metric,
        num_visits=num_visits,
        num_patients=num_patients,
        num_neighbors=num_neighbors,
        representation_method=representation_method # <-- Passed here!
    )

def get_global_config():
    assert __global_config, "Must call 'setup_config()' before 'get_global_config()'"
    return __global_config