import sys
import os

# --- Dynamic sys.path adjustment for module imports ---
# Get the absolute path to the directory containing the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Calculate the project root.
# This assumes your 'config.py' (or other root-level modules like 'main.py')
# is located two directories up from where this script resides.
# Example: If this script is in 'project_root/scripts/read_data/your_script.py'
# then '..' takes you to 'project_root/scripts/'
# and '..', '..' takes you to 'project_root/'
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))

# Add the project root to sys.path if it's not already there.
# Inserting at index 0 makes it the first place Python looks for modules,
# so 'import config' will find it quickly.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- End of sys.path adjustment ---

class ProjectConfig:
    def __init__(self,
                 num_patients: int,
                 num_visits: int,
                 num_neighbors: int,
                 vectorizer_method: str,
                 distance_metric: str):
        self.num_patients = num_patients
        self.num_visits = num_visits
        self.num_neighbors = num_neighbors
        self.vectorizer_method = vectorizer_method
        self.distance_metric = distance_metric

__global_config = None

def setup_config(
        vectorizer_method: str,
        distance_metric: str,
        num_visits: int,
        num_patients: int,
        num_neighbors: int,
    ):
    global __global_config
    __global_config = ProjectConfig(
        vectorizer_method=vectorizer_method,
        distance_metric=distance_metric,
        num_visits=num_visits,
        num_patients=num_patients,
        num_neighbors=num_neighbors
    )

def get_global_config():
    assert __global_config, "Must call 'setup_config()' before 'get_global_config()'"
    return __global_config