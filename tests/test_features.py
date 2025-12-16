import pytest
import sys
from pathlib import Path
from tests.conftest import MockPatientBuilder

# We need to tell python where the scripts directory is - the root project directory
sys.path.append(Path(__file__).parent.parent)
from scripts.common.data_loading.features import prior_adequate_trials
from scripts.common.data_loading.med_definitions import SSRI

def test_prior_adequate_trials_just_short(builder: MockPatientBuilder):
    """
    Docstring for test_prior_adequate_trials_just_short
    
    :param builder: Description
    :type builder: MockPatientBuilder
    """
    pass