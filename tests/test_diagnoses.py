import sys
from pathlib import Path

# We need to tell python where the scripts directory is - the root project directory
sys.path.append(Path(__file__).parent.parent)

from scripts.common.data_loading.diagnoses_definitions import (
    get_diagnosis_arm,
    MDD, 
    SUD, 
    SUICIDE_ATTEMPT,
    SUICIDE_IDEATION
)

def test_mdd_regex():
    """
    Confirm various regexes for MDD work as expected.
    """
    assert get_diagnosis_arm(diagnosis_code="F32.9") == MDD\
        and get_diagnosis_arm(diagnosis_code="F33.1") == MDD\
            and get_diagnosis_arm(diagnosis_code="F32") == None\
                and get_diagnosis_arm(diagnosis_code="F99.9") == None

def test_sud_regex():
    """
    Confirm various regexes for SUD work as expected
    """
    assert get_diagnosis_arm(diagnosis_code="F10.10") == SUD\
        and get_diagnosis_arm(diagnosis_code="F19") == SUD

def test_suicide_regex():
    """
    Confirm various codes for suicide attempt/ideation work as expected
    """
    assert get_diagnosis_arm(diagnosis_code="R45.851") == SUICIDE_IDEATION\
        and get_diagnosis_arm(diagnosis_code="X71") == SUICIDE_ATTEMPT