import sys
from pathlib import Path

# We need to tell python where the scripts directory is - the root project directory
sys.path.append(Path(__file__).parent.parent)

from scripts.data_loading.diagnoses_definitions import (
    get_diagnosis_arm,
    MDD, 
    SUD, 
    SUICIDE_ATTEMPT,
    SUICIDE_IDEATION,
    get_mdd_description
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
        
def test_mdd_severity_parsing():
    """
    Confirm severity and recurrence are properly parsed for given codes
    """
    assert get_mdd_description(code="F32.0")=="Single Episode, Mild"\
        and get_mdd_description(code="F33.2")=="Recurrent, Severe"\
            and get_mdd_description(code="F33.3")=="Recurrent, Psychotic"\
                and get_mdd_description(code="F33.4")=="Recurrent, Remission"\
                    and get_mdd_description(code="F32.9")=="Single Episode, Unspecified"\
                        and get_mdd_description(code="F32")=="Single Episode, Unspecified"