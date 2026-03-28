import sys
from pathlib import Path

# We need to tell python where the scripts directory is - the root project directory
sys.path.append(Path(__file__).parent.parent)

from scripts.data_loading.diagnoses_definitions import (
    get_diagnosis_arm,
    MDD, 
    SUD, 
    DYSTHYMIA,
    SUICIDE_ATTEMPT,
    SUICIDE_IDEATION,
    get_mdd_components,
    get_sud_substance
)

def test_mdd_regex():
    """
    Confirm various regexes for MDD work as expected.
    """
    assert get_diagnosis_arm(diagnosis_code="F32.9") == MDD\
        and get_diagnosis_arm(diagnosis_code="F33.1") == MDD\
            and get_diagnosis_arm(diagnosis_code="F32") == None\
                and get_diagnosis_arm(diagnosis_code="F99.9") == None\
                    and get_diagnosis_arm(diagnosis_code="296.20") == MDD\
                        and get_diagnosis_arm(diagnosis_code="296.32") == MDD\
                            and get_diagnosis_arm(diagnosis_code="296.2") == MDD

def test_dysthymia_regex():
    """
    Confirm various regexes for Dysthymia work as expected.
    """
    assert get_diagnosis_arm(diagnosis_code="300.4") == DYSTHYMIA\
        and get_mdd_components(code="300.4") == ("Dysthymia", "Unspecified")

def test_sud_regex():
    """
    Confirm various regexes for SUD work as expected
    """
    assert get_diagnosis_arm(diagnosis_code="F10.10") == SUD\
        and get_diagnosis_arm(diagnosis_code="F19") == SUD\
            and get_diagnosis_arm(diagnosis_code="305.00") == SUD\
                and get_diagnosis_arm(diagnosis_code="304.0") == SUD

def test_sud_specifics():
    """
    Ensure correct substances are derived from various SUD codes
    """
    assert get_sud_substance(code="F10.10") == "Alcohol"\
        and get_sud_substance(code="305.00") == "Alcohol"\
            and get_sud_substance(code="305.2") == "Cannabis"\
                and get_sud_substance(code="291.81") == "Alcohol"

def test_suicide_regex():
    """
    Confirm various codes for suicide attempt/ideation work as expected
    """
    assert get_diagnosis_arm(diagnosis_code="R45.851") == SUICIDE_IDEATION\
        and get_diagnosis_arm(diagnosis_code="X71") == SUICIDE_ATTEMPT\
            and get_diagnosis_arm(diagnosis_code="E950") == SUICIDE_ATTEMPT\
                and get_diagnosis_arm(diagnosis_code="V62.84") == SUICIDE_IDEATION
        
def test_mdd_severity_parsing():
    """
    Confirm severity and recurrence are properly parsed for given codes
    """
    assert get_mdd_components(code="F32.0")==("Single Episode", "Mild")\
        and get_mdd_components(code="F33.2")==("Recurrent", "Severe")\
            and get_mdd_components(code="F33.3")==("Recurrent", "Psychotic")\
                and get_mdd_components(code="F33.4")==("Recurrent", "Remission")\
                    and get_mdd_components(code="F32.9")==("Single Episode", "Unspecified")\
                        and get_mdd_components(code="F32")==("Single Episode", "Unspecified")\
                            and get_mdd_components(code="296.20")==("Single Episode", "Unspecified")\
                                and get_mdd_components(code="296.32")==("Recurrent", "Moderate")\
                                    and get_mdd_components(code="296.2")==("Single Episode", "Unspecified")