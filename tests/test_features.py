import sys
from pathlib import Path
from tests.conftest import MockPatientBuilder

# We need to tell python where the scripts directory is - the root project directory
sys.path.append(Path(__file__).parent.parent)
from scripts.common.data_loading.features import (
    prior_adequate_trials, 
    benzo_days, 
    augmentation_flag,
    polypharmacy
)
from scripts.common.data_loading.med_definitions import SSRI

def test_prior_adequate_trials_just_short(builder: MockPatientBuilder):
    """
    For a drug category to count, we need at least six weeks of an active medication of said category within our window.
    41 days does not count
    
    :param builder: Patient json creator
    :type builder: MockPatientBuilder
    """
    patient_dict = builder.add_active_med(name='Fluoxetine', start=-41, end=0).build()
    features_with_adequate_trials = prior_adequate_trials(patient_dict=patient_dict)
    assert features_with_adequate_trials[SSRI] == 0
    
def test_prior_adequate_trials_just_success(builder: MockPatientBuilder):
    """
    For a drug category to count, we need at least six weeks of an active medication of said category within our window.
    42 days counts
    
    :param builder: Patient json creator
    :type builder: MockPatientBuilder
    """
    patient_dict = builder.add_active_med(name='Fluoxetine', start=-42, end=0).build()
    features_with_adequate_trials = prior_adequate_trials(patient_dict=patient_dict)
    assert features_with_adequate_trials[SSRI] == 1
    
def test_prior_adequate_trails_ongoing(builder: MockPatientBuilder):
    """
    For a drug category to count, we need at least six weeks of an active medication of said category within our window.
    50 days prior to anchor and still ongoing counts
    
    :param builder: Patient json creator
    :type builder: MockPatientBuilder
    """
    patient_dict = builder.add_active_med(name='Fluoxetine', start=-50, end='ongoing').build()
    features_with_adequate_trials = prior_adequate_trials(patient_dict=patient_dict)
    assert features_with_adequate_trials[SSRI] == 1
    
def test_benzo_days_overlap_logic(builder: MockPatientBuilder):
    """
    For the total number of days of coverage for an ingredient, we do not double count overlapping ingredients as additional days
    
    :param builder: Patient json creator
    :type builder: MockPatientBuilder
    """
    patient_dict = builder.add_active_med(name='Alprazolam', start=-20, end=-10)\
                            .add_active_med(name='Alprazolam', start=-15, end=-5)\
                            .build()
    assert benzo_days(patient_dict=patient_dict) == 15
    
def test_augmentation_flag_overlap_threshold(builder: MockPatientBuilder):
    """
    If resulting overlaps from an antidepressant and lithium/antipsychotics only last for 13 days (edge case < 14), then it does not count as an augmentation
    
    :param builder: Patient json creator
    :type builder: MockPatientBuilder
    """
    patient_dict = builder.add_active_med(name="Sertraline", start=-100, end=-50)\
                            .add_active_med(name='Lithium', start=-63, end=0)\
                            .build()
    assert not augmentation_flag(patient_dict=patient_dict)
    
def test_augmentation_flag_overlap_success(builder: MockPatientBuilder):
    """
    If resulting overlaps from an antidepressant and lithium/antipsychotics is 14 days, then it counts as an augmentation
    
    :param builder: Patient json creator
    :type builder: MockPatientBuilder
    """
    patient_dict = builder.add_active_med(name="Sertraline", start=-100, end=-50)\
                            .add_active_med(name='Lithium', start=-64, end=0)\
                            .build()
    assert augmentation_flag(patient_dict=patient_dict)
    
def test_polypharmacy_grouping(builder: MockPatientBuilder):
    """
    Ensure only distinc ingredients contribute to the polypharmacy count
    """
    patient_dict = builder.add_active_med(name="Sertraline 50mg", start=-100, end=-20)\
        .add_active_med(name="Sertraline 100mg", start=-20, end="ongoing")\
            .add_active_med(name="Ibuprofen", start=-15, end="ongoing").build()
    assert len(polypharmacy(patient_dict=patient_dict)) == 2 # sertraline and ibuprofen
    
# def test_suicidality_flag_time_window(builder: MockPatientBuilder):
#     """
#     Ensure suicidality flag only pertains to the last year and not prior
#     """
#     patient_dict = builder.add_encounter(start=-400, end=-400,  diagnoses=[{}])
    
# def test_psych_utilization_days(builder: MockPatientBuilder):
#     """
#     Ensure summing of inpatient days behaves as expected
#     """
#     patient_dict = builder.add_encounter()