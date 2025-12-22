import pytest
from typing import Self, List, Dict, Any

class MockPatientBuilder:
    
    def __init__(self):
        self.patient = {
            'active_medications' : [],
            'encounters' : [],
        }
    
    def add_active_med(self, name: str, start: int, end: Any, generic: str=None) -> Self:
        """
        Add an active medication to the patient's history with the given name, start date, end date, and generic name 
        
        :param name: Name of medication
        :type name: str
        :param start: Start date of medication
        :type start: int
        :param end: End date of medication (which may be a string to indicate it is still ongoing)
        :type end: Any
        :param generic: Generic name of medication (optional)
        :type generic: str
        :return: MockPatientBuilder this is called on to enable command chaining
        :rtype: Self
        """
        if generic is None:
            generic = name.lower()
        med_dict = {
            "MedName" : name,
            "MedSimpleGenericName" : generic,
            "MedStartInstant" : start,
            "MedEndInstant" : end,
        }
        self.patient['active_medications'].append(med_dict)
        return self # So that we can chain commands
    
    def add_explicit_encounter(self, start: int, end: int, patient_class: str) -> Self:
        """
        Manually add an encounter with a specific class (e.g., Inpatient, Emergency).
        Useful for testing utilization metrics.
        
        :param start: Description
        :type start: Start date (relative to anchor in days)
        :param end: Description
        :type end: End date (relative to anchor in days)
        :param patient_class: Class of patient for this encounter
        :type patient_class: str
        :return: MockPatientBuilder this is called on to enable command chaining
        :rtype: Self
        """
        encounter = {
            "details": {
                "start_visit": start,
                "end_visit": end,
                "patient_class": patient_class
            },
            "diagnoses": [],
            "procedures": []
        }
        self.patient['encounters'].append(encounter)
        return self
    
    def _get_active_encounter(self, date: int) -> Dict:
        """
        Find the encounter that corresponds to the given date for the patinet
        
        :param date: Date of interest
        :type date: int
        :return: Respective encounter
        :rtype: Dict
        """
        for encounter in self.patient['encounters']:
            details = encounter['details']
            if details['start_visit'] <= date and details['end_visit'] >= date:
                return encounter
        # No corresponding encounter existed in the patient's history
        new_encounter = {
            'details': {
                'start_visit': date,
                'end_visit': date,
            },
            'diagnoses': [],
            'procedures': [],
        }
        self.patient['encounters'].append(new_encounter)
        return new_encounter
    
    def add_diagnosis(self, code: str, date: int, description: str, vocab: str="ICD-10-CM") -> Self:
        """
        Add the diagnosis with the given information to the patient's history
        
        :param code: Diagnosis code
        :type code: str
        :param date: Date (relative to anchor in days) of diagnosis
        :type date: int
        :param description: Description/name of diagnosis
        :type description: str
        :param vocab: Vocab corresponding to the given code
        :type vocab: str
        :return: MockPatientBuilder this is called on to enable command chaining
        :rtype: Self
        """
        self._get_active_encounter(date=date)['diagnoses'].append(
            {
                "name": description,
                "codes": [
                    {
                        "code": code,
                        "vocab": vocab,
                        "description": description
                    }
                ]
            }
        )
        return self # Again for command chaining
    
    def add_procedure(self, description: str, date: int) -> Self:
        """
        Add the procedure with the given information to the patient's history
        
        :param description: Description of the procedure
        :type description: str
        :param date: Date of the procedure (relative to anchor in days)
        :type date: int
        :return: MockPatientBuilder this is called on to enable command chaining
        :rtype: Self
        """
        self._get_active_encounter(date=date)['procedures'].append(
            {
                'Procedure_Description': description, 
                'ProcedureStartInstant': date, 
                'ProcedureEndInstant': date
            }
        )
    
    def build(self) -> Dict:
        """
        Return resulting patient dictionary
        
        :return: Patient dictionary of this MockPatientBuilder object
        :rtype: Dict
        """
        return self.patient
    
@pytest.fixture
def builder() -> MockPatientBuilder:
    """
    Return new instance of a MockPatientBuilder
    
    :return: MockPatientBuilder instance
    :rtype: MockPatientBuilder
    """
    return MockPatientBuilder()