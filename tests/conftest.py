import pytest
from typing import Self, List, Dict, Any

class MockPatientBuilder:
    
    def __init__(self):
        self.patient = {
            
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
    
    def add_encounter(self, start: int, end: int, diagnoses: List[Dict] = [], procedures: List[Dict] = []) -> Self:
        """
        Add an encounter to the patient's history
        
        :param start: Encounter start date
        :type start: int
        :param end: Encounter end date
        :type end: int
        :param diagnoses: Diagnoses attached to this encounter
        :type diagnoses: List[Dict]
        :param procedures: Procedures attached to this encounter
        :type procedures: List[Dict]
        :return: Return MockPatientBuilder on which this is called to allow chaining of commands
        :rtype: Self
        """
        encounter_dict = {
            "details" : {
                "start_visit" : start,
                "end_visit" : end,
                "diagnoses" : diagnoses,
                "procedures" : procedures,
            }
        }
        self.patient['encounters'].append(encounter_dict)
        return self # Again for command chaining