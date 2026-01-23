from typing import Optional
import re

MDD_SEVERITY_MAP = {
    '0': 'Mild',
    '1': 'Moderate',
    '2': 'Severe',
    '3': 'Psychotic',
    '4': 'Remission'
}
MDD_RECURRENCE_MAP = {
    'F32': 'Single Episode',
    'F33': 'Recurrent'
}

ICD9_MDD_SEVERITY_MAP = {
    '0' : "Unspecified",
    '1' : "Mild",
    '2' : "Moderate",
    '3' : "Severe",
    '4' : "Psychotic",
    '5' : "Remission",
    '6' : "Remission"
}

def get_mdd_description(code: str) -> str:
    """
    Helper method for getting the severity and recurrence for the given MDD code
    
    :param code: Input MDD code
    :type code: str
    :return: Describes severity and recurrence of diagnosis
    :rtype: str
    """
    if code == "300.4":
        return "Dysthymia, Unspecified"
    code_segments = code.split(".")
    
    # ICD-9 Check
    if len(code_segments) > 1 and code_segments[0] == "296":
        suffix = code_segments[1]
        recurrence = ''
        if suffix[0] == '2':
            recurrence = 'Single Episode'
        elif suffix[0] == '3':
            recurrence = 'Recurrent'
        if len(suffix) > 1:
            severity = ICD9_MDD_SEVERITY_MAP.get(suffix[1], 'Unspecified')
        else:
            severity = 'Unspecified'
    else:
        # ICD-10 Check
        if len(code_segments) > 1 and code_segments[0] in MDD_RECURRENCE_MAP.keys() and code_segments[1] in MDD_SEVERITY_MAP.keys():
            recurrence, severity = MDD_RECURRENCE_MAP[code_segments[0]], MDD_SEVERITY_MAP[code_segments[1]]
        elif code_segments[0] in MDD_RECURRENCE_MAP.keys():
            recurrence, severity = MDD_RECURRENCE_MAP[code_segments[0]], "Unspecified"
        else:
            # Non-mdd code
            return None
        
    return f"{recurrence}, {severity}"
    

DIAGNOSIS_CODES = {
    "MDD" : [
        r'F32\.\d+',
        r'F33\.\d+',
        r'296\.2\d*', 
        r'296\.3\d*'
    ],
    "SOCIAL_ANXIETY" : [
        r'F40\.\d',
        r'300\.23'
    ],
    "ADJUSTMENT_DISORDER" : [
        r'F43\.22',
        r'309\.\d'
    ],
    "ANXIETY" : [
        r'F41\.\d',
        r'300\.0\d'
    ],
    "PTSD" : [
        r'F43\.1\d',
        r'309\.81'
    ],
    "OCD" : [
        r'F42\.\d',
        r'300\.3'
    ],
    "DYSTHYMIA" : [
        r'F34\.1',
        r'300\.4',
    ],
    "SUD" : [
        r'F1\d(\.\d*)?',
        r'291\.\d', 
        r'292\.\d', 
        r'303\.\d', 
        r'304\.\d', 
        r'305\.\d'
    ],
    "SUICIDE_IDEATION" : [
        r'R45\.851',
        r'V62\.84'
    ],
    "SUICIDE_ATTEMPT" : [
        r'X7\d',
        r'X8[123]',
        r'T14\.91',
        r'E95\d(\.\d)?'
    ],
    "DIABETES" : [
        r'E11\.\d',
        r'250\.\d+',
    ],
    "HYPERLIPIDEMIA" : [
        r'E78\.\d',
        r'272\.\d',
    ],
    "THYROID" : [
        r'E0[35]',
        r'240\.\d', 
        r'241\.\d', 
        r'242\.\d', 
        r'244\.\d'
    ],
    "CHRONIC_PAIN" : [
        r'M5[46]',
        r'M79',
        r'G5[6789]',
        r'G6[012345]',
        r'724\.\d', 
        r'729\.1', 
        r'337\.\d', 
        r'35[0-7]\.\d'
    ],
    "INSOMNIA" : [
        r'G47\.0',
        r'F51\.\d',
        r'780\.5\d', 
        r'307\.4\d'
    ],
    "EPILEPSY" : [
        r'G40\.\d',
        r'G41\.\d',
        r'R56\.9',
        r'345\.\d', 
        r'780\.3\d'
    ],
    "UNCONTROLLED_HTN" : [
        r'I16\.\d',
        r'401\.9', 
        r'401\.0'
    ],
}
MDD = "MDD"
SOCIAL_ANXIETY = "SOCIAL_ANXIETY"
ADJUSTMENT_DISORDER = "ADJUSTMENT_DISORDER"
ANXIETY = "ANXIETY"
PTSD = "PTSD"
OCD = "OCD"
DYSTHYMIA = "DYSTHYMIA"
SUD = "SUD"
SUICIDE_IDEATION = "SUICIDE_IDEATION"
SUICIDE_ATTEMPT = "SUICIDE_ATTEMPT"
DIABETES = "DIABETES"
HYPERLIPIDEMIA = "HYPERLIPIDEMIA"
THYROID = "THYROID"
CHRONIC_PAIN = "CHRONIC_PAIN"
INSOMNIA = "INSOMNIA"
EPILEPSY = "EPILEPSY"
UNCONTROLLED_HTN = "UNCONTROLLED_HTN"

# When caring about a particular substance abuse
def get_sud_substance(code: str) -> str:
    """
    Determines the substance class for both ICD-10 and ICD-9 codes.
    """
    # ICD-10 Logic (Simple Prefix)
    if code.startswith("F"):
        prefix = code[:3]
        if prefix == "F10": return "Alcohol"
        if prefix == "F11": return "Opioid"
        if prefix == "F12": return "Cannabis"
        if prefix == "F13": return "Sedative/Hypnotic"
        if prefix == "F14": return "Cocaine"
        if prefix == "F15": return "Other Stimulant"
        if prefix == "F16": return "Hallucinogen"
        if prefix == "F17": return "Nicotine"
        if prefix == "F18": return "Inhalant"
        return "Other Substance"

    # ICD-9 Logic (Complex mapping)
    # Alcohol: 291 (Induced), 303 (Dependence), 305.0 (Abuse)
    if code.startswith("291") or code.startswith("303") or code.startswith("305.0"):
        return "Alcohol"
    
    # Opioid: 304.0, 305.5
    if code.startswith("304.0") or code.startswith("305.5"):
        return "Opioid"

    # Cannabis: 304.3, 305.2
    if code.startswith("304.3") or code.startswith("305.2"):
        return "Cannabis"

    # Sedative: 304.1, 305.4
    if code.startswith("304.1") or code.startswith("305.4"):
        return "Sedative/Hypnotic"

    # Cocaine: 304.2, 305.6
    if code.startswith("304.2") or code.startswith("305.6"):
        return "Cocaine"

    # Stimulants (Amphetamines): 304.4, 305.7
    if code.startswith("304.4") or code.startswith("305.7"):
        return "Other Stimulant"

    # Hallucinogens: 304.5, 305.3
    if code.startswith("304.5") or code.startswith("305.3"):
        return "Hallucinogen"

    # Nicotine: 305.1
    if code.startswith("305.1"):
        return "Nicotine"
        
    return "Other Substance"

# NOTE - by design all patients have MDD, so we are omitting that here
PSYCH_ARMS = set([SOCIAL_ANXIETY, ADJUSTMENT_DISORDER, ANXIETY, PTSD, OCD, DYSTHYMIA, SUD, INSOMNIA])

SUICIDE_ARMS = set([SUICIDE_IDEATION, SUICIDE_ATTEMPT])

SAFETY_ARMS = set([EPILEPSY, UNCONTROLLED_HTN])

MEDICAL_ARMS = set([DIABETES, HYPERLIPIDEMIA, THYROID, CHRONIC_PAIN])

def get_diagnosis_arm(diagnosis_code: str) -> Optional[str]:
    for arm, regex_codes in DIAGNOSIS_CODES.items():
        for regex in regex_codes:
            pattern = re.compile(regex)
            if re.match(pattern, diagnosis_code) != None:
                return arm
    return None

SDOH_MAP = {
    "Z55": "Education/Literacy",
    "Z56": "Employment",
    "Z57": "Occupational Exposure",
    "Z59": "Housing/Economic",
    "Z60": "Social Environment",
    "Z62": "Upbringing", # NOTE: Often pediatric, but keep if cohort includes it in history
    "Z63": "Primary Support Group/Family",
    "Z64": "Psychosocial Circumstances", # e.g. Unwanted pregnancy
    "Z65": "Legal/Crime/Other Psychosocial"
}