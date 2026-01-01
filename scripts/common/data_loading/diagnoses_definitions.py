from typing import Optional
import re

# TODO - incorporate into narrative production
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

def get_mdd_description(code: str) -> str:
    """
    Helper method for getting the severity and recurrence for the given MDD code
    
    :param code: Input MDD code
    :type code: str
    :return: Describes severity and recurrence of diagnosis
    :rtype: str
    """
    code_segments = code.split(".")
    if len(code_segments) > 1 and code_segments[1] in MDD_SEVERITY_MAP.keys():
        recurrence, severity = MDD_RECURRENCE_MAP[code_segments[0]], MDD_SEVERITY_MAP[code_segments[1]]
    else:
        recurrence, severity = MDD_RECURRENCE_MAP[code_segments[0]], "Unspecified"
    return f"{recurrence}, {severity}"
    

DIAGNOSIS_CODES = {
    "MDD" : [
        r'F32\.\d+',
        r'F33\.\d+'
    ],
    "SOCIAL_ANXIETY" : [
        r'F40\.\d'
    ],
    "ADJUSTMENT_DISORDER" : [
        r'F43\.22'
    ],
    "ANXIETY" : [
        r'F41\.\d'
    ],
    "PTSD" : [
        r'F43\.1\d'
    ],
    "OCD" : [
        r'F42\.\d'
    ],
    "DYSTHYMIA" : [
        r'F34\.1'
    ],
    "SUD" : [
        r'F1\d(\.\d*)?'
    ],
    "SUICIDE_IDEATION" : [
        r'R45\.851'
    ],
    "SUICIDE_ATTEMPT" : [
        r'X7\d',
        r'X8[123]',
        r'T14\.91'
    ],
    "DIABETES" : [
        r'E11\.\d',
    ],
    "HYPERLIPIDEMIA" : [
        r'E78\.\d',
    ],
    "THYROID" : [
        r'E0[35]'
    ],
    "CHRONIC_PAIN" : [
        r'M5[46]',
        r'M79',
        r'G5[6789]',
        r'G6[012345]'
    ],
    "INSOMNIA" : [
        r'G47\.0',
        r'F51\.\d'
    ],
    "EPILEPSY" : [
        r'G40\.\d',
        r'G41\.\d',
        r'R56\.9'
    ],
    "UNCONTROLLED_HTN" : [
        r'I16\.\d'
    ],
    "SDOH" : [
        r'Z5[56789]\.\d',
        r'Z6[012345]\.\d'
    ]
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
SDOH = "SDOH"

# When caring about a particular substance abuse
SUD_MAP = {
    "F10": "Alcohol",
    "F11": "Opioid",
    "F12": "Cannabis",
    "F13": "Sedative/Hypnotic",
    "F14": "Cocaine",
    "F15": "Other Stimulant",
    "F16": "Hallucinogen",
    "F17": "Nicotine",
    "F18": "Inhalant",
    "F19": "Other Substance"
}

PSYCH_ARMS = set([MDD, SOCIAL_ANXIETY, ADJUSTMENT_DISORDER, ANXIETY, PTSD, OCD, DYSTHYMIA, SUD])

SUICIDE_ARMS = set([SUICIDE_IDEATION, SUICIDE_ATTEMPT])

SAFETY_ARMS = set([EPILEPSY, UNCONTROLLED_HTN])

MEDICAL_ARMS = set([DIABETES, HYPERLIPIDEMIA, THYROID, CHRONIC_PAIN, UNCONTROLLED_HTN])

def get_diagnosis_arm(diagnosis_code: str) -> Optional[str]:
    for arm, regex_codes in DIAGNOSIS_CODES.items():
        for regex in regex_codes:
            pattern = re.compile(regex)
            if re.match(pattern, diagnosis_code) != None:
                return arm
    return None