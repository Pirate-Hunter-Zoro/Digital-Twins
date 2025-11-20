from typing import List, Dict

structured_delimiter = "# Structured Note"
summary_header = "# Patient Summary Narrative"
medications_header = "## Medications"
diagnoses_header = "## Diagnostics (labs, radiology, vitals, procedures)"

def parse_single_narrative_sections(content: str) -> Dict[str, str]:
    result = {}
    result['summary'] = ""
    result['medications'] = ""
    result['diagnoses'] = ""
    
    result['full'] = content
        
    # Now grab the individual sections of the narrative
    if structured_delimiter in content:
        parts = content.split(structured_delimiter)
        summary = parts[0]
        
        # Clean up the narrative
        summary = summary.replace(summary_header, "").strip()
        # Store the extracted narrative
        result['summary'] = summary
        
        structured_block = parts[1]
        
        # The structured block contains the other things we care about
        if diagnoses_header in structured_block:
            structured_parts = structured_block.split(diagnoses_header)
            medications = structured_parts[0]
            diagnoses = structured_parts[1]
            
            # Clean the extracted parts by removing their headings
            medications = medications.replace(medications_header, "").strip()
            diagnoses = diagnoses.replace(diagnoses_header, "").strip()
            
            # Store extracted parts
            result['medications'] = medications
            result['diagnoses'] = diagnoses   
            
        else:
            # Assume the entire structured block is the medications section
            result['medications'] = structured_block

    else:
        # Assume entire content is narrative
        result['summary'] = content
        
    return result

def parse_narrative_sections(full_narratives: List[str]) -> List[Dict[str,str]]:
    # For each patient, return a dictionary of their summary, medications, diagnoses, and full_text
    results = []
    for narrative in enumerate(full_narratives):
        results.append(parse_single_narrative_sections(narrative))
            
    return results