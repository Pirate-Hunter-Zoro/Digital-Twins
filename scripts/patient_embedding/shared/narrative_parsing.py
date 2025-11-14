from typing import List, Dict

structured_delimiter = "# Structured Note"
segment_narrative_header = "# Patient Summary Narrative"
segment_medications_header = "## Medications"
segment_diagnoses_header = "## Diagnostics (labs, radiology, vitals, procedures)"

def parse_single_narrative_sections(content: str) -> Dict[str, str]:
    result = {}
    result['summary'] = ""
    result['medications'] = ""
    result['diagnoses'] = ""
    
    result['full'] = content
        
    # Now grab the individual sections of the narrative
    if structured_delimiter in content:
        parts = content.split(structured_delimiter)
        segment_narrative = parts[0]
        
        # Clean up the narrative
        segment_narrative = segment_narrative.replace(segment_narrative_header, "").strip()
        # Store the extracted narrative
        result['summary'] = segment_narrative
        
        structured_block = parts[1]
        
        # The structured block contains the other things we care about
        if segment_diagnoses_header in structured_block:
            structured_parts = structured_block.split(segment_diagnoses_header)
            segment_medications = structured_parts[0]
            segment_diagnoses = structured_parts[1]
            
            # Clean the extracted parts by removing their headings
            segment_medications = segment_medications.replace(segment_medications_header, "").strip()
            segment_diagnoses = segment_diagnoses.replace(segment_diagnoses_header, "").strip()
            
            # Store extracted parts
            result['medications'] = segment_medications
            result['diagnoses'] = segment_diagnoses   
            
        else:
            # Assume the entire structured block is the medications section
            result['medications'] = structured_block

    else:
        # Assume entire content is narrative
        result['summary'] = content
        
    return result

def parse_narrative_sections(full_narratives: List[str]) -> List[Dict[str,str]]:
    # For each patient, return a dictionary of their segment_narrative, segment_medications, segment_diagnoses, and full_text
    results = []
    for narrative in enumerate(full_narratives):
        results.append(parse_single_narrative_sections(narrative))
            
    return results