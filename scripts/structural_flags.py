import re

def get_structural_flags(extracted_data):
    """
    Analyzes ROAR sections for deterministic quality issues.
    Returns a dictionary of lists containing specific error strings.
    """
    flags = {
        "plo": [],
        "methods": [],
        "results": [],
        "improvement_plan": []
    }

    # passive verb list
    passive_verbs = ['understand', 'appreciate', 'gain knowledge', 'become familiar', 'learn about']

    for section, text in extracted_data.items():
        if section not in flags or not text:
            continue
        
        # text cleaning
        text_clean = text.lower().strip()
        words = text_clean.split()
        word_count = len(words)

        # global flags 
        if word_count < 10:
            flags[section].append("Section is too short (Lacks sufficient detail)")
        elif section == "results":
            if word_count > 250:
                flags[section].append("Section is too complex (Over 250 words; consider being more concise)")
        elif word_count > 200:
            flags[section].append("Section is too complex (Over 200 words; consider being more concise)")

        # plo flags
        if section == "plo":
            # multiple goals
            if text_clean.count(' and ') >= 2 or ';' in text_clean or ' as well as ' in text_clean:
                flags["plo"].append("Multiple goals detected (PLO should focus on a single achievement)")
            
            # measurable req
            if any(verb in text_clean for verb in passive_verbs):
                flags["plo"].append("Contains non-measurable verbs consider changing language")

        # methods flags
        if section == "methods":
            # use of a specific tool to assess
            if not any(keyword in text_clean for keyword in ['rubric', 'exam', 'test', 'score', 'standard', 'artifact']):
                flags["methods"].append("No specific assessment instrument mentioned, consider use")
            
            # unallowed measures
            if "gpa" in text_clean or "course grade" in text_clean:
                flags["methods"].append("Uses prohibited indirect measures (GPA or Course Grades)")

        # results flag
        if section == "results_conclusions":
            # quantitative req
            if not any(char.isdigit() for char in text):
                flags["results"].append("No numeric data detected (Results must quantitative measures")

    return flags