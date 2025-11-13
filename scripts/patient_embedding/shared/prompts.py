import json
from pathlib import Path

SCORE_PATTERN = r"[\*\s]*SCORE[\*\s]*:\s*(\d*\.?\d+)"
EXPLANATION_PATTERN = r"[\*\s]*EXPLANATION[\*\s]*:\s*(.*)"

class PromptLoader:
    """
    Strict prompt loader that assumes templates live in ./prompts.
    """

    _PROMPTS_DIR = Path("prompts")

    _NARRATIVE_SYSTEM_INITIAL = "narrative_system_initial.txt"
    _NARRATIVE_USER_INITIAL = "narrative_user_initial.txt"
    _NARRATIVE_SYSTEM_EXTRACTION = "narrative_system_extraction.txt"
    _NARRATIVE_USER_EXTRACTION = "narrative_user_extraction.txt"
    _NARRATIVE_USER_FINALIZATION = "narrative_user_finalization.txt"
    _NARRATIVE_SYSTEM_FINALIZATION = "narrative_system_finalization.txt"
    _JUDGE_SYSTEM = "judge_system.txt"

    def __init__(self) -> None:
        pass

    # -------- system prompt getters --------

    def get_narrative_system_initial(self) -> str:
        return self._read(self._NARRATIVE_SYSTEM_INITIAL)

    def get_narrative_system_extraction(self) -> str:
        return self._read(self._NARRATIVE_SYSTEM_EXTRACTION)
    
    def get_narrative_system_finalization(self) -> str:
        return self._read(self._NARRATIVE_SYSTEM_FINALIZATION)

    def get_judge_system(self) -> str:
        return self._read(self._JUDGE_SYSTEM)

    # -------- user prompt renderers --------

    def render_narrative_user_extraction(self, patient_json: dict) -> str:
        return self._render_user(self._NARRATIVE_USER_EXTRACTION, patient_json)
    
    def render_narrative_user_initial(self, patient_json: dict) -> str:
        return self._render_user(self._NARRATIVE_USER_INITIAL, patient_json)
    
    def render_narrative_user_finalization(self, base_narrative: str, additional_visits: list[str]) -> str:
        return self._render_user_finalization(self._NARRATIVE_USER_FINALIZATION, base_narrative, additional_visits)

    def render_judge_user(self, narrative_a: str, narrative_b: str) -> str:
        """
        Formats the two narratives into the user prompt for the judging task.
        """
        # This is the simple, direct format the user prompt should have.
        return (
            f"PATIENT NARRATIVE 1:\n---\n{narrative_a}\n---\n\n"
            f"PATIENT NARRATIVE 2:\n---\n{narrative_b}\n---"
        )

    # -------- internals --------

    def _read(self, filename: str) -> str:
        path = self._PROMPTS_DIR / filename
        return path.read_text(encoding="utf-8")

    def _render_user(self, filename: str, patient_json: dict) -> str:
        tmpl = self._read(filename)
        payload = json.dumps(patient_json, ensure_ascii=False)
        if "{patient_json}" in tmpl:
            return tmpl.replace("{patient_json}", payload)
        return tmpl
    
    def _render_user_finalization(self, filename: str, base_narrative: str, additional_visits: list[str]) -> str:
        tmpl = self._read(filename)
        if "{base_narrative}" in tmpl:
            tmpl = tmpl.replace("{base_narrative}", base_narrative)
        if "{additional_visits}" in tmpl:
            tmpl = tmpl.replace("{additional_visits}", "\n---\n".join(additional_visits))
        return tmpl