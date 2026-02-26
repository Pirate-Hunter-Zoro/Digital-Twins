from pathlib import Path

SCORE_PATTERN = r"[\*\s]*SCORE[\*\s]*:\s*(\d*\.?\d+)"
EXPLANATION_PATTERN = r"[\*\s]*EXPLANATION[\*\s]*:\s*(.*)"

class PromptLoader:
    """
    Strict prompt loader that assumes templates live in ./prompts.
    """

    _PROMPTS_DIR = Path("prompts")

    _JUDGE_SYSTEM = "judge_system.txt"
    _JUDGE_USER = "judge_user.txt"

    def __init__(self) -> None:
        pass

    # -------- system prompt getters --------

    def get_judge_system(self) -> str:
        return self._read(self._JUDGE_SYSTEM)

    # -------- user prompt renderers --------

    def render_judge_user(self, narrative_a: str, narrative_b: str) -> str:
        """
        Formats the two narratives into the user prompt for the judging task.
        """
        template = self._read(filename=self._JUDGE_USER)
        return template.replace("{narrative_a}", narrative_a).replace("{narrative_b}", narrative_b)

    # -------- internals --------

    def _read(self, filename: str) -> str:
        path = self._PROMPTS_DIR / filename
        return path.read_text(encoding="utf-8")