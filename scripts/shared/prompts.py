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
    # The phenotype-free rubric. Dimension 1 of the published rubric is headed
    # "Baseline symptom phenotype (PHQ-9 subitems)" and this extract records no PHQ-9
    # item, so the variant drops that dimension and rescales the surviving five from
    # 20/20/20/10/5 to 27/27/27/13/6. Nothing else about the rubric differs, which is
    # what makes the old-versus-new judgement correlation a test of one change.
    _JUDGE_SYSTEM_NO_PHENOTYPE = "judge_system_no_phenotype.txt"
    _JUDGE_USER_NO_PHENOTYPE = "judge_user_no_phenotype.txt"

    def __init__(self) -> None:
        pass

    # -------- system prompt getters --------

    def get_judge_system(self) -> str:
        return self._read(self._JUDGE_SYSTEM)

    def get_judge_system_no_phenotype(self) -> str:
        return self._read(self._JUDGE_SYSTEM_NO_PHENOTYPE)

    # -------- user prompt renderers --------

    def render_judge_user(self, narrative_a: str, narrative_b: str) -> str:
        """
        Formats the two narratives into the user prompt for the judging task.
        """
        template = self._read(filename=self._JUDGE_USER)
        return template.replace("{narrative_a}", narrative_a).replace("{narrative_b}", narrative_b)

    def render_judge_user_no_phenotype(self, narrative_a: str, narrative_b: str) -> str:
        """
        Formats the two narratives into the phenotype-free user prompt.
        """
        template = self._read(filename=self._JUDGE_USER_NO_PHENOTYPE)
        return template.replace("{narrative_a}", narrative_a).replace("{narrative_b}", narrative_b)

    # -------- internals --------

    def _read(self, filename: str) -> str:
        path = self._PROMPTS_DIR / filename
        return path.read_text(encoding="utf-8")