"""Re-judge a sample of cached pairs under the phenotype-free rubric and compare.

Review item (2026-08-28): the judge rubric's first dimension is headed "Baseline symptom
phenotype (PHQ-9 subitems)" and this extract records no PHQ-9 item. The published cache
was measured rather than re-run -- the phenotype sub-score is non-zero for most pairs, so
the model scored that dimension off the MDD coding and psychiatric flags that ARE present
instead of obeying the prompt's "output 0 if Missing" rule -- but a measurement of what
the model did is not the same test as re-judging under a corrected prompt. This module is
that test.

The design is deliberately narrow. One change to the prompt (dimension 1 removed, the
surviving five rescaled from 20/20/20/10/5 to 27/27/27/13/6), one LLM instance, a sample
of pairs that already carry a cached judgement, and a correlation between the old score
and the new one. Isolating a single change is what makes the correlation interpretable;
the rubric's two other wording faults -- dimension 5 naming no-show behaviour, dimension
4 saying three years where the lookback is two -- are deliberately left in place.

The decision the correlation drives:
  close agreement    -> the 1.71M-judgement cache stands, and the supplement states that
                        no PHQ-9 item exists in this extract.
  material divergence-> the cache is re-judged in full and the supplement must print the
                        corrected prompt, since that would then be the prompt used.

Only overall_similarity ever becomes a neighbour weight, and it enters as
(score/100)**WEIGHTING_EXPONENT, so the weight correlation is reported beside the raw
score correlation: a monotone but non-affine shift in level moves the weights even when
the ranking is intact.

Nothing here writes to the canonical judgements.db. New judgements go to their own
database file with its own table, so the published cache cannot be touched.
"""

import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from dotenv import load_dotenv
load_dotenv()

from scripts.models.vllm_client import VllmClient
from scripts.pipeline.review.paths import review_output_dir
from scripts.shared.prompts import PromptLoader

ANALYSIS_NAME = "judge_prompt"
NEW_DB_NAME = "judgements_no_phenotype.db"
NEW_TABLE = "llm_judgements_no_phenotype"

# The response schema with the phenotype field removed. Everything else is byte-identical
# to scorer.guided_json, so a divergence cannot be blamed on a different output contract.
guided_json_no_phenotype = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "overall_similarity",
        "psych_comorbidity",
        "metabolic_pain",
        "treatment_burden",
        "social_functional",
        "safety",
        "top_similarity_drivers",
        "key_mismatches",
    ],
    "properties": {
        "overall_similarity": {"type": "integer"},
        "psych_comorbidity": {"type": "integer"},
        "metabolic_pain": {"type": "integer"},
        "treatment_burden": {"type": "integer"},
        "social_functional": {"type": "integer"},
        "safety": {"type": "integer"},
        "top_similarity_drivers": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "key_mismatches": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
    },
}


def canonical_db_path() -> Path:
    """Path of the published 1.71M-judgement cache.

    Returns:
        Path: JUDGEMENTS_DIR/judgements.db.
    """
    return Path(os.environ['JUDGEMENTS_DIR']) / "judgements.db"


def new_db_path() -> Path:
    """Path of this analysis' own judgement database.

    Returns:
        Path: ARTIFACTS_DIR/review/judge_prompt/judgements_no_phenotype.db.
    """
    return review_output_dir(ANALYSIS_NAME) / NEW_DB_NAME


def init_new_db() -> sqlite3.Connection:
    """Open (creating if absent) the phenotype-free judgement cache.

    Returns:
        sqlite3.Connection: Connection with the table guaranteed to exist.
    """
    connection = sqlite3.connect(new_db_path(), timeout=30.0)
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute(f'''
CREATE TABLE IF NOT EXISTS {NEW_TABLE} (
    id_a TEXT,
    id_b TEXT,
    overall_score INTEGER,
    full_response TEXT,
    PRIMARY KEY (id_a, id_b)
);
''')
    connection.commit()
    return connection


def sample_cached_pairs(n_pairs: int) -> pd.DataFrame:
    """Draw a reproducible uniform sample of already-judged pairs.

    SQLite's RANDOM() cannot be seeded, so the row identifiers are read out first and the
    sample is drawn with a SEED-derived numpy generator. That makes the sampled pair set a
    function of (SEED, n_pairs, cache contents) rather than of when the job happened to run.

    Args:
        n_pairs (int): How many pairs to draw.

    Returns:
        pd.DataFrame: Columns id_a, id_b, old_overall, old_phenotype -- one row per pair.
    """
    connection = sqlite3.connect(f"file:{canonical_db_path()}?mode=ro", uri=True)
    cursor = connection.cursor()
    cursor.execute(f"SELECT rowid FROM llm_judgements")
    rowids = np.array([row[0] for row in cursor.fetchall()], dtype=np.int64)
    print(f"Canonical cache holds {len(rowids):,} judgements.", flush=True)
    rng = np.random.default_rng(int(os.environ['SEED']))
    chosen = rng.choice(rowids, size=min(n_pairs, len(rowids)), replace=False)
    chosen.sort()
    rows = []
    for rowid in chosen:
        cursor.execute(
            "SELECT id_a, id_b, overall_score, full_response FROM llm_judgements WHERE rowid=?",
            (int(rowid),),
        )
        id_a, id_b, overall, full_response = cursor.fetchone()
        parsed = json.loads(full_response)
        rows.append({
            'id_a': id_a,
            'id_b': id_b,
            'old_overall': int(overall),
            'old_phenotype': parsed.get('phenotype'),
        })
    connection.close()
    return pd.DataFrame(rows)


def load_narrative(patient_id: str) -> str:
    """Read one patient's baseline narrative -- the text the judge was shown.

    Args:
        patient_id (str): Cohort patient identifier.

    Returns:
        str: The narrative markdown.
    """
    return (Path(os.environ['NARRATIVES_DIR']) / f"{patient_id}.md").read_text()


def already_judged(connection: sqlite3.Connection) -> set[tuple[str, str]]:
    """Pairs already present in the phenotype-free cache, so a resume skips them.

    Args:
        connection (sqlite3.Connection): Open connection to the new database.

    Returns:
        set[tuple[str, str]]: (id_a, id_b) keys.
    """
    cursor = connection.execute(f"SELECT id_a, id_b FROM {NEW_TABLE}")
    return set(cursor.fetchall())


async def judge_pair(
    client: VllmClient,
    loader: PromptLoader,
    semaphore: asyncio.Semaphore,
    id_a: str,
    id_b: str,
) -> Optional[dict]:
    """Score one pair under the phenotype-free rubric.

    Args:
        client (VllmClient): Shared async client pointed at the single server.
        loader (PromptLoader): Prompt loader.
        semaphore (asyncio.Semaphore): Concurrency gate, sized by LLM_MAX_CONCURRENCY.
        id_a (str): First patient (the cache's canonical ordering, id_a <= id_b).
        id_b (str): Second patient.

    Returns:
        Optional[dict]: The parsed response, or None if every attempt failed.
    """
    system_prompt = loader.get_judge_system_no_phenotype()
    user_prompt = loader.render_judge_user_no_phenotype(
        narrative_a=load_narrative(id_a), narrative_b=load_narrative(id_b)
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    async with semaphore:
        for _ in range(3):
            try:
                response = await client.chat_async(
                    messages=messages, guided_json=guided_json_no_phenotype
                )
                return json.loads(response)
            except Exception as exc:
                print(f"{id_a} vs {id_b}: {repr(exc)}", file=sys.stderr, flush=True)
    return None


async def rejudge(pairs: pd.DataFrame) -> pd.DataFrame:
    """Re-judge every sampled pair not already cached, writing results as they land.

    Args:
        pairs (pd.DataFrame): Output of sample_cached_pairs.

    Returns:
        pd.DataFrame: pairs with a new_overall column joined on (id_a, id_b).
    """
    connection = init_new_db()
    done = already_judged(connection)
    todo = [
        (row.id_a, row.id_b)
        for row in pairs.itertuples()
        if (row.id_a, row.id_b) not in done
    ]
    print(f"{len(pairs):,} sampled pairs; {len(done):,} already judged; {len(todo):,} to judge.", flush=True)

    if todo:
        client = VllmClient()
        loader = PromptLoader()
        semaphore = asyncio.Semaphore(int(os.environ['LLM_MAX_CONCURRENCY']))
        batch_size = 200
        failures = 0
        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            responses = await asyncio.gather(*[
                judge_pair(client, loader, semaphore, id_a, id_b) for (id_a, id_b) in batch
            ])
            writable = [
                (id_a, id_b, int(response['overall_similarity']), json.dumps(response, indent=4))
                for (id_a, id_b), response in zip(batch, responses)
                if response is not None
            ]
            failures += len(batch) - len(writable)
            connection.executemany(
                f"INSERT OR REPLACE INTO {NEW_TABLE} (id_a, id_b, overall_score, full_response) VALUES (?, ?, ?, ?)",
                writable,
            )
            connection.commit()
            print(f"  judged {min(start + batch_size, len(todo)):,}/{len(todo):,} ({failures} failures)", flush=True)

    scored = pd.read_sql_query(
        f"SELECT id_a, id_b, overall_score AS new_overall, full_response FROM {NEW_TABLE}", connection
    )
    connection.close()
    sub_scores = scored['full_response'].apply(json.loads)
    for field in ('psych_comorbidity', 'metabolic_pain', 'treatment_burden', 'social_functional', 'safety'):
        scored[f"new_{field}"] = sub_scores.apply(lambda d: d.get(field))
    scored = scored.drop(columns=['full_response'])
    return pairs.merge(scored, on=['id_a', 'id_b'], how='inner')


def compare(judged: pd.DataFrame) -> dict:
    """Statistics on old versus new overall similarity, and on the derived weight.

    The weight is what the KNN arm actually consumes: (score/100)**WEIGHTING_EXPONENT.
    Reporting both separates two different failure modes -- a re-ranking of neighbours
    (Spearman falls) from a level shift that preserves ranking but changes how sharply
    the weights concentrate (Spearman holds, weight correlation falls).

    Args:
        judged (pd.DataFrame): Output of rejudge, one row per pair with both scores.

    Returns:
        dict: Correlations, agreement, and the two score distributions.
    """
    old = judged['old_overall'].to_numpy(dtype=float)
    new = judged['new_overall'].to_numpy(dtype=float)
    alpha = float(os.environ['WEIGHTING_EXPONENT'])
    old_weight = (old / 100.0) ** alpha
    new_weight = (new / 100.0) ** alpha
    difference = new - old
    pearson = stats.pearsonr(old, new)
    spearman = stats.spearmanr(old, new)
    weight_pearson = stats.pearsonr(old_weight, new_weight)
    weight_spearman = stats.spearmanr(old_weight, new_weight)
    old_phenotype = judged['old_phenotype'].dropna().to_numpy(dtype=float)
    return {
        'n_pairs': int(len(judged)),
        'pearson_r': float(pearson.statistic),
        'pearson_p': float(pearson.pvalue),
        'spearman_rho': float(spearman.statistic),
        'spearman_p': float(spearman.pvalue),
        'weight_pearson_r': float(weight_pearson.statistic),
        'weight_spearman_rho': float(weight_spearman.statistic),
        'mean_old': float(old.mean()),
        'mean_new': float(new.mean()),
        'sd_old': float(old.std(ddof=1)),
        'sd_new': float(new.std(ddof=1)),
        'mean_signed_difference': float(difference.mean()),
        'mean_absolute_difference': float(np.abs(difference).mean()),
        'median_absolute_difference': float(np.median(np.abs(difference))),
        'p95_absolute_difference': float(np.percentile(np.abs(difference), 95)),
        'share_within_5_points': float((np.abs(difference) <= 5).mean()),
        'share_within_10_points': float((np.abs(difference) <= 10).mean()),
        'old_phenotype_mean': float(old_phenotype.mean()) if len(old_phenotype) else None,
        'old_phenotype_zero_share': float((old_phenotype == 0).mean()) if len(old_phenotype) else None,
    }


def plot_comparison(judged: pd.DataFrame, statistics: dict, save_dir: Path) -> Path:
    """Two panels: old against new, and the distribution of their difference.

    Args:
        judged (pd.DataFrame): Output of rejudge.
        statistics (dict): Output of compare, used for the annotation.
        save_dir (Path): Where to write the PNG.

    Returns:
        Path: The written figure.
    """
    old = judged['old_overall'].to_numpy(dtype=float)
    new = judged['new_overall'].to_numpy(dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    hexbin = axes[0].hexbin(old, new, gridsize=40, cmap='viridis', mincnt=1)
    limits = [min(old.min(), new.min()) - 2, max(old.max(), new.max()) + 2]
    axes[0].plot(limits, limits, color='crimson', linewidth=1.2, linestyle='--', label='identity')
    axes[0].set_xlim(limits)
    axes[0].set_ylim(limits)
    axes[0].set_xlabel("Published rubric: overall similarity")
    axes[0].set_ylabel("Phenotype-free rubric: overall similarity")
    axes[0].set_title(
        f"n = {statistics['n_pairs']:,} pairs\n"
        f"Pearson r = {statistics['pearson_r']:.3f}, Spearman rho = {statistics['spearman_rho']:.3f}"
    )
    axes[0].legend(loc='upper left', fontsize=8)
    figure.colorbar(hexbin, ax=axes[0], label='pairs per cell')

    difference = new - old
    axes[1].hist(difference, bins=41, color='steelblue', edgecolor='white')
    axes[1].axvline(0.0, color='black', linewidth=1.0)
    axes[1].axvline(
        statistics['mean_signed_difference'], color='crimson', linewidth=1.4,
        label=f"mean {statistics['mean_signed_difference']:+.2f}",
    )
    axes[1].set_xlabel("New minus published (points)")
    axes[1].set_ylabel("Pairs")
    axes[1].set_title(
        f"Mean |difference| = {statistics['mean_absolute_difference']:.2f} points; "
        f"{statistics['share_within_10_points'] * 100:.1f}% within 10"
    )
    axes[1].legend(loc='upper right', fontsize=8)

    figure.suptitle("Judge rubric with the PHQ-9 phenotype dimension removed", fontsize=11)
    figure.tight_layout()
    save_path = save_dir / "judge_prompt_agreement.png"
    figure.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(figure)
    return save_path
