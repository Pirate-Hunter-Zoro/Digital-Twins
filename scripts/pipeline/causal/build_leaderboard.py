from pathlib import Path
import os
import json
import pandas as pd
from statsmodels.stats.multitest import multipletests

from dotenv import load_dotenv
load_dotenv()

results_path = Path(os.environ['ARTIFACTS_DIR']) / 'causal_pipeline'
os.makedirs(results_path, exist_ok=True)
# Should already exist
results = list(results_path.glob("results_*.json"))
records = []
for res in results:
    with open(res, 'r') as f:
        records.append(json.load(f))
passed = []
skipped = []
for record in records:
    if record['passed_overlap']:
        passed.append(record)
    else:
        skipped.append(record)
if passed:
    # Non-empty
    blp_p_vals = [record['blp_res']['blp_pval'] for record in passed]
    rejected, corrected_p_vals, _, _ = multipletests(blp_p_vals, alpha=0.05, method='fdr_bh')
    for i, record in enumerate(passed):
        # Corrected p-value = (m/k) * p-value, where k is rank of p-value out of all records, and m is the total number of (passed) records
        # This means higher ranked p-values are scaled up more
        # EXCEPT if rank 3's corrected p-value is lower than rank 2's corrected p-value, then rank 2's corrected p-value is replaced with rank 3's corrected p-value, and if rank 3 counts as significant, so does rank 2
        record['blp_pval_bh'] = float(corrected_p_vals[i])
        record['blp_significant'] = bool(rejected[i])

rows = []
for record in passed + skipped:
    blp_res = record.get('blp_res', {})
    uplift_res = record.get('uplift_res', {})
    cal_r_squared = record.get('cal_r_squared')
    blp_significant = record.get('blp_significant', False)
    is_hit = bool(cal_r_squared is not None and cal_r_squared > 0 and blp_significant)
    rows.append({
        'key': record['key'],
        'display_name': record['display_name'],
        'passed_overlap': record['passed_overlap'],
        'total': record['total'],
        'compar_count': record['compar_count'],
        'arm_count': record['arm_count'],
        'minority_arm_n': record['minority_arm_n'],
        'cal_r_squared': cal_r_squared,
        'blp_est': blp_res.get('blp_est'),
        'blp_se': blp_res.get('blp_se'),
        'blp_pval': blp_res.get('blp_pval'),
        'blp_pval_bh': record.get('blp_pval_bh'),
        'blp_significant': blp_significant,
        'qini_est': uplift_res.get('qini_est'),
        'qini_se': uplift_res.get('qini_se'),
        'qini_pval': uplift_res.get('qini_pval'),
        'autoc_est': uplift_res.get('autoc_est'),
        'autoc_se': uplift_res.get('autoc_se'),
        'autoc_pval': uplift_res.get('autoc_pval'),
        'is_hit': is_hit,
    })

leaderboard = pd.DataFrame(rows)
# Hits first, then most-significant (lowest BH p) first; skips (NaN BH p) sink to the bottom
leaderboard = leaderboard.sort_values(
    by=['is_hit', 'blp_pval_bh'],
    ascending=[False, True],
    na_position='last',
).reset_index(drop=True)
leaderboard.to_csv(results_path / 'leaderboard.csv', index=False)
print(f"Wrote leaderboard with {len(leaderboard)} contrast(s) "
      f"({int(leaderboard['is_hit'].sum()) if len(leaderboard) else 0} hit(s)) to {results_path / 'leaderboard.csv'}")