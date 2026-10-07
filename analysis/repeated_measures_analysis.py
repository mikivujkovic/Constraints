# -*- coding: utf-8 -*-
"""
Repeated-measures and sensitivity analyses for
"Constraints are all you need: Exploring AI-powered landing page design."

Addresses Reviewer 4 (pseudo-replication, small stimulus set). Reproduces the
numbers reported in the Results subsection "Repeated-measures and sensitivity
analyses" and in S10 Table.

Input : the anonymized human evaluation dataset (S8), one row per rater x page,
        with columns: rater, role, image_id, company, generation_type, and the
        20 criterion scores. (Use S8_Table_anonymized.csv -- never the raw file
        that contains email addresses.)
Usage : python repeated_measures_analysis.py S8_Table_anonymized.csv
Deps  : pandas, numpy, scipy, statsmodels
"""

import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

CSV = sys.argv[1] if len(sys.argv) > 1 else "data/Human_evaluations.csv"
df = pd.read_csv(CSV)
RATER = "rater" if "rater" in df.columns else "email"  # works with either label

CATS = {
    "Visual Design": [
        "color_harmony_contrast",
        "typography_readability",
        "spatial_layout_balance",
        "visual_hierarchy",
        "design_consistency",
    ],
    "Content Quality": [
        "message_clarity",
        "brand_voice_consistency",
        "cta_effectiveness",
        "value_proposition",
        "content_structure",
    ],
    "User Experience": [
        "navigation_intuitiveness",
        "information_architecture",
        "mobile_responsiveness",
        "load_time_performance",
        "interaction_design",
    ],
    "Business Effectiveness": [
        "conversion_potential",
        "brand_alignment",
        "target_audience",
        "competitive_differentiation",
        "market_readiness",
    ],
}
for c, cols in CATS.items():
    df[c] = df[cols].mean(axis=1)
df["Overall"] = df[list(CATS)].mean(axis=1)


def cliffs(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    U, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    return 2 * U / (len(a) * len(b)) - 1


print("=" * 70)
print("A. RATER-LEVEL AGGREGATION (Friedman + Wilcoxon signed-rank, Bonferroni)")
print("   one mean per rater per approach; n =", df[RATER].nunique(), "raters")
rl = (
    df.groupby([RATER, "generation_type"])[["Overall"] + list(CATS)]
    .mean()
    .reset_index()
)
for metric in ["Overall"] + list(CATS):
    piv = rl.pivot(index=RATER, columns="generation_type", values=metric)
    fr = stats.friedmanchisquare(piv["netai"], piv["prompt_eng"], piv["naive"])
    wcu = min(stats.wilcoxon(piv["netai"], piv["naive"]).pvalue * 3, 1)
    wcp = min(stats.wilcoxon(piv["netai"], piv["prompt_eng"]).pvalue * 3, 1)
    wpu = min(stats.wilcoxon(piv["prompt_eng"], piv["naive"]).pvalue * 3, 1)
    d = cliffs(piv["netai"], piv["naive"])
    print(
        f"  {metric:24s} chi2={fr.statistic:6.2f} p={fr.pvalue:.2e} | "
        f"CvU p={wcu:.1e} CvP p={wcp:.1e} PvU p={wpu:.1e} | cliff CvU={d:+.3f}"
    )

print("=" * 70)
print("B. LINEAR MIXED-EFFECTS MODEL (random intercept for rater), DV = Overall")
d2 = df.copy()
d2["gen"] = pd.Categorical(
    d2["generation_type"], categories=["naive", "prompt_eng", "netai"]
)
m = smf.mixedlm(
    "Overall ~ C(gen, Treatment(reference='naive'))", d2, groups=d2[RATER]
).fit(reml=True)
for nm in m.params.index:
    if nm.startswith("C(gen"):
        print(
            f"  {nm:52s} beta={m.params[nm]:+.3f} SE={m.bse[nm]:.3f} p={m.pvalues[nm]:.2e}"
        )
rv = float(np.ravel(m.cov_re.values)[0])
res = m.scale
print(f"  rater var={rv:.4f} residual={res:.4f} ICC={rv / (rv + res):.3f}")

print("=" * 70)
print("C. SENSITIVITY (small stimulus set)")
comp = df.company.unique()
for hold in comp:
    sub = df[df.company != hold].groupby("generation_type")["Overall"].mean()
    print(
        f"  leave-out {hold:22s} improvement CvU = {(sub['netai'] - sub['naive']) / sub['naive'] * 100:5.2f}%"
    )
full = df.groupby("generation_type")["Overall"].mean()
print(
    f"  full sample improvement CvU = {(full['netai'] - full['naive']) / full['naive'] * 100:.2f}%"
)
rng = np.random.default_rng(7)
boot = []
for _ in range(3000):
    cs = rng.choice(comp, len(comp), True)
    r = (
        pd.concat([df[df.company == c] for c in cs])
        .groupby("generation_type")["Overall"]
        .mean()
    )
    boot.append((r["netai"] - r["naive"]) / r["naive"] * 100)
print(
    f"  bootstrap-over-companies 95% CI improvement: [{np.percentile(boot, 2.5):.1f}%, {np.percentile(boot, 97.5):.1f}%]"
)
rng = np.random.default_rng(11)
raters = df[RATER].unique()
bd = []
for _ in range(2000):
    rs = rng.choice(raters, len(raters), True)
    p = pd.concat([df[df[RATER] == x] for x in rs])
    bd.append(
        cliffs(
            p[p.generation_type == "netai"]["Overall"],
            p[p.generation_type == "naive"]["Overall"],
        )
    )
print(
    f"  bootstrap-over-raters 95% CI cliff CvU: [{np.percentile(bd, 2.5):+.3f}, {np.percentile(bd, 97.5):+.3f}]"
)
