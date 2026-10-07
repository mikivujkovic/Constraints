# Constraints are all you need — reproducibility materials

Data, analysis scripts, prompts, and generated landing-page code for the study:

> **Constraints are all you need: Exploring AI-powered landing page design**
> Vujković M, Popović T, Jovović I, Drakić-Grgur M. *PLOS ONE* (under review). Manuscript PONE-D-25-55869.

The netAI system itself (the constrained generation pipeline) lives in a separate repository:
**https://github.com/mikivujkovic/netAI**

This repository contains everything needed to reproduce the evaluation and statistics reported in the paper, plus the prompts and generated pages used as stimuli.

---

## Data

Two fully anonymized datasets, one row per (rater/model × landing page).

| Column | Description |
|---|---|
| `rater` | anonymous participant code (R01–R65); `llm_evaluations.csv` uses the model name instead |
| `role` | self-reported professional role: developer, designer, business, project_manager, other |
| `image_id` | landing-page identifier (`imgC_P`) |
| `company` | fictional company the page was generated for (Eco Harvest, Quantum Logistics, Sentinel Cybershield, Wellness Horizon) |
| `generation_type` | `naive` (unconstrained), `prompt_eng` (prompt-engineered), `netai` (constrained) |
| 20 score columns | criterion ratings on a 5-point Likert scale (1 = poor, 5 = excellent) |

The 20 criteria are grouped into four dimensions:

- **Visual Design** — color_harmony_contrast, typography_readability, spatial_layout_balance, visual_hierarchy, design_consistency
- **Content Quality** — message_clarity, brand_voice_consistency, cta_effectiveness, value_proposition, content_structure
- **User Experience** — navigation_intuitiveness, information_architecture, mobile_responsiveness, load_time_performance, interaction_design
- **Business Effectiveness** — conversion_potential, brand_alignment, target_audience, competitive_differentiation, market_readiness

**Privacy note.** These files contain no personal data: no names, emails, IP addresses, free-text, or timestamps. Participants are identified only by an arbitrary code and professional role.

---

## Reproducing the analyses

Requires Python 3.10+ and:

```bash
pip install pandas numpy scipy statsmodels
```

Run from the repository root:

```bash
# Primary human-evaluation statistics (Kruskal–Wallis, Dunn, Cliff's delta)  -> S5
python analysis/main.py

# Effect sizes with bootstrap confidence intervals
python analysis/stat_with_conf.py

# LLM evaluation analysis (rating inflation, compressed differentiation)      -> S6
python analysis/llm_analysis.py

# Human-vs-LLM correlation (Pearson r = 0.67, Spearman r = 0.60)              -> S7
python analysis/corr.py

# Repeated-measures + sensitivity analyses (Friedman, Wilcoxon, mixed model,
# leave-one-company-out, bootstrap)                                           -> S10
python analysis/repeated_measures_analysis.py data/Human_evaluations.csv

# Second-round additions: participant counts per role and a crossed
# rater + landing-page random-effects model                                   -> S10 (S10.3b), Methods
python analysis/r2_additional_analyses.py data/Human_evaluations.csv
```

The `repeated_measures_analysis.py` script reproduces the values reported in the
Results subsection *Repeated-measures and sensitivity analyses* and in S10 Table,
including:

- Friedman χ²(2) = 26.86, p = 1.5 × 10⁻⁶ (overall)
- Wilcoxon signed-rank (Bonferroni): constrained vs unconstrained p = 9.6 × 10⁻⁷
- Mixed-effects model: constrained β = +0.595 (p < 10⁻³⁰); ICC (rater) = 0.50
- Leave-one-company-out improvement range 13.4%–21.7% (full sample 16.8%)
- Bootstrap 95% CIs: improvement [7.9%, 24.8%]; Cliff's δ (CvU) [0.25, 0.47]

`r2_additional_analyses.py` reproduces the participant counts per professional role (15 developers, 9 designers, 6 business, 6 project managers, 29 other) and the crossed random-effects model (constrained β = +0.595, p < 10⁻⁴; landing-page variance 0.042 vs rater variance 0.343) reported in S10 Table.

---

## Generation and evaluation setup

- **Unconstrained / prompt-engineered:** generated through the GUI chat interfaces of chatgpt.com (GPT-4o) and claude.ai (Claude 3.5 Sonnet); prompts in `prompts/`.
- **Constrained (netAI):** generated through the OpenAI API using the multi-stage pipeline in the [netAI repository](https://github.com/mikivujkovic/netAI)
- **Generation date:** 17 February 2025. **LLM assessment date:** 18 March 2025.
- **LLM evaluators:** Claude 3.5 Sonnet, GPT-4o, Gemini 2.0, DeepSeek V3, each given a rendered screenshot of the page plus the standardized prompt in `prompts/llm_assessment.md` (vision-capable endpoints).

---

## License

Released for academic reproducibility. Recommended: code under the MIT License and
data/text under CC BY 4.0 (matching the PLOS ONE article license). Add `LICENSE`
files before making the repository public.

## Citation

```bibtex
@article{vujkovic_constraints,
  title   = {Constraints are all you need: Exploring AI-powered landing page design},
  author  = {Vujkovi\'c, Miodrag and Popovi\'c, Tomo and Jovovi\'c, Ivan and Draki\'c-Grgur, Maja},
  journal = {PLOS ONE},
  note    = {Manuscript PONE-D-25-55869, under review},
  year    = {2025}
}
```
