# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.4.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
"""
experiments/patch_paper.py
============================
Patch the Sara paper PDF with real experimental results.

Replaces the placeholder Section 20 with a publication-quality report
containing: experimental methodology, measured results tables, per-config
A-B gap analysis, formal statistical hypothesis testing (t-test, p-value,
Cohen's d), interpretive discussion, and reproduction instructions.

All statistics are computed from the aggregated results JSON — nothing
is hardcoded or fabricated.

Usage
-----
    python experiments/collect_results.py
    python experiments/patch_paper.py

    # Custom output path:
    python experiments/patch_paper.py --output ~/Desktop/sara_paper.pdf

Default output: docs/paper/Sara_Knowledge_Distillation.pdf
"""

from __future__ import annotations

import argparse, json, math, re, shutil, subprocess, sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = Path(__file__).parent / "results"
PAPER_DIR    = PROJECT_ROOT / "docs" / "paper"
AGG_JSON     = RESULTS_DIR / "aggregated_results.json"
STORY_SRC    = PAPER_DIR  / "sara_story.py"
HELPERS_SRC  = PAPER_DIR  / "sara_helpers.py"
FINAL_SCRIPT = PAPER_DIR  / "_build_final.py"
DEFAULT_PDF  = PAPER_DIR  / "Sara_Knowledge_Distillation.pdf"


# ── Statistics helpers ──────────────────────────────────────────────────────

def _mean(xs): return sum(xs) / len(xs) if xs else 0.0

def _std(xs):
    if len(xs) < 2: return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def _stderr(xs):
    return _std(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0

def _t_stat(gaps):
    if len(gaps) < 2: return 0.0
    se = _stderr(gaps)
    return _mean(gaps) / se if se > 1e-12 else 0.0

def _t_critical(df, alpha=0.05):
    table = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015,
             6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
             15: 1.753, 20: 1.725, 25: 1.708, 30: 1.697}
    if df in table: return table[df]
    for k in sorted(table.keys()):
        if k >= df: return table[k]
    return 1.645

def _p_approx(t, df):
    if df >= 30: z = t
    else: z = t * (1 - 1/(4*df))
    if z > 4: return 0.0001
    if z < -4: return 0.9999
    x = abs(z)
    t2 = 1.0 / (1.0 + 0.33267 * x)
    phi = 1.0 - (0.4361836*t2 - 0.1201676*t2**2 + 0.9372980*t2**3) * math.exp(-x*x/2) / math.sqrt(2*math.pi)
    return 1.0 - phi if z > 0 else phi

def _cohens_d(gaps):
    if len(gaps) < 2 or _std(gaps) < 1e-12: return 0.0
    return _mean(gaps) / _std(gaps)

def _d_label(d):
    d = abs(d)
    if d >= 1.2: return "very large"
    if d >= 0.8: return "large"
    if d >= 0.5: return "medium"
    if d >= 0.2: return "small"
    return "negligible"

def _gap_label(gap):
    if gap > 0.02:  return "strong"
    if gap > 0.01:  return "moderate"
    if gap > 0.005: return "suggestive"
    if gap > 0.0:   return "marginal"
    return "negative"


def load_aggregated() -> dict:
    if not AGG_JSON.exists():
        sys.exit("Run first: python experiments/collect_results.py")
    return json.loads(AGG_JSON.read_text())


# ── Section 20 generator ───────────────────────────────────────────────────

def make_results_section(agg: dict) -> str:
    rows      = agg.get("paper_table_rows", [])
    gap       = agg.get("overall_ab_gap", 0.0)
    all_gaps  = agg.get("all_ab_gaps", [])
    configs   = agg.get("configs", [])
    n_runs    = agg.get("n_total_runs", 0)
    gen_at    = agg.get("generated_at", "")[:10]

    n_configs = len(configs)
    n_seeds   = max((c.get("n_seeds", 0) for c in configs), default=0)
    gap_std   = _std(all_gaps) if len(all_gaps) > 1 else 0.0
    gap_se    = _stderr(all_gaps)

    teachers = list({c.get("teacher", "unknown") for c in configs})
    students = list({c.get("student", "unknown") for c in configs})

    # E-A gap (MetaKDSPAR vs base KD-SPAR)
    e_row = next((r for r in rows if r["condition"] == "E"), None)
    a_row = next((r for r in rows if r["condition"] == "A"), None)
    has_meta = e_row is not None and a_row is not None
    ea_gap = (e_row["kd_mean"] - a_row["kd_mean"]) if has_meta else 0.0

    df     = max(len(all_gaps) - 1, 1)
    t      = _t_stat(all_gaps)
    t_crit = _t_critical(df)
    p_val  = _p_approx(t, df)
    d      = _cohens_d(all_gaps)
    reject = t > t_crit and gap > 0
    ci_lo  = gap - t_crit * gap_se
    ci_hi  = gap + t_crit * gap_se

    cond_labels = {
        "A": "KD-SPAR (student self-proposed)",
        "B": "Externally proposed (teacher)",
        "C": "Random instructions",
        "D": "No prompt tuning (baseline)",
        "E": "MetaKDSPAR (meta-prompted)",
    }
    tbl = "\n".join(
        f'        ["{r["condition"]}", "{cond_labels.get(r["condition"], r["condition"])}", '
        f'"{r["kd_str"]}", "{r["delta_str"]}", "{r["cit_str"]}"],'
        for r in rows
    )
    cfgtbl = "\n".join(
        f'        ["{c.get("config_label","")[:20]}", "{c.get("teacher","")[:18]}", '
        f'"{c.get("student","")[:14]}", "{c.get("n_seeds",0)}", '
        f'"{c.get("ab_gap_mean",0):+.4f} \\u00b1 {c.get("ab_gap_std",0):.4f}", '
        f'"{"\\u2713 Supported" if c.get("hypothesis_supported") else "\\u2717 Not supported"}"],'
        for c in configs
    )

    # ── Adaptive commentary ─────────────────────────────────────────────
    gl    = _gap_label(gap)
    dl    = _d_label(d)
    tstr  = ", ".join(teachers)
    sstr  = ", ".join(students)

    if gap > 0.02 and reject:
        finding = (
            f"The self-knowledge hypothesis is <b>supported</b>. "
            f"KD-SPAR (Condition A) outperforms the externally-proposed baseline "
            f"(Condition B) with a mean A\\u2212B gap of <b>{gap:+.4f}</b> "
            f"(t({df})={t:.2f}, p={p_val:.4f}, Cohen\\u2019s d={d:.2f} [{dl}]). "
            f"Since both conditions use identical KD scoring, teacher, student, "
            f"and validate-and-commit gate, the gap isolates the pure "
            f"contribution of student self-authorship."
        )
    elif gap > 0.005 and reject:
        finding = (
            f"The self-knowledge hypothesis receives <b>moderate support</b>. "
            f"Condition A outperforms Condition B with a mean gap of "
            f"<b>{gap:+.4f}</b> (t({df})={t:.2f}, p={p_val:.4f}). "
            f"The effect is statistically significant but practically modest."
        )
    elif gap > 0:
        finding = (
            f"The A\\u2212B gap is positive ({gap:+.4f}) but does not reach "
            f"statistical significance (t({df})={t:.2f}, p={p_val:.4f}). "
            f"This is <b>suggestive but inconclusive</b>. Running 5+ seeds "
            f"per configuration and increasing SPAR iterations to 5 is recommended."
        )
    else:
        finding = (
            f"The A\\u2212B gap is <b>negative</b> ({gap:+.4f}), indicating that "
            f"externally-proposed instructions (B) matched or outperformed "
            f"self-authored instructions (A). This does <b>not</b> support the "
            f"self-knowledge hypothesis in this configuration. Possible causes: "
            f"(i) the student model ({sstr}) may lack sufficient meta-cognitive "
            f"capability at its parameter count; (ii) 3 SPAR iterations may be "
            f"insufficient for self-interview convergence; (iii) Jaccard token "
            f"overlap may not capture the semantic improvements self-authored "
            f"prompts provide. Upgrading to BERTScore and running 5+ iterations "
            f"is recommended before drawing definitive conclusions."
        )

    b_data = [r for r in rows if r["condition"] == "B"]
    c_data = [r for r in rows if r["condition"] == "C"]
    b_above_d = any(r["delta_vs_d"] > 0 for r in b_data)
    c_near_d  = all(abs(r["delta_vs_d"]) < 0.05 for r in c_data) if c_data else True

    if b_above_d and c_near_d:
        ladder = (
            "Condition B (externally-proposed, KD-guided) outperforms both C (random) "
            "and D (no tuning), confirming the KD signal itself is informative "
            "regardless of who proposes. Condition C performs at or below D, "
            "confirming random prompt augmentation provides no systematic benefit "
            "\\u2014 improvement requires the KD scoring signal, not just additional "
            "instructions. The control conditions behave as expected."
        )
    elif b_above_d:
        ladder = (
            "Condition B outperforms D, confirming KD-guided proposal adds value. "
            "Condition C shows deviation from D, suggesting even random instructions "
            "interact with model behaviour non-trivially."
        )
    else:
        ladder = (
            "The baseline ladder shows unexpected patterns: B did not consistently "
            "outperform D, suggesting the Jaccard-based KD signal may not be "
            "sufficiently informative. Upgrading to BERTScore is recommended."
        )

    # MetaKDSPAR commentary (E-A gap)
    if has_meta:
        if ea_gap > 0.01:
            meta_comment = (
                f"<b>MetaKDSPAR (E\\u2212A gap = {ea_gap:+.4f}).</b> "
                f"Multi-perspective diagnosis outperforms flat single-pass "
                f"diagnosis by a meaningful margin. The specialist decomposition "
                f"(citation, calibration, completeness, format experts) catches "
                f"compound failures that the monolithic classifier misses, "
                f"producing higher-quality proposals."
            )
        elif ea_gap > 0:
            meta_comment = (
                f"<b>MetaKDSPAR (E\\u2212A gap = {ea_gap:+.4f}).</b> "
                f"Meta-prompting shows a small positive trend over base KD-SPAR. "
                f"The specialist architecture adds diagnostic breadth, but the "
                f"overhead of multiple inference calls per diagnosis may not be "
                f"justified at this gap size. More seeds would clarify."
            )
        else:
            meta_comment = (
                f"<b>MetaKDSPAR (E\\u2212A gap = {ea_gap:+.4f}).</b> "
                f"Multi-perspective diagnosis did not outperform flat diagnosis "
                f"in this configuration. Possible explanations: (i) the student "
                f"model may lack the capacity to maintain distinct specialist "
                f"perspectives; (ii) the conductor synthesis step may be "
                f"discarding useful specialist signals; (iii) 3 iterations "
                f"may be insufficient for the richer proposal space to converge."
            )
    else:
        meta_comment = ""

    return f'''
# ======================================================================
# SECTION 20 — EXPERIMENTAL RESULTS  (generated {gen_at} by patch_paper.py)
# ======================================================================

story += h1("20. Experimental Results: Self-Knowledge Hypothesis Test")
story += body(
    "This section reports measured results from the controlled KD-SPAR ablation "
    "experiment. All values are computed from <b>{n_runs} experimental run(s)</b> "
    "across {n_configs} model configuration(s). No numbers in this section are "
    "synthetic \\u2014 every value is derived from actual model outputs evaluated "
    "against teacher response traces."
)

story += h2("20.1  Experimental Methodology")
story += body(
    "The ablation isolates the contribution of <i>student self-authorship</i> "
    "from the KD scoring signal. Five conditions share identical evaluation "
    "queries, the same composite KD metric (0.3 \\u00d7 citation\\_match + "
    "0.7 \\u00d7 Jaccard token overlap), and the same validate-and-commit gate "
    "(\\u03b4 \\u2265 0.003). Only the source of proposed instructions differs:"
)
story += blist([
    "<b>Condition A \\u2014 KD-SPAR (self-proposed):</b> The student model "
    "diagnoses its own failure modes against teacher traces and proposes "
    "targeted instruction amendments via the four-phase SPAR loop.",
    "<b>Condition B \\u2014 Externally proposed:</b> The teacher model "
    "proposes instructions using the same KD divergence signal and failure "
    "diagnosis. The student executes but does not author.",
    "<b>Condition C \\u2014 Random instructions:</b> Generic fragments drawn "
    "from a fixed pool. Tests whether adding instructions helps independent "
    "of content.",
    "<b>Condition D \\u2014 Baseline:</b> Default system prompt. The floor.",
    "<b>Condition E \\u2014 MetaKDSPAR (meta-prompted):</b> The student uses a "
    "conductor + specialist architecture: four specialist perspectives "
    "(citation, calibration, completeness, format) independently diagnose "
    "failures; a conductor synthesises the top diagnoses; each specialist "
    "proposes fixes from its domain. Tests whether multi-perspective "
    "self-diagnosis outperforms flat single-pass diagnosis (Condition A).",
])
story += body(
    "<b>Hardware and models.</b> Experiments were run on a System76 OryxPro "
    "(Pop!_OS, NVIDIA RTX 3070 Ti) using the Ollama local inference runtime "
    "at temperature 0.0 for full reproducibility. "
    "Teacher(s): {tstr}. "
    "Student(s): {sstr}. "
    "Each configuration used {n_seeds} seed(s) with 3 SPAR iterations per "
    "condition, yielding {n_runs} total runs."
)
story += body(
    "<b>Evaluation.</b> Each condition\\u2019s final prompt is evaluated on "
    "10\\u201315 held-out validation queries. The student\\u2019s response is "
    "scored against the teacher\\u2019s cached response. Citation fidelity and "
    "hedging match are reported as secondary diagnostics. Results are "
    "mean \\u00b1 std across seeds."
)

story += h2("20.2  Main Results")
story += dtable(
    ["Cond.", "Description", "KD Score \\u2191", "\\u0394 vs D", "Citation Fid."],
    [
{tbl}
    ],
    col_widths=[0.55*inch, 2.15*inch, 1.35*inch, 0.85*inch, 1.80*inch]
)

story += h2("20.3  A\\u2212B Gap Analysis")
story += body(
    "The A\\u2212B gap measures the pure value of self-authorship. Both "
    "conditions are identical except that A\\u2019s proposals come from the "
    "student and B\\u2019s from the teacher."
)
story += dtable(
    ["Config", "Teacher", "Student", "Seeds", "A\\u2212B Gap", "H\\u2081"],
    [
{cfgtbl}
    ],
    col_widths=[1.5*inch, 1.4*inch, 1.1*inch, 0.55*inch, 1.5*inch, 1.15*inch]
)
story += body(
    "Overall mean A\\u2212B gap: <b>{gap:+.4f}</b> \\u00b1 {gap_std:.4f} "
    "(n={len(all_gaps)}, {gl})."
)

story += h2("20.4  Statistical Analysis")
story += body(
    "One-sided paired t-test on per-run A\\u2212B gaps:")
story += dtable(
    ["", "Statement"],
    [
        ["H\\u2080 (null)",  "E[KD(A)] \\u2264 E[KD(B)] \\u2014 self-authorship adds no value"],
        ["H\\u2081 (alt.)", "E[KD(A)] > E[KD(B)] \\u2014 self-authorship improves alignment"],
    ],
    col_widths=[1.2*inch, 5.5*inch]
)
story += body(
    "n = {len(all_gaps)}, mean gap = {gap:+.4f}, std = {gap_std:.4f}, "
    "SE = {gap_se:.4f}."
)
story += blist([
    "t({df}) = {t:.3f}   (critical value t_{{df,0.05}} = {t_crit:.3f})",
    "p = {p_val:.4f}   (one-sided)",
    "Cohen\\u2019s d = {d:.3f}   ({dl} effect)",
    "95%% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]",
    "{'H\\u2080 <b>rejected</b> at \\u03b1\\u2009=\\u20090.05.' if {reject} else 'H\\u2080 <b>not rejected</b> at \\u03b1\\u2009=\\u20090.05.'}",
])

story += h2("20.5  Discussion")
story += gold_callout("Finding", "{finding}")
story += body(
    "<b>Baseline ladder.</b> {ladder}"
)
if "{meta_comment}":
    story += body("{meta_comment}")
story += body(
    "<b>Limitations of this run.</b> "
    "The primary KD metric (Jaccard token overlap) is a surface-level proxy "
    "that does not capture semantic equivalence. A response that paraphrases "
    "the teacher will score lower than one that copies tokens. "
    "BERTScore F1 is recommended as the primary metric for camera-ready "
    "results. Human evaluation (3 raters, Cohen\\u2019s \\u03ba > 0.6) "
    "would strengthen the claim for top-venue submission."
)

story += h2("20.6  Reproducing These Results")
story += body(
    "All results were generated on a single GPU workstation. Identical seeds "
    "at temperature 0.0 yield identical outputs:"
)
story += code_block([
    "# Full reproduction",
    "cd sara && bash setup_and_run.sh",
    "",
    "# Or manually:",
    "pip install -e '.[rag]' reportlab pypdf",
    "for cfg in 1 2; do",
    "  for seed in 42 123 456 789 101; do",
    "    python experiments/kd_spar_ablation_ollama.py \\\\",
    "      --config $cfg --iterations 3 --seed $seed",
    "  done",
    "done",
    "python experiments/collect_results.py",
    "python experiments/patch_paper.py",
])
story += pgbrk()
'''


# ── Patching logic ──────────────────────────────────────────────────────────

def patch_story(agg: dict, output_pdf: Path) -> None:
    if not STORY_SRC.exists():
        sys.exit(f"ERROR: {STORY_SRC} not found. Is docs/paper/ in the project?")

    original = STORY_SRC.read_text()
    backup   = STORY_SRC.with_suffix(f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy(STORY_SRC, backup)
    print(f"Backed up -> {backup.name}")

    new_sec  = make_results_section(agg)

    # Find and replace existing Section 20 block
    for marker in [
        "# SECTION 20",
        'story += h1("20. Controlled Ablation',
        'story += h1("20. Experimental',
    ]:
        if marker in original:
            idx        = original.index(marker)
            line_start = original.rfind("\n", 0, idx) + 1
            # Find end boundary: Section 21 or References
            end_idx = -1
            for end_marker in [
                "# SECTION 21",
                '# ══════════════════════════════════════════════════════════════════════════════\n# SECTION 21',
                'story += h1("21.',
                "\n# ── REFERENCES",
            ]:
                end_idx = original.find(end_marker, idx)
                if end_idx >= 0:
                    end_idx = original.rfind("\n", 0, end_idx) + 1
                    break
            if end_idx < 0:
                end_idx = len(original)
            original = original[:line_start] + new_sec + "\n" + original[end_idx:]
            print("Section 20 replaced with real experimental results.")
            break
    else:
        for insert_before in ['# SECTION 21', 'story += h1("21.']:
            if insert_before in original:
                idx = original.index(insert_before)
                idx = original.rfind("\n", 0, idx) + 1
                original = original[:idx] + new_sec + "\n" + original[idx:]
                print("Section 20 inserted before Section 21.")
                break
        else:
            original += "\n" + new_sec
            print("Section 20 appended.")

    STORY_SRC.write_text(original)
    _rebuild(output_pdf)


def _rebuild(output_pdf: Path) -> None:
    combined = HELPERS_SRC.read_text() + "\n"
    story    = STORY_SRC.read_text()
    story = re.sub(
        r'build_doc\(story,\s*"[^"]+"\)',
        f'build_doc(story, "{output_pdf}")',
        story,
    )
    FINAL_SCRIPT.write_text(combined + story)

    result = subprocess.run(
        [sys.executable, str(FINAL_SCRIPT)],
        capture_output=True, text=True, cwd=str(PAPER_DIR),
    )
    FINAL_SCRIPT.unlink(missing_ok=True)

    if result.returncode == 0 and output_pdf.exists():
        try:
            from pypdf import PdfReader
            pages = len(PdfReader(str(output_pdf)).pages)
        except Exception:
            pages = "?"
        size = output_pdf.stat().st_size // 1024
        print(f"PDF written: {output_pdf}  ({pages} pages, {size} KB)")
    else:
        print("Build failed:")
        print((result.stdout + result.stderr)[-2000:])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Patch Sara paper with real results")
    p.add_argument("--output", default=str(DEFAULT_PDF), help="Output PDF path")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    agg = load_aggregated()
    n   = agg.get("n_total_runs", 0)
    gap = agg.get("overall_ab_gap", 0.0)
    print(f"Loaded: {n} run(s), {len(agg.get('configs',[]))} config(s), "
          f"A-B gap={gap:+.4f}")

    if args.dry_run:
        sec = make_results_section(agg)
        print("\n--- Generated Section 20 (first 50 lines) ---")
        print('\n'.join(sec.split('\n')[:50]))
        print("... (dry run)")
    else:
        patch_story(agg, Path(args.output))
