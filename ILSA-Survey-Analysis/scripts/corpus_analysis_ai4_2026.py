"""
Corpus analysis script for AI4 2026 conference paper.
Generates 5 analyses with matplotlib figures + LaTeX tables.
"""

import json
import glob
import re
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

JSON_DIR = "/Users/mrved/Desktop/ILSA-Survey-Analysis/ilsa_survey_articles/json/"
OUT_DIR = "/Users/mrved/Desktop/ILSA-Survey-Analysis/outputs/corpus_analysis/"
os.makedirs(OUT_DIR, exist_ok=True)

ILSA_PROGRAMS = ["PISA", "TIMSS", "PIRLS", "PIAAC", "TALIS", "ICILS", "ICCS", "SACMEQ", "EGRA", "LLECE"]

STYLE = {
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Calibri", "DejaVu Sans"],
}
plt.rcParams.update(STYLE)

# ── helpers ────────────────────────────────────────────────────────────────────

def load_all():
    records = []
    for path in sorted(glob.glob(JSON_DIR + "*.json")):
        with open(path, encoding="utf-8") as fh:
            try:
                d = json.load(fh)
                records.append(d)
            except json.JSONDecodeError:
                pass
    return records


def detect_programs(record):
    """Return list of ILSA programs mentioned in a record."""
    text_fields = [
        record["metadata"].get("title", ""),
        record["metadata"].get("venue", ""),
    ]
    for finding in record["data"].get("main_findings", []):
        text_fields.append(finding.get("dataset_used", ""))
    combined = " ".join(f or "" for f in text_fields)
    found = [p for p in ILSA_PROGRAMS if p in combined]
    return list(set(found)) if found else ["Other"]


def normalise_ml(raw):
    """Normalise ML technique label."""
    r = raw.strip()
    mapping = {
        r"random.?forest": "Random Forest",
        r"gradient.?boost": "Gradient Boosting",
        r"xgb": "XGBoost",
        r"neural.?net|ann|mlp|deep.?learn": "Neural Network / Deep Learning",
        r"svm|support.?vector": "SVM",
        r"lasso": "LASSO",
        r"ridge": "Ridge Regression",
        r"logistic": "Logistic Regression",
        r"decision.?tree": "Decision Tree",
        r"k-?nn|k.nearest": "k-NN",
        r"naive.?bayes": "Naive Bayes",
        r"cluster": "Clustering",
        r"pca|principal.?component": "PCA",
        r"factor.?analy": "Factor Analysis",
        r"structural.?equat|sem\b": "SEM",
        r"latent.?class|lca": "Latent Class Analysis",
        r"irt\b|item.?response": "IRT",
        r"bert|transformer|llm|gpt|language.?model": "LLM / Transformer",
        r"anfis": "ANFIS",
        r"bagging|ensemble": "Ensemble Methods",
    }
    for pattern, label in mapping.items():
        if re.search(pattern, r, re.IGNORECASE):
            return label
    return r.title() if r else None


def normalise_outcome(raw):
    """Bucket outcome variable into broad category."""
    r = (raw or "").lower()
    if re.search(r"read", r):
        return "Reading Literacy"
    if re.search(r"math|mathemat|numeracy", r):
        return "Mathematics Achievement"
    if re.search(r"scien", r):
        return "Science Achievement"
    if re.search(r"writ", r):
        return "Writing / Composition"
    if re.search(r"digital|ict|computer|technolog", r):
        return "Digital / ICT Literacy"
    if re.search(r"civic|citizenship", r):
        return "Civic Knowledge"
    if re.search(r"problem.?solv|process.?data|click", r):
        return "Problem-Solving / Process Data"
    if re.search(r"socioeconom|ses|escs", r):
        return "SES / Background"
    if re.search(r"wellbeing|motivat|anxi|attitud|engagement|non-cogn", r):
        return "Well-being / Motivation"
    if re.search(r"teacher|school|principal|instruct", r):
        return "Teacher / School Variables"
    if re.search(r"overall|academic|achievement|performance|score|proficien", r):
        return "General Academic Achievement"
    return "Other"


def normalise_confounder(raw):
    """Bucket confounder into broad category."""
    r = (raw or "").lower()
    if re.search(r"ses|socioeconom|escs|income|wealth|poverty|econom", r):
        return "SES / Economic"
    if re.search(r"gender|sex\b", r):
        return "Gender"
    if re.search(r"immigr|migrant|native|foreign|ethnic|race|minorit", r):
        return "Immigration / Ethnicity"
    if re.search(r"language|home.?lang|bilingual", r):
        return "Language Background"
    if re.search(r"parent|family|home|mother|father", r):
        return "Family / Parental"
    if re.search(r"school|class|grade|urban|rural|region|location|type", r):
        return "School / Region"
    if re.search(r"teacher|instruction|curricul|teach", r):
        return "Teacher / Instruction"
    if re.search(r"motivat|attitud|self.?effica|belief|wellbeing|anxi", r):
        return "Motivation / Well-being"
    if re.search(r"prior|pretest|baseline|abilit|cognitive", r):
        return "Prior Achievement / Ability"
    if re.search(r"ict|technolog|digital|computer|internet", r):
        return "ICT / Technology"
    if re.search(r"book|read.?habit|read.?enjoy|leisure", r):
        return "Reading Habits"
    return "Other"


# ── latex helpers ──────────────────────────────────────────────────────────────

def latex_table(rows, header, caption, label, col_fmt=None):
    ncols = len(header)
    if col_fmt is None:
        col_fmt = "l" + "r" * (ncols - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{col_fmt}}}",
        r"    \toprule",
        "    " + " & ".join(header) + r" \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append("    " + " & ".join(str(c) for c in row) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def save_latex(content, fname):
    path = os.path.join(OUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"  LaTeX → {path}")


def save_fig(fig, fname):
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 1: Article distribution by ILSA program
# ══════════════════════════════════════════════════════════════════════════════

def analysis_1(records):
    print("\n[1] ILSA program distribution")
    counter = Counter()
    for r in records:
        for prog in detect_programs(r):
            counter[prog] += 1

    programs = [k for k, _ in counter.most_common()]
    counts = [counter[k] for k in programs]
    total = sum(counts)
    pcts = [100 * c / total for c in counts]

    # Figure
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.tab10.colors
    bars = ax.barh(programs[::-1], counts[::-1], color=[colors[i % 10] for i in range(len(programs))])
    for bar, pct in zip(bars, pcts[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.0f}  ({pct:.1f}%)", va="center", fontsize=8)
    ax.set_xlabel("Number of Articles")
    ax.set_title("Article Distribution by ILSA Programme")
    ax.set_xlim(0, max(counts) * 1.25)
    fig.tight_layout()
    save_fig(fig, "fig1_ilsa_program_distribution.pdf")

    # LaTeX
    rows = [(p, c, f"{100*c/total:.1f}\\%") for p, c in zip(programs, counts)]
    rows.append(("\\textbf{Total}", f"\\textbf{{{total}}}", "\\textbf{100.0\\%}"))
    tex = latex_table(
        rows,
        header=["ILSA Programme", "Articles", "\\%"],
        caption="Distribution of articles by ILSA programme in the corpus.",
        label="tab:ilsa_programs",
    )
    save_latex(tex, "tab1_ilsa_program_distribution.tex")


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 2: Publication year trend (2020–2026)
# ══════════════════════════════════════════════════════════════════════════════

def analysis_2(records):
    print("\n[2] Publication year trend")
    year_counter = Counter()
    for r in records:
        y = r["metadata"].get("year")
        if y and 2020 <= int(y) <= 2026:
            year_counter[int(y)] += 1

    years = list(range(2020, 2027))
    counts = [year_counter.get(y, 0) for y in years]

    # Figure
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(years, counts, color="#4C72B0", width=0.6, edgecolor="white")
    ax.plot(years, counts, "o-", color="#DD8452", linewidth=1.8, markersize=5, zorder=5)
    for x, c in zip(years, counts):
        if c:
            ax.text(x, c + 0.15, str(c), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Publication Year")
    ax.set_ylabel("Number of Articles")
    ax.set_title("Publication Trend (2020–2026)")
    ax.set_xticks(years)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    save_fig(fig, "fig2_publication_trend.pdf")

    # LaTeX
    rows = [(y, c) for y, c in zip(years, counts)]
    tex = latex_table(
        rows,
        header=["Year", "Articles"],
        caption="Number of corpus articles published per year (2020–2026).",
        label="tab:pub_trend",
    )
    save_latex(tex, "tab2_publication_trend.tex")


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 3: ML method categories (top 10)
# ══════════════════════════════════════════════════════════════════════════════

def analysis_3(records):
    print("\n[3] ML method categories (top 10)")
    counter = Counter()
    for r in records:
        mt = r["data"].get("ml_techniques", {})
        techs = []
        if isinstance(mt, dict):
            techs = mt.get("all_techniques", []) or []
        elif isinstance(mt, list):
            techs = mt
        for t in techs:
            if t:
                norm = normalise_ml(str(t))
                if norm:
                    counter[norm] += 1

    top10 = counter.most_common(10)
    labels = [x[0] for x in top10]
    counts = [x[1] for x in top10]

    # Figure
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set2.colors
    bars = ax.barh(labels[::-1], counts[::-1],
                   color=[colors[i % len(colors)] for i in range(len(labels))],
                   edgecolor="white")
    for bar in bars:
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(int(bar.get_width())), va="center", fontsize=9)
    ax.set_xlabel("Frequency (articles)")
    ax.set_title("Top 10 ML Method Categories")
    ax.set_xlim(0, max(counts) * 1.2)
    fig.tight_layout()
    save_fig(fig, "fig3_ml_methods_top10.pdf")

    # LaTeX
    total = sum(counter.values())
    rows = [(i + 1, lbl, cnt, f"{100*cnt/total:.1f}\\%")
            for i, (lbl, cnt) in enumerate(top10)]
    tex = latex_table(
        rows,
        header=["Rank", "ML Method", "Count", "\\%"],
        caption="Top 10 machine learning method categories identified in the corpus.",
        label="tab:ml_methods",
        col_fmt="clrr",
    )
    save_latex(tex, "tab3_ml_methods_top10.tex")


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 4: Top 10 confounder variables
# ══════════════════════════════════════════════════════════════════════════════

def analysis_4(records):
    print("\n[4] Top 10 confounder variables")
    counter = Counter()
    for r in records:
        confounders = r["data"].get("confounders_identified", []) or []
        for cf in confounders:
            if isinstance(cf, dict):
                name = cf.get("variable_name", "")
            else:
                name = str(cf)
            norm = normalise_confounder(name)
            counter[norm] += 1

    top10 = counter.most_common(10)
    labels = [x[0] for x in top10]
    counts = [x[1] for x in top10]

    # Figure
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Pastel1.colors
    bars = ax.barh(labels[::-1], counts[::-1],
                   color=[colors[i % len(colors)] for i in range(len(labels))],
                   edgecolor="grey", linewidth=0.5)
    for bar in bars:
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                str(int(bar.get_width())), va="center", fontsize=9)
    ax.set_xlabel("Frequency (mentions)")
    ax.set_title("Top 10 Confounder Variable Categories")
    ax.set_xlim(0, max(counts) * 1.2)
    fig.tight_layout()
    save_fig(fig, "fig4_confounders_top10.pdf")

    # LaTeX
    total = sum(counter.values())
    rows = [(i + 1, lbl, cnt, f"{100*cnt/total:.1f}\\%")
            for i, (lbl, cnt) in enumerate(top10)]
    tex = latex_table(
        rows,
        header=["Rank", "Confounder Category", "Mentions", "\\%"],
        caption="Top 10 confounder variable categories reported in the corpus.",
        label="tab:confounders",
        col_fmt="clrr",
    )
    save_latex(tex, "tab4_confounders_top10.tex")


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 5: Outcome variable distribution
# ══════════════════════════════════════════════════════════════════════════════

def analysis_5(records):
    print("\n[5] Outcome variable distribution")
    counter = Counter()
    for r in records:
        findings = r["data"].get("main_findings", []) or []
        seen = set()
        for finding in findings:
            tv = finding.get("target_variable", "")
            norm = normalise_outcome(tv)
            if norm not in seen:
                counter[norm] += 1
                seen.add(norm)

    sorted_items = counter.most_common()
    labels = [x[0] for x in sorted_items]
    counts = [x[1] for x in sorted_items]
    total = sum(counts)

    # Pie chart
    fig, ax = plt.subplots(figsize=(7, 6))
    explode = [0.03] * len(labels)
    wedges, texts, autotexts = ax.pie(
        counts, labels=None, autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        startangle=140, explode=explode,
        colors=plt.cm.tab20.colors[:len(labels)],
        pctdistance=0.82,
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.legend(wedges, [f"{l} ({c})" for l, c in zip(labels, counts)],
              loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
    ax.set_title("Outcome Variable Distribution")
    fig.tight_layout()
    save_fig(fig, "fig5_outcome_distribution.pdf")

    # Also bar chart for clarity
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab20.colors
    bars = ax2.barh(labels[::-1], counts[::-1],
                    color=[colors[i % 20] for i in range(len(labels))],
                    edgecolor="white")
    for bar in bars:
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 str(int(bar.get_width())), va="center", fontsize=9)
    ax2.set_xlabel("Number of Articles")
    ax2.set_title("Outcome Variable Distribution (Bar)")
    ax2.set_xlim(0, max(counts) * 1.2)
    fig2.tight_layout()
    save_fig(fig2, "fig5b_outcome_distribution_bar.pdf")

    # LaTeX
    rows = [(i + 1, lbl, cnt, f"{100*cnt/total:.1f}\\%")
            for i, (lbl, cnt) in enumerate(sorted_items)]
    tex = latex_table(
        rows,
        header=["Rank", "Outcome Category", "Articles", "\\%"],
        caption="Distribution of outcome (target) variable categories across the corpus.",
        label="tab:outcomes",
        col_fmt="clrr",
    )
    save_latex(tex, "tab5_outcome_distribution.tex")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"Loading JSON records from {JSON_DIR}")
    records = load_all()
    print(f"  {len(records)} records loaded.")

    analysis_1(records)
    analysis_2(records)
    analysis_3(records)
    analysis_4(records)
    analysis_5(records)

    print(f"\nAll outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
