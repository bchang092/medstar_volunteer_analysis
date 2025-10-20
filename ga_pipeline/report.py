# ga_pipeline/report.py
from __future__ import annotations
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import textwrap
import numpy as np
import pandas as pd

from .graphs import page_for_department


# ---------------- helpers ----------------
def _wrap_text(s: str, width: int = 90) -> str:
    if not s:
        return ""
    return "\n".join(textwrap.fill(line, width=width) for line in s.splitlines())


def _add_langchain_theme_pages(
    pdf: PdfPages,
    df: pd.DataFrame,
    title_prefix: str,   # "Positive" or "Negative"
    max_width: int = 100,
):
    """
    Render one page per department with a human-friendly summary (LangChain map-reduce result).
    Expects df with columns at least: ['department', 'summary']
    """
    if df is None or df.empty or "department" not in df.columns or "summary" not in df.columns:
        return

    for dept in sorted(df["department"].dropna().unique()):
        sub = df[df["department"] == dept]
        # If multiple rows per dept, join summaries
        summary = "\n\n".join(str(x) for x in sub["summary"].fillna("").tolist()).strip()
        if not summary:
            summary = "(No summary text available.)"

        fig, ax = plt.subplots(figsize=(10.5, 8))  # landscape-ish single page
        ax.axis("off")

        header = f"{title_prefix} Themes — {dept}"
        ax.text(
            0.5, 0.96, header,
            ha="center", va="top",
            fontsize=20, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f1fb", alpha=0.9)
        )

        # body
        wrapped = _wrap_text(summary, width=max_width)
        ax.text(
            0.02, 0.90, "Summary",
            ha="left", va="top", fontsize=14, fontweight="bold"
        )
        ax.text(
            0.02, 0.86, wrapped,
            ha="left", va="top", fontsize=12
        )

        # optional: a light footer
        ax.text(0.98, 0.02, "LangChain map-reduce summary", ha="right", va="bottom", fontsize=9, alpha=0.6)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _add_bertopic_theme_pages(
    pdf: PdfPages,
    df: pd.DataFrame,
    title_prefix: str,   # "Positive" or "Negative"
    top_n: int = 10,
):
    """
    Render a bar chart of top-N topics per department for BERTopic outputs.
    Expects df with columns ['department', 'Name', 'Count'] at minimum.
    """
    if df is None or df.empty:
        return
    required = {"department", "Name", "Count"}
    if not required.issubset(set(df.columns)):
        return

    for dept in sorted(df["department"].dropna().unique()):
        sub = df[df["department"] == dept]
        if sub.empty:
            continue
        # remove the outlier topic -1 if present
        sub = sub[sub["Name"] != -1] if np.issubdtype(sub["Name"].dtype, np.number) else sub

        # ensure we have Count numeric
        try:
            sub["Count"] = pd.to_numeric(sub["Count"], errors="coerce")
        except Exception:
            pass
        sub = sub.dropna(subset=["Count"])

        top = sub.sort_values("Count", ascending=False).head(top_n)
        if top.empty:
            continue

        fig, ax = plt.subplots(figsize=(10.5, 8))
        labels = top["Name"].astype(str).tolist()
        counts = top["Count"].astype(float).tolist()

        ax.barh(labels[::-1], counts[::-1])
        ax.set_title(f"{title_prefix} Themes — {dept}", fontsize=18, pad=12)
        ax.set_xlabel("Count")
        ax.set_ylabel("Topic")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# ---------------- public API ----------------
def build_pdf(
    output_pdf_path,
    department_ind_dict: Dict[str, List[int]],
    monthly_data: Dict[str, Any],
    dep_service: Dict[str, Any],
    dep_wait: Dict[str, Any],
    months: List[str],
    pos_theme_df: Optional[pd.DataFrame] = None,
    neg_theme_df: Optional[pd.DataFrame] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
):
    """
    Build the PDF:
      1) One 2×2 dashboard page per department (from graphs.page_for_department)
      2) Extra 'themes' pages per department:
         - If DataFrame has ['department','summary'] → LangChain summary pages
         - Else if DataFrame has ['department','Name','Count'] → BERTopic bar charts
         - Else: skipped gracefully
    """

    with PdfPages(output_pdf_path) as pdf:
        # --- (1) Dashboard pages
        for dept in department_ind_dict.keys():
            page_for_department(
                pdf, dept, months, monthly_data, dep_service, dep_wait,
                date_start=date_start, date_end=date_end
            )

        # --- (2a) Positive theme pages
        if pos_theme_df is not None and not pos_theme_df.empty:
            if {"department", "summary"}.issubset(pos_theme_df.columns):
                _add_langchain_theme_pages(pdf, pos_theme_df, title_prefix="Positive")
            elif {"department", "Name", "Count"}.issubset(pos_theme_df.columns):
                _add_bertopic_theme_pages(pdf, pos_theme_df, title_prefix="Positive", top_n=10)

        # --- (2b) Negative theme pages
        if neg_theme_df is not None and not neg_theme_df.empty:
            if {"department", "summary"}.issubset(neg_theme_df.columns):
                _add_langchain_theme_pages(pdf, neg_theme_df, title_prefix="Negative")
            elif {"department", "Name", "Count"}.issubset(neg_theme_df.columns):
                _add_bertopic_theme_pages(pdf, neg_theme_df, title_prefix="Negative", top_n=10)

    print(f"Saved PDF report → {output_pdf_path}")
