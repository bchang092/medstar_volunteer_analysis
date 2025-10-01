# ga_pipeline/report.py
from typing import Dict, Any, List, Optional
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from .graphs import page_for_department

def build_pdf(
    output_pdf_path,
    department_ind_dict: Dict[str, List[int]],
    monthly_data: Dict[str, Any],
    dep_service: Dict[str, Any],
    dep_wait: Dict[str, Any],
    months: List[str],
    pos_theme_df=None,
    neg_theme_df=None,
):
    with PdfPages(output_pdf_path) as pdf:
        # one page per department with the 2x2 dashboard
        for dept in department_ind_dict.keys():
            page_for_department(pdf, dept, months, monthly_data, dep_service, dep_wait)

        # (Optional) add simple top-themes pages if available
        if pos_theme_df is not None and not pos_theme_df.empty:
            for dept in sorted(pos_theme_df["department"].unique()):
                sub = pos_theme_df[pos_theme_df["department"] == dept]
                # try to find simple columns from BERTopic's topic_info
                # (Name, Count) are common; fall back gracefully
                name_col = "Name" if "Name" in sub.columns else None
                count_col = "Count" if "Count" in sub.columns else None
                if name_col and count_col:
                    top = sub.nlargest(10, count_col)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(top[name_col][::-1], top[count_col][::-1])
                    ax.set_title(f"{dept} — Top Positive Themes")
                    ax.set_xlabel("Count")
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

        if neg_theme_df is not None and not neg_theme_df.empty:
            for dept in sorted(neg_theme_df["department"].unique()):
                sub = neg_theme_df[neg_theme_df["department"] == dept]
                name_col = "Name" if "Name" in sub.columns else None
                count_col = "Count" if "Count" in sub.columns else None
                if name_col and count_col:
                    top = sub.nlargest(10, count_col)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(top[name_col][::-1], top[count_col][::-1])
                    ax.set_title(f"{dept} — Top Negative Themes")
                    ax.set_xlabel("Count")
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

    print(f"Saved PDF report → {output_pdf_path}")
