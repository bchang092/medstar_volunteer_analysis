# control.py
from pathlib import Path
import logging
import pandas as pd

from ga_pipeline import processing, report
from ga_pipeline.notebooklm_packager import (
    notebooklm_pdf_workflow,
    load_summary_library,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

# ----------------------------
# User-configurable switches
# ----------------------------
PROCESS_DATA        = True
RUN_NOTEBOOKLM_FLOW = True      # Create copy/paste pack, pause, ingest NotebookLM outputs
BUILD_GRAPHS        = True
ASSEMBLE_PDF        = True

OUTPUT_FOLDER_NAME = "2025-12-31"
excel_name        = "01062026_Rounding_Surveys.xlsx"
OUTPUT_PDF_NAME   = "01062026_results_qual_v3.pdf"

# NotebookLM files (you control these names)
NOTEBOOKLM_PACK_PDF_NAME    = "NotebookLM_CopyPaste_Pack.pdf"
NOTEBOOKLM_OUTPUT_FILENAME  = "notebooklm_outputs.csv"  # you will place this into OUTPUT_DIR

# Analysis windows
# MONTH_OF_ANALYSIS: the single month to highlight in “Monthly” charts (YYYY-MM)
# QUARTER_START / QUARTER_END: bounds for the quarter (date strings) used to filter data and drive quarter heatmaps
MONTH_OF_ANALYSIS = "2025-12"  # e.g., "2025-10"
QUARTER_START     = "2025-06-01"
QUARTER_END       = "2025-12-31"


# Keep legacy names for processing/report compatibility
DATE_START = QUARTER_START
DATE_END   = QUARTER_END

# ----------------------------
# Inputs / Outputs
# ----------------------------
DATA_PATH   = Path("data") / excel_name
OUTPUT_DIR  = Path("outputs") / (DATE_END or "latest")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[paths] Writing outputs to: {OUTPUT_DIR}")


def compute_review_stats(df, months, date_col="date", pos_col="pos_exp", neg_col="neg_exp"):
    def _nonempty(x):
        return (x.notna()) & (x.astype(str).str.strip() != "")

    total_pos = int(_nonempty(df[pos_col]).sum()) if (df is not None and pos_col in df.columns) else 0
    total_neg = int(_nonempty(df[neg_col]).sum()) if (df is not None and neg_col in df.columns) else 0
    total_all = total_pos + total_neg

    latest_month = months[-1] if months else None
    latest_pos = latest_neg = latest_all = 0

    if df is not None and latest_month and date_col in df.columns:
        latest_period = pd.Period(latest_month, freq="M")
        m = df[date_col].dt.to_period("M") == latest_period
        if pos_col in df.columns:
            latest_pos = int((_nonempty(df[pos_col]) & m).sum())
        if neg_col in df.columns:
            latest_neg = int((_nonempty(df[neg_col]) & m).sum())
        latest_all = latest_pos + latest_neg

    return {
        "total_positive_reviews": total_pos,
        "total_negative_reviews": total_neg,
        "total_reviews": total_all,
        "latest_month": latest_month or "",
        "latest_month_positive_reviews": latest_pos,
        "latest_month_negative_reviews": latest_neg,
        "latest_month_total_reviews": latest_all,
    }


def _themes_df_from_library(lib: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_pos, rows_neg = [], []

    for dept, payload in (lib or {}).items():
        pos = str(payload.get("positive_summary", "")).strip()
        neg = str(payload.get("negative_summary", "")).strip()
        if pos:
            rows_pos.append({"department": dept, "summary": pos, "method": "notebooklm"})
        if neg:
            rows_neg.append({"department": dept, "summary": neg, "method": "notebooklm"})

    pos_df = pd.DataFrame(rows_pos, columns=["department", "summary", "method"])
    neg_df = pd.DataFrame(rows_neg, columns=["department", "summary", "method"])
    return pos_df, neg_df


def _write_notebooklm_status(output_dir: Path, ok: bool, reason: str = "") -> None:
    p = output_dir / "notebooklm_STATUS.txt"
    if ok:
        txt = "SUCCESS: NotebookLM summaries were loaded successfully.\n"
    else:
        txt = (
            "FAILURE: NotebookLM summaries were NOT loaded.\n"
            f"Reason: {reason}\n"
        )
    try:
        p.write_text(txt)
    except Exception:
        pass


def main():
    df = None
    artifacts = {}

    # ---------------- [1/3] PROCESSING ----------------
    if PROCESS_DATA:
        print("[1/3] Processing data...")
        df, artifacts = processing.run_processing(
            excel_path=DATA_PATH,
            date_start=DATE_START,
            date_end=DATE_END,
        )

    artifacts = artifacts or {}

    # gather stats
    months = artifacts.get("months", [])
    stats = compute_review_stats(df, months)

    logging.getLogger("control").info(
        "Review stats: total=%d (pos=%d, neg=%d); latest_month=%s total=%d (pos=%d, neg=%d)",
        stats["total_reviews"],
        stats["total_positive_reviews"],
        stats["total_negative_reviews"],
        stats["latest_month"],
        stats["latest_month_total_reviews"],
        stats["latest_month_positive_reviews"],
        stats["latest_month_negative_reviews"],
    )

    stats_path = OUTPUT_DIR / "review_stats.csv"
    pd.DataFrame([stats]).to_csv(stats_path, index=False)
    print(f"[stats] Wrote {stats_path}")

    # Will hold the summaries used in the final report
    pos_theme_df = pd.DataFrame(columns=["department", "summary", "method"])
    neg_theme_df = pd.DataFrame(columns=["department", "summary", "method"])

    # ---------------- [2/3] NOTEBOOKLM ONLY ----------------
    notebooklm_used = False

    if RUN_NOTEBOOKLM_FLOW:
        print("[2/3] NotebookLM copy/paste workflow (only path)...")

        notebooklm_pdf_workflow(
            pos_texts_by_dept=artifacts.get("pos_texts_by_dept", {}),
            neg_texts_by_dept=artifacts.get("neg_texts_by_dept", {}),
            output_dir=OUTPUT_DIR,
            output_pdf_name=NOTEBOOKLM_PACK_PDF_NAME,
            notebooklm_output_filename=NOTEBOOKLM_OUTPUT_FILENAME,
            date_start=DATE_START,
            date_end=DATE_END,
            fallback_pos_summaries=None,
            fallback_neg_summaries=None,
        )

        lib = load_summary_library(OUTPUT_DIR)
        npos, nneg = _themes_df_from_library(lib)

        if len(npos) or len(nneg):
            pos_theme_df, neg_theme_df = npos, nneg
            notebooklm_used = True
            msg = f"Using NotebookLM summaries (pos={len(npos)}, neg={len(nneg)})."
            logging.getLogger("control").info(msg)
            print("[NotebookLM]", msg)
            _write_notebooklm_status(OUTPUT_DIR, ok=True)
        else:
            reason = (
                "Summary library was empty or invalid. "
                f"Expected to find '{NOTEBOOKLM_OUTPUT_FILENAME}' in {OUTPUT_DIR} "
                "and for it to contain department summaries."
            )
            logging.getLogger("control").warning(reason)
            print("[NotebookLM]", reason)
            _write_notebooklm_status(OUTPUT_DIR, ok=False, reason=reason)

    # ---------------- [3/3] BUILD FINAL REPORT PDF ----------------
    if ASSEMBLE_PDF and BUILD_GRAPHS:
        print("[3/3] Building PDF report...")

        report.build_pdf(
            output_pdf_path=OUTPUT_DIR / OUTPUT_PDF_NAME,
            department_ind_dict=artifacts.get("department_ind_dict", {}),
            monthly_data=artifacts.get("monthly_data", {}),
            dep_service=artifacts.get("dep_service", {}),
            dep_wait=artifacts.get("dep_wait", {}),
            months=artifacts.get("months", []),

            # summaries (NotebookLM only; may be empty if missing)
            pos_theme_df=pos_theme_df,
            neg_theme_df=neg_theme_df,

            date_start=DATE_START,
            date_end=DATE_END,
            month_of_analysis=MONTH_OF_ANALYSIS,
        )

    print("Done!")


if __name__ == "__main__":
    main()
