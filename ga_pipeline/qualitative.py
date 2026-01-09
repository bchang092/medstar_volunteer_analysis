# ga_pipeline/qualitative.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

import pandas as pd


def run_qualitative(
    pos_texts_by_dept: Dict[str, List[Any]],  # kept for compatibility (unused in NotebookLM-only mode)
    neg_texts_by_dept: Dict[str, List[Any]],  # kept for compatibility (unused in NotebookLM-only mode)
    output_dir: Path,
    *,
    notebooklm_csv: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    NotebookLM-only qualitative summaries.

    EXPECTS NotebookLM CSV in THIS WIDE FORMAT:
      columns: ['department', 'positive_summary', 'negative_summary']

    Behavior:
      - If notebooklm_csv is missing/unreadable/malformed:
          * writes output_dir / 'qualitative_ERROR.txt' explaining why
          * logs the same reason
          * returns empty pos/neg dfs with method='ERROR'
      - No TF-IDF fallback (disabled).
      - Writes:
          * output_dir / 'pos_rev_themes.csv'
          * output_dir / 'neg_rev_themes.csv'
    """
    log = logging.getLogger("qualitative")
    output_dir.mkdir(parents=True, exist_ok=True)
    error_log_path = output_dir / "qualitative_ERROR.txt"

    def fail(reason: str):
        msg = (
            "NotebookLM qualitative summaries were NOT generated.\n"
            f"Reason: {reason}\n"
            "TF-IDF fallback is DISABLED.\n"
        )
        log.error(msg)
        try:
            error_log_path.write_text(msg)
        except Exception:
            # if we can't write the error file, still return structured empties
            pass

        pos_df = pd.DataFrame([{"department": "(all)", "summary": "", "method": "ERROR"}],
                              columns=["department", "summary", "method"])
        neg_df = pd.DataFrame([{"department": "(all)", "summary": "", "method": "ERROR"}],
                              columns=["department", "summary", "method"])

        # still write the usual outputs so downstream report.py doesn't crash
        try:
            pos_df.to_csv(output_dir / "pos_rev_themes.csv", index=False)
            neg_df.to_csv(output_dir / "neg_rev_themes.csv", index=False)
        except Exception:
            pass

        return pos_df, neg_df, {
            "pos_summaries_by_dept": {},
            "neg_summaries_by_dept": {},
            "error": msg,
        }

    # ---- Validate CSV path ----
    if notebooklm_csv is None:
        return fail("notebooklm_csv was not provided to run_qualitative().")

    if not Path(notebooklm_csv).exists():
        return fail(f"NotebookLM CSV not found at: {notebooklm_csv}")

    # ---- Read CSV ----
    try:
        df = pd.read_csv(notebooklm_csv)
    except Exception as e:
        return fail(f"Failed to read NotebookLM CSV: {e!r}")

    # ---- Validate columns (match YOUR file format) ----
    required = {"department", "positive_summary", "negative_summary"}
    if not required.issubset(set(df.columns)):
        return fail(
            f"NotebookLM CSV is missing required columns. "
            f"Expected at least {sorted(required)} but found {list(df.columns)}."
        )

    # ---- Build outputs ----
    pos_rows: List[Dict[str, Any]] = []
    neg_rows: List[Dict[str, Any]] = []
    pos_summaries_by_dept: Dict[str, str] = {}
    neg_summaries_by_dept: Dict[str, str] = {}

    for _, row in df.iterrows():
        dept = str(row.get("department", "")).strip()
        if not dept:
            continue

        pos = row.get("positive_summary", None)
        neg = row.get("negative_summary", None)

        pos_txt = "" if (pos is None or (isinstance(pos, float) and pd.isna(pos))) else str(pos).strip()
        neg_txt = "" if (neg is None or (isinstance(neg, float) and pd.isna(neg))) else str(neg).strip()

        # Keep rows even if empty summaries? Usually better to keep only non-empty.
        if pos_txt:
            pos_rows.append({"department": dept, "summary": pos_txt, "method": "notebooklm"})
            pos_summaries_by_dept[dept] = pos_txt

        if neg_txt:
            neg_rows.append({"department": dept, "summary": neg_txt, "method": "notebooklm"})
            neg_summaries_by_dept[dept] = neg_txt

    pos_df = pd.DataFrame(pos_rows, columns=["department", "summary", "method"])
    neg_df = pd.DataFrame(neg_rows, columns=["department", "summary", "method"])

    # Write outputs expected by downstream pipeline
    pos_df.to_csv(output_dir / "pos_rev_themes.csv", index=False)
    neg_df.to_csv(output_dir / "neg_rev_themes.csv", index=False)

    # If everything worked, remove stale error file if it exists
    try:
        if error_log_path.exists():
            error_log_path.unlink()
    except Exception:
        pass

    log.info(
        "Loaded NotebookLM summaries: pos=%d departments, neg=%d departments",
        len(pos_df),
        len(neg_df),
    )

    return pos_df, neg_df, {
        "pos_summaries_by_dept": pos_summaries_by_dept,
        "neg_summaries_by_dept": neg_summaries_by_dept,
    }