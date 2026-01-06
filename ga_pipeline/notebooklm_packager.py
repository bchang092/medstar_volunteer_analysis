from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
import re

import pandas as pd

# ---- PDF (ReportLab) ----
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors


# -------------------------
# Helpers: normalize + extract
# -------------------------
def _as_flat_str_list(docs: List[Any]) -> List[str]:
    flat: List[str] = []
    for d in docs or []:
        if isinstance(d, list):
            flat.extend(d)
        else:
            flat.append(d)

    out: List[str] = []
    for s in flat:
        if s is None:
            continue
        s = str(s).strip()
        if s:
            out.append(s)
    return out


def _filter_and_extract_texts(
    docs: List[Any],
    date_start: Optional[str],
    date_end: Optional[str],
    keep_undated_when_filtering: bool = True,
) -> List[str]:
    """
    Supports:
      - list[str]
      - list[{'text': str, 'date': parseable}]
    Filters by [date_start, date_end] if provided.
    """
    if date_start is None and date_end is None:
        return _as_flat_str_list(docs)

    s = pd.to_datetime(date_start) if date_start else None
    e = pd.to_datetime(date_end) if date_end else None

    texts: List[str] = []
    for item in (docs or []):
        if item is None:
            continue

        if isinstance(item, dict):
            t = str(item.get("text", "")).strip()
            if not t:
                continue

            d_raw = item.get("date", None)
            if d_raw is not None:
                try:
                    d = pd.to_datetime(d_raw)
                except Exception:
                    if keep_undated_when_filtering:
                        texts.append(t)
                    continue

                if (s is not None and d < s) or (e is not None and d > e):
                    continue
                texts.append(t)
            else:
                if keep_undated_when_filtering:
                    texts.append(t)
        else:
            t = str(item).strip()
            if t and keep_undated_when_filtering:
                texts.append(t)

    return texts


def _safe_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w\-. ]+", "_", name)
    name = re.sub(r"\s+", " ", name)
    return name


# -------------------------
# NotebookLM import file parsing
# -------------------------
def _load_notebooklm_output_file(path: Path) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Accepts CSV or XLSX.
    Expected columns (minimum):
      - department
      - positive_summary
      - negative_summary

    Optional:
      - window_start, window_end

    Returns (df, issues)
    """
    issues: List[str] = []
    if not path.exists():
        return None, [f"NotebookLM output file not found: {path}"]

    try:
        if path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
    except Exception as e:
        return None, [f"Could not read file ({path.name}): {e}"]

    # normalize headers
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = {"department", "positive_summary", "negative_summary"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        issues.append(
            "Missing required columns: " + ", ".join(missing) +
            ". Expected at least: department, positive_summary, negative_summary"
        )
        return None, issues

    # basic cleanup
    df["department"] = df["department"].astype(str).str.strip()
    df["positive_summary"] = df["positive_summary"].fillna("").astype(str).str.strip()
    df["negative_summary"] = df["negative_summary"].fillna("").astype(str).str.strip()

    # remove blank departments
    before = len(df)
    df = df[df["department"] != ""].copy()
    if len(df) < before:
        issues.append(f"Dropped {before - len(df)} rows with blank department.")

    # warn if summaries are empty
    empty_pos = int((df["positive_summary"] == "").sum())
    empty_neg = int((df["negative_summary"] == "").sum())
    if empty_pos > 0:
        issues.append(f"{empty_pos} rows have empty positive_summary.")
    if empty_neg > 0:
        issues.append(f"{empty_neg} rows have empty negative_summary.")

    return df, issues


# -------------------------
# Summary “library” persistence
# -------------------------
def _library_path(output_dir: Path) -> Path:
    return output_dir / "notebooklm_summary_library.json"


def load_summary_library(output_dir: Path) -> dict:
    p = _library_path(output_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_summary_library(output_dir: Path, lib: dict) -> None:
    p = _library_path(output_dir)
    p.write_text(json.dumps(lib, indent=2, ensure_ascii=False), encoding="utf-8")


def upsert_library_from_df(output_dir: Path, df: pd.DataFrame) -> None:
    lib = load_summary_library(output_dir)
    for _, row in df.iterrows():
        dept = str(row["department"]).strip()
        lib.setdefault(dept, {})
        lib[dept]["positive_summary"] = str(row.get("positive_summary", "")).strip()
        lib[dept]["negative_summary"] = str(row.get("negative_summary", "")).strip()
        # optional metadata
        if "window_start" in df.columns:
            lib[dept]["window_start"] = str(row.get("window_start", "")).strip()
        if "window_end" in df.columns:
            lib[dept]["window_end"] = str(row.get("window_end", "")).strip()
    save_summary_library(output_dir, lib)


# -------------------------
# PDF generator
# -------------------------
def build_notebooklm_pack_pdf(
    *,
    pos_texts_by_dept: Dict[str, List[Any]],
    neg_texts_by_dept: Dict[str, List[Any]],
    output_pdf: Path,
    output_dir: Path,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    keep_undated_when_filtering: bool = True,
    include_summaries_from_library: bool = True,
    fallback_pos_summaries: Optional[Dict[str, str]] = None,
    fallback_neg_summaries: Optional[Dict[str, str]] = None,
    title: str = "NotebookLM Copy/Paste Pack",
) -> None:
    """
    Creates a PDF that’s easy to copy/paste into NotebookLM:
      - Each department gets a section
      - Positive reviews numbered
      - Negative reviews numbered
      - Optional: insert summaries (from library JSON if present; else fallback summaries)
    """
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()

    H1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=18,
        spaceAfter=10,
    )

    H2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=14,
        spaceBefore=8,
        spaceAfter=4,
    )

    BODY = ParagraphStyle(
        "BODY",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=13,
        spaceAfter=4,
    )

    MONO = ParagraphStyle(
        "MONO",
        parent=styles["BodyText"],
        fontName="Courier",
        fontSize=9.5,
        leading=12,
        spaceAfter=3,
    )

    SMALL = ParagraphStyle(
        "SMALL",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=11,
        textColor=colors.grey,
        spaceAfter=6,
    )

    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title=title,
        author="ga_pipeline",
    )

    story = []
    story.append(Paragraph(title, H1))
    window = f"Window: {date_start or '(none)'} → {date_end or '(none)'}"
    story.append(Paragraph(window, SMALL))
    story.append(Paragraph(
        "Tip: This document is formatted for easy copy/paste into NotebookLM. "
        "Each department section contains raw reviews and (optionally) summaries.",
        SMALL
    ))
    story.append(Spacer(1, 8))

    # load library if requested
    lib = load_summary_library(output_dir) if include_summaries_from_library else {}

    all_depts = sorted(set(list(pos_texts_by_dept.keys()) + list(neg_texts_by_dept.keys())))

    for di, dept in enumerate(all_depts):
        pos = _filter_and_extract_texts(
            pos_texts_by_dept.get(dept, []),
            date_start, date_end, keep_undated_when_filtering
        )
        neg = _filter_and_extract_texts(
            neg_texts_by_dept.get(dept, []),
            date_start, date_end, keep_undated_when_filtering
        )

        # page break between depts (except first)
        if di > 0:
            story.append(PageBreak())

        story.append(Paragraph(f"Department: {dept}", H1))
        story.append(Paragraph(f"Positive reviews: {len(pos)} &nbsp;&nbsp;|&nbsp;&nbsp; Negative reviews: {len(neg)}", SMALL))

        # summaries block (library preferred, fallback optional)
        pos_sum = ""
        neg_sum = ""
        if include_summaries_from_library and isinstance(lib, dict) and dept in lib:
            pos_sum = str(lib[dept].get("positive_summary", "")).strip()
            neg_sum = str(lib[dept].get("negative_summary", "")).strip()

        if (not pos_sum) and fallback_pos_summaries is not None:
            pos_sum = str(fallback_pos_summaries.get(dept, "")).strip()
        if (not neg_sum) and fallback_neg_summaries is not None:
            neg_sum = str(fallback_neg_summaries.get(dept, "")).strip()

        if pos_sum or neg_sum:
            story.append(Paragraph("Summaries", H2))
            if pos_sum:
                story.append(Paragraph(f"<b>Positive summary:</b> {pos_sum}", BODY))
            if neg_sum:
                story.append(Paragraph(f"<b>Negative summary:</b> {neg_sum}", BODY))
            story.append(Spacer(1, 6))

        # positive reviews
        story.append(Paragraph("Positive reviews (copy/paste block)", H2))
        if not pos:
            story.append(Paragraph("(none)", SMALL))
        else:
            # KeepTogether for the header + first few lines helps avoid orphan headers
            blocks = []
            for i, txt in enumerate(pos, start=1):
                blocks.append(Paragraph(f"[POS {i}] {txt}", MONO))
            story.append(KeepTogether(blocks[: min(8, len(blocks))]))
            for b in blocks[min(8, len(blocks)) :]:
                story.append(b)

        story.append(Spacer(1, 8))

        # negative reviews
        story.append(Paragraph("Negative reviews (copy/paste block)", H2))
        if not neg:
            story.append(Paragraph("(none)", SMALL))
        else:
            blocks = []
            for i, txt in enumerate(neg, start=1):
                blocks.append(Paragraph(f"[NEG {i}] {txt}", MONO))
            story.append(KeepTogether(blocks[: min(8, len(blocks))]))
            for b in blocks[min(8, len(blocks)) :]:
                story.append(b)

    doc.build(story)


# -------------------------
# Orchestrator: build PDF, wait for NotebookLM outputs, rebuild
# -------------------------
def notebooklm_pdf_workflow(
    *,
    pos_texts_by_dept: Dict[str, List[Any]],
    neg_texts_by_dept: Dict[str, List[Any]],
    output_dir: Path,
    # YOU set these:
    output_pdf_name: str = "NotebookLM_CopyPaste_Pack.pdf",
    notebooklm_output_filename: str = "notebooklm_outputs.csv",
    # filtering:
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    keep_undated_when_filtering: bool = True,
    # fallback summaries (from your old qualitative pipeline)
    fallback_pos_summaries: Optional[Dict[str, str]] = None,
    fallback_neg_summaries: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    1) Writes a copy/paste PDF with raw reviews.
    2) Pauses until you put NotebookLM output file in output_dir and press Enter.
    3) Validates/imports that file → stores into a JSON library.
    4) Rewrites PDF including imported summaries.
    5) If invalid, logs issues and rewrites PDF using fallback summaries if provided.

    Returns a dict of paths.
    """
    log = logging.getLogger("notebooklm_packager")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pdf_name = _safe_filename(output_pdf_name)
    notebooklm_output_filename = _safe_filename(notebooklm_output_filename)

    pdf_path = output_dir / output_pdf_name
    imported_path = output_dir / notebooklm_output_filename

    # Step 1: raw pack PDF
    build_notebooklm_pack_pdf(
        pos_texts_by_dept=pos_texts_by_dept,
        neg_texts_by_dept=neg_texts_by_dept,
        output_pdf=pdf_path,
        output_dir=output_dir,
        date_start=date_start,
        date_end=date_end,
        keep_undated_when_filtering=keep_undated_when_filtering,
        include_summaries_from_library=True,  # if you already imported before, it’ll show
        fallback_pos_summaries=fallback_pos_summaries,
        fallback_neg_summaries=fallback_neg_summaries,
        title="NotebookLM Copy/Paste Pack",
    )
    log.info("Wrote PDF (raw reviews): %s", pdf_path)

    # Step 2: wait for NotebookLM output file
    print("\n=== NotebookLM workflow ===")
    print(f"1) Open: {pdf_path.name}")
    print("2) Copy/paste into NotebookLM and produce department summaries.")
    print(f"3) Export your results as CSV/XLSX with columns:")
    print("     department, positive_summary, negative_summary")
    print(f"4) Save it into: {imported_path}")
    input("\nWhen the file is in place, press Enter to import and rebuild the PDF... ")

    # Step 3: import & validate
    df, issues = _load_notebooklm_output_file(imported_path)
    if issues:
        print("\n[NotebookLM import] Issues found:")
        for it in issues:
            print(" - " + it)

    if df is not None:
        upsert_library_from_df(output_dir, df)
        log.info("Imported summaries into library: %s", _library_path(output_dir))
        print("[NotebookLM import] Imported successfully.")
        used_import = True
    else:
        log.warning("NotebookLM import failed; will fall back.")
        print("[NotebookLM import] Import failed; falling back to previous summaries (if provided).")
        used_import = False

    # Step 4: rebuild PDF (now includes imported summaries OR fallback summaries)
    build_notebooklm_pack_pdf(
        pos_texts_by_dept=pos_texts_by_dept,
        neg_texts_by_dept=neg_texts_by_dept,
        output_pdf=pdf_path,
        output_dir=output_dir,
        date_start=date_start,
        date_end=date_end,
        keep_undated_when_filtering=keep_undated_when_filtering,
        include_summaries_from_library=True,  # will use library if import succeeded
        fallback_pos_summaries=fallback_pos_summaries,
        fallback_neg_summaries=fallback_neg_summaries,
        title="NotebookLM Copy/Paste Pack (with Summaries)",
    )
    log.info("Rewrote PDF (with summaries): %s", pdf_path)
    print(f"\nDone. Updated PDF: {pdf_path}\n")

    return {
        "pdf_path": str(pdf_path),
        "imported_file_expected": str(imported_path),
        "library_json": str(_library_path(output_dir)),
        "import_used": str(used_import),
    }
