# ga_pipeline/report.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import io
import re
import unicodedata

from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.graphics.shapes import Drawing, Line, Rect

from .graphs import page_for_department


# -------- Figure collector for the dashboard (Matplotlib -> PNG bytes) --------
class _FigureCollector:
    """
    A tiny shim that looks like a PdfPages object, but
    captures each figure as PNG bytes in memory.
    """
    def __init__(self):
        self.images: List[bytes] = []

    def savefig(self, fig, **kwargs):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=168, bbox_inches="tight")
        buf.seek(0)
        self.images.append(buf.getvalue())


# -------- styles --------
def _styles():
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleBig",
        parent=styles["Title"],
        fontSize=19,
        leading=22,
        spaceAfter=8,
        textColor=colors.HexColor("#0f172a"),
    )
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=15,
        leading=18,
        spaceBefore=0,
        spaceAfter=4,
        textColor=colors.HexColor("#111827"),
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10.2,
        leading=13.6,
        textColor=colors.HexColor("#111827"),
    )
    small = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontSize=9,
        leading=11.5,
        textColor=colors.HexColor("#4b5563"),
    )
    return title, h2, body, small


# -------- page decorations (header/footer) --------
def _page_decorators():
    """
    No top header.
    Keep a small page number at the bottom-right.
    """
    def header_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8.8)
        canvas.setFillColor(colors.HexColor("#6b7280"))
        canvas.drawRightString(doc.pagesize[0] - 0.6 * inch, 0.45 * inch, f"Page {doc.page}")
        canvas.restoreState()
    return header_footer


# -------- utilities --------
def _scale_image_to_box(img: Image, max_w: float, max_h: float) -> None:
    iw, ih = img.imageWidth, img.imageHeight
    if iw <= 0 or ih <= 0:
        return
    scale = min(max_w / iw, max_h / ih)
    img.drawWidth = iw * scale
    img.drawHeight = ih * scale


def _image_from_png_bytes(png_bytes: Optional[bytes], max_w: float, max_h: float) -> Optional[Image]:
    if not png_bytes:
        return None
    img = Image(io.BytesIO(png_bytes))
    _scale_image_to_box(img, max_w, max_h)
    return img


def _norm_dept(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _summary_for_department(df: Optional[Any], dept: str) -> str:
    """
    Given pos_theme_df/neg_theme_df with columns ['department','summary','method'],
    return the first non-empty summary string for that department.
    Uses normalized dept matching to avoid unicode/whitespace mismatches.
    """
    if df is None:
        return ""
    try:
        if df.empty or "department" not in df.columns or "summary" not in df.columns:
            return ""
    except AttributeError:
        return ""

    dept_n = _norm_dept(dept)
    # build normalized lookup
    dep_col = df["department"].fillna("").astype(str).map(_norm_dept)
    sub = df[dep_col == dept_n]
    if sub.empty:
        return ""
    texts = [str(s).strip() for s in sub["summary"].fillna("").tolist() if str(s).strip()]
    return texts[0] if texts else ""


def _format_review_stats_line(review_stats: Optional[Dict[str, Any]]) -> str:
    """
    Small line under the title:
      "Volunteer submissions: total 123 | 2025-11: 17 (pos 9, neg 8)"
    """
    if not review_stats:
        return ""

    total = int(review_stats.get("total_reviews", 0) or 0)
    lm = str(review_stats.get("latest_month", "") or "").strip()
    lm_total = int(review_stats.get("latest_month_total_reviews", 0) or 0)
    lm_pos = int(review_stats.get("latest_month_positive_reviews", 0) or 0)
    lm_neg = int(review_stats.get("latest_month_negative_reviews", 0) or 0)

    if not lm:
        return f"Volunteer submissions: total <b>{total}</b>"

    return (
        f"Volunteer submissions: total <b>{total}</b> "
        f"| <b>{lm}</b>: <b>{lm_total}</b> "
        f"(pos {lm_pos}, neg {lm_neg})"
    )


def _cover_page_flowables(
    frame_w: float,
    frame_h: float,
    month_of_analysis: Optional[str],
    date_start: Optional[str],
    date_end: Optional[str],
    review_stats: Optional[Dict[str, Any]],
):
    """
    Build a simple, centered cover page with a solid backdrop.
    """
    title_style, _, body, _ = _styles()

    # palette
    bg = colors.HexColor("#0b1f36")
    accent = colors.HexColor("#d9a35a")
    text_main = colors.white
    text_sub = colors.HexColor("#e5e7eb")

    month_str = month_of_analysis or "Not specified"
    quarter_str = f"{date_start or '…'} → {date_end or '…'}"
    stats_line = _format_review_stats_line(review_stats)

    title_para = Paragraph("Volunteer Rounding Report", title_style.clone(
        "CoverTitle",
        fontSize=30,
        leading=36,
        textColor=text_main,
        spaceAfter=14,
    ))
    subtitle_para = Paragraph(
        f"Month of analysis: <b>{month_str}</b><br/>Quarter window: <b>{quarter_str}</b>",
        body.clone("CoverBody", textColor=text_main, fontSize=12.5, leading=16),
    )
    stats_para = Paragraph(
        stats_line or "",
        body.clone("CoverStats", textColor=text_sub, fontSize=11, leading=14),
    ) if stats_line else None

    content = [
        Spacer(1, frame_h * 0.18),
        title_para,
        subtitle_para,
    ]
    if stats_para:
        content.extend([Spacer(1, 8), stats_para])
    content.append(Spacer(1, frame_h * 0.10))

    inner = Table(
        [[c] for c in content],
        colWidths=[frame_w * 0.7],
        style=TableStyle([
            ("LEFTPADDING", (0, 0), (-1, -1), 16),
            ("RIGHTPADDING", (0, 0), (-1, -1), 16),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]),
    )

    cover = Table(
        [[inner]],
        colWidths=[frame_w],
        rowHeights=[frame_h * 0.9],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), frame_w * 0.08),
            ("RIGHTPADDING", (0, 0), (-1, -1), frame_w * 0.08),
            ("TOPPADDING", (0, 0), (-1, -1), frame_h * 0.05),
            ("BOTTOMPADDING", (0, 0), (-1, -1), frame_h * 0.05),
        ]),
    )

    accent_rule = Table(
        [[Spacer(1, 0.2 * inch)]],
        colWidths=[frame_w * 0.25],
        style=TableStyle([
            ("LINEABOVE", (0, 0), (-1, -1), 2, accent),
            ("BACKGROUND", (0, 0), (-1, -1), None),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]),
    )

    return [cover, Spacer(1, 6), accent_rule, PageBreak()]


def _dept_sort_key(dept: str):
    """
    Order PDF pages:
      - MUMH first
      - MGSH second
      - everything else after
    Then alphabetical within each group.
    """
    d = _norm_dept(dept)
    prefix = d.split(":", 1)[0].strip().upper() if ":" in d else ""
    order = {"MUMH": 0, "MGSH": 1}
    return (order.get(prefix, 99), d)


# -------- one-page layout per department --------
def _dept_page_flowables(
    dept: str,
    months: List[str],
    monthly_data: Dict[str, Any],
    dep_service: Dict[str, Any],
    dep_wait: Dict[str, Any],
    *,
    date_start: Optional[str],
    date_end: Optional[str],
    month_of_analysis: Optional[str],
    frame_w: float,
    frame_h: float,
    pos_summary: str,
    neg_summary: str,
    review_stats: Optional[Dict[str, Any]],
) -> List:
    """
    Layout (single page):
      Title line: "<DEPT> — Monthly Dashboard (Window: ...)"
      Stats line: "Volunteer submissions: total ... | latest_month: ..."
      Thin divider
      [  2×2 grid of the four dashboard charts (PNG images)    ]
      [  Bottom row: Positive & Negative themes text boxes      ]
    """
    _, h2, body, small = _styles()

    window_str = ""
    if date_start or date_end:
        s = date_start or "…"
        e = date_end or "…"
        window_str = f"(Window: {s} → {e})"

    heading = Paragraph(f"<b>{dept}</b> — Monthly Dashboard {window_str}", h2)

    stats_line = _format_review_stats_line(review_stats)
    stats_para = Paragraph(stats_line, small) if stats_line else None

    divider = Drawing(frame_w, 0.2 * inch)
    divider.add(Line(0, 0, frame_w, 0, strokeColor=colors.HexColor("#94a3b8"), strokeWidth=1.1))

    collector = _FigureCollector()
    page_for_department(
        collector,
        dept,
        months,
        monthly_data,
        dep_service,
        dep_wait,
        date_start=date_start,
        date_end=date_end,
        month_of_analysis=month_of_analysis,
    )

    flows: List = []
    flows.append(heading)
    if stats_para:
        flows.append(Spacer(1, 0.02 * inch))
        flows.append(stats_para)

    flows.append(Spacer(1, 0.04 * inch))
    flows.append(divider)
    flows.append(Spacer(1, 0.10 * inch))

    # Submission counts (month + quarter)
    def _submissions_for_month(m: Optional[str]) -> int:
        if not m:
            return 0
        try:
            return int(monthly_data.get(m, {}).get("submissions", {}).get(dept, 0) or 0)
        except Exception:
            return 0

    snapshot_month = None
    if months:
        if month_of_analysis and month_of_analysis in months:
            snapshot_month = month_of_analysis
        else:
            snapshot_month = months[-1]
    month_submissions = _submissions_for_month(snapshot_month)
    quarter_submissions = sum(_submissions_for_month(m) for m in months)
    if snapshot_month or months:
        counts_line = (
            f"Reviews — Month {snapshot_month or 'N/A'}: <b>{month_submissions}</b> | "
            f"Quarter: <b>{quarter_submissions}</b>"
        )
        flows.append(Paragraph(counts_line, small))
        flows.append(Spacer(1, 0.06 * inch))

    if collector.images:
        col_w = frame_w / 2.0 - 0.08 * inch
        max_month_h = frame_h * 0.30
        max_qtr_h = frame_h * 0.30

        imgs_month: List[Optional[Image]] = [
            _image_from_png_bytes(b, max_w=col_w, max_h=max_month_h)
            for b in collector.images[:2]
        ]
        imgs_qtr: List[Optional[Image]] = [
            _image_from_png_bytes(b, max_w=col_w, max_h=max_qtr_h)
            for b in collector.images[2:4]
        ]
        while len(imgs_month) < 2:
            imgs_month.append(None)
        while len(imgs_qtr) < 2:
            imgs_qtr.append(None)

        def _cell(content):
            return content if content is not None else Spacer(1, 0.05 * inch)

        left_table = Table(
            [
                [Paragraph(f"<b>Monthly ({snapshot_month or 'latest'}) Report</b>", small)],
                [_cell(imgs_month[0])],
                [Spacer(1, 0.08 * inch)],
                [_cell(imgs_month[1])],
            ],
            colWidths=[col_w],
            style=TableStyle([
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]),
        )

        right_table = Table(
            [
                [Paragraph("<b>Quarterly Report</b>", small)],
                [_cell(imgs_qtr[0])],
                [Spacer(1, 0.08 * inch)],
                [_cell(imgs_qtr[1])],
            ],
            colWidths=[col_w],
            style=TableStyle([
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]),
        )

        combo = Table(
            [[left_table, right_table]],
            colWidths=[col_w, col_w],
            style=TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]),
        )

        flows.append(combo)
        flows.append(Spacer(1, 0.12 * inch))
    else:
        flows.append(Paragraph("(No dashboard data available.)", small))
        flows.append(Spacer(1, 0.20 * inch))

    # --- Bottom: Positive / Negative theme summary boxes ---
    pos_text = pos_summary or "(Insufficient data for a positive themes summary.)"
    neg_text = neg_summary or "(Insufficient data for a negative themes summary.)"

    pos_para = Paragraph(
        "<b>Positive themes</b><br/><font size=9 color='#4b5563'>"
        + pos_text.replace("\n", "<br/>")
        + "</font>",
        body,
    )
    neg_para = Paragraph(
        "<b>Negative themes</b><br/><font size=9 color='#4b5563'>"
        + neg_text.replace("\n", "<br/>")
        + "</font>",
        body,
    )

    col_w = (frame_w - 0.12 * inch) / 2.0
    bottom_table = Table(
        [[pos_para, neg_para]],
        colWidths=[col_w, col_w],
        hAlign="LEFT",
        style=TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("BOX", (0, 0), (0, 0), 0.8, colors.HexColor("#d1d5db")),
            ("BOX", (1, 0), (1, 0), 0.8, colors.HexColor("#d1d5db")),
            ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#f9fafb")),
            ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#f9fafb")),
        ]),
    )

    flows.append(bottom_table)
    return flows


# -------- public API --------
def build_pdf(
    output_pdf_path,
    department_ind_dict: Dict[str, List[int]],
    monthly_data: Dict[str, Any],
    dep_service: Dict[str, Any],
    dep_wait: Dict[str, Any],
    months: List[str],
    *,
    pos_theme_df: Optional[Any] = None,
    neg_theme_df: Optional[Any] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    review_stats: Optional[Dict[str, Any]] = None,  # NEW
    month_of_analysis: Optional[str] = None,
):
    """
    Build the PDF:
      • Cover page
      • One page per department:
          - Left: monthly charts for selected month
          - Right: quarter heatmaps (within date window)
          - Positive & Negative qualitative summaries as text boxes
          - Stats line under title (total + latest-month submissions)
    """
    pagesize = landscape(letter)
    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=pagesize,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.6 * inch,
        title="Volunteer Rounding Report",
        author="MedStar Volunteer Analysis",
    )

    frame_w, frame_h = doc.width, doc.height
    title_style, _, body, _ = _styles()
    story: List = []

    # ----- Cover page -----
    story.extend(_cover_page_flowables(frame_w, frame_h, month_of_analysis, date_start, date_end, review_stats))

    # ----- Per-department pages -----
    depts = sorted(list(department_ind_dict.keys()), key=_dept_sort_key)

    for i, dept in enumerate(depts):
        pos_summary = _summary_for_department(pos_theme_df, dept)
        neg_summary = _summary_for_department(neg_theme_df, dept)

        story.extend(
            _dept_page_flowables(
                dept,
                months,
                monthly_data,
                dep_service,
                dep_wait,
                date_start=date_start,
                date_end=date_end,
                month_of_analysis=month_of_analysis,
                frame_w=frame_w,
                frame_h=frame_h,
                pos_summary=pos_summary,
                neg_summary=neg_summary,
                review_stats=review_stats,
            )
        )
        if i < len(depts) - 1:
            story.append(PageBreak())

    doc.build(
        story,
        onFirstPage=_page_decorators(),
        onLaterPages=_page_decorators(),
    )
    print(f"Saved PDF report → {output_pdf_path}")
