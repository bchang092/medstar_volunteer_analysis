# report.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import io
import pandas as pd

from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Your module that draws the 4 dashboard plots; it calls collector.savefig(fig)
from .graphs import page_for_department


# ---------- Figure collector for the 4 dashboard plots ----------
class _FigureCollector:
    """Collects matplotlib figures as PNG bytes via .savefig(fig)."""
    def __init__(self):
        self.images: List[bytes] = []
    def savefig(self, fig, **kwargs):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
        buf.seek(0)
        self.images.append(buf.getvalue())


# ---------- styles / header-footer ----------
def _styles():
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleBig", parent=styles["Title"], fontSize=20, leading=24,
        spaceAfter=10, textColor=colors.HexColor("#0f172a")
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=15, leading=18,
        spaceBefore=0, spaceAfter=6, textColor=colors.HexColor("#111827")
    )
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=10.5, leading=14,
        textColor=colors.HexColor("#111827")
    )
    small = ParagraphStyle(
        "Small", parent=styles["BodyText"], fontSize=9, leading=12,
        textColor=colors.HexColor("#6b7280")
    )
    return title, h2, body, small


def _page_decorators(report_title: str, window_str: str):
    def header_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 11)
        canvas.setFillColor(colors.HexColor("#0f172a"))
        canvas.drawString(0.7 * inch, doc.pagesize[1] - 0.6 * inch, report_title)
        if window_str:
            canvas.setFont("Helvetica", 9)
            canvas.setFillColor(colors.HexColor("#334155"))
            canvas.drawRightString(doc.pagesize[0] - 0.7 * inch, doc.pagesize[1] - 0.6 * inch, window_str)
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.HexColor("#6b7280"))
        canvas.drawRightString(doc.pagesize[0] - 0.7 * inch, 0.5 * inch, f"Page {doc.page}")
        canvas.restoreState()
    return header_footer


# ---------- utilities ----------
def _scale_image_to_box(img: Image, max_w: float, max_h: float) -> None:
    iw, ih = img.imageWidth, img.imageHeight
    if iw <= 0 or ih <= 0:
        return
    scale = min(max_w / iw, max_h / ih)
    img.drawWidth  = iw * scale
    img.drawHeight = ih * scale


def _image_from_png_bytes(png_bytes: Optional[bytes], max_w: float, max_h: float) -> Optional[Image]:
    if not png_bytes:
        return None
    img = Image(io.BytesIO(png_bytes))
    _scale_image_to_box(img, max_w, max_h)
    return img


def _image_from_mpl_fig(fig, max_w: float, max_h: float) -> Optional[Image]:
    """Convert a live Matplotlib Figure to a ReportLab Image (in memory)."""
    if fig is None:
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    img = Image(buf)
    _scale_image_to_box(img, max_w, max_h)
    return img


# ---------- 2×3 dashboard page (uses topic-map Figures directly) ----------
def _dept_dashboard_grid_flowables(
    dept: str,
    months: List[str],
    monthly_data: Dict[str, Any],
    dep_service: Dict[str, Any],
    dep_wait: Dict[str, Any],
    *,
    date_start: Optional[str],
    date_end: Optional[str],
    frame_w: float,
    frame_h: float,
    pos_topic_fig,   # Matplotlib Figure (Positive) or None
    neg_topic_fig,   # Matplotlib Figure (Negative) or None
) -> List:
    """
    One page per department with 6 plots in a 2×3 grid:
      - First 4: dashboard plots from graphs.page_for_department
      - Last 2: Positive & Negative qualitative topic-map Figures
    """
    title, h2, body, small = _styles()

    # Collect the 4 dashboard plots (PNG bytes)
    collector = _FigureCollector()
    page_for_department(
        collector, dept, months, monthly_data, dep_service, dep_wait,
        date_start=date_start, date_end=date_end
    )
    dash_pngs = collector.images[:4]

    # ---- Layout: fill the available frame ----
    reserved_header = 0.35 * inch
    grid_h = max(0.5 * inch, frame_h - reserved_header)
    n_rows, n_cols = 2, 3
    gutter_x = 0.12 * inch
    gutter_y = 0.12 * inch

    total_gutter_x = gutter_x * (n_cols - 1)
    col_w = (frame_w - total_gutter_x) / n_cols
    total_gutter_y = gutter_y * (n_rows - 1)
    row_h = (grid_h - total_gutter_y) / n_rows

    # Convert to ReportLab images (scale to cell)
    cells: List[Optional[Image]] = []
    for b in dash_pngs:
        cells.append(_image_from_png_bytes(b, col_w, row_h))
    cells.append(_image_from_mpl_fig(pos_topic_fig, col_w, row_h))
    cells.append(_image_from_mpl_fig(neg_topic_fig, col_w, row_h))
    while len(cells) < 6:
        cells.append(None)

    data = [
        [cells[0], cells[1], cells[2]],
        [cells[3], cells[4], cells[5]],
    ]

    tbl = Table(
        data,
        colWidths=[col_w, col_w, col_w],
        rowHeights=[row_h, row_h],
        hAlign="LEFT",  # span the full frame width
        style=TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ])
    )

    heading = Paragraph(f"<b>{dept}</b> — Monthly dashboard + thematic maps", h2)
    spacer = Spacer(1, gutter_y * 0.5)
    return [KeepTogether([heading, spacer, tbl])]


# ---------- public API ----------
def build_pdf(
    output_pdf_path,
    department_ind_dict: Dict[str, List[int]],
    monthly_data: Dict[str, Any],
    dep_service: Dict[str, Any],
    dep_wait: Dict[str, Any],
    months: List[str],
    *,
    # Preferred: pass these directly from qualitative.run_qualitative()
    pos_theme_figs_mpl: Optional[Dict[str, Any]] = None,  # {dept: Matplotlib Figure}
    neg_theme_figs_mpl: Optional[Dict[str, Any]] = None,  # {dept: Matplotlib Figure}
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
):
    """
    Build the PDF:
      • One page per department with a 2×3 grid:
          4 dashboard plots + Positive & Negative qualitative topic maps (Matplotlib Figures).
    """
    pos_theme_figs_mpl = pos_theme_figs_mpl or {}
    neg_theme_figs_mpl = neg_theme_figs_mpl or {}

    window_str = ""
    if date_start or date_end:
        s = date_start or "…"; e = date_end or "…"
        window_str = f"Window: {s} → {e}"
    report_title = "Monthly Volunteer Rounding Report"

    pagesize = landscape(letter)
    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=pagesize,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.9 * inch, bottomMargin=0.7 * inch,
        title=report_title, author="MedStar Volunteer Analysis",
    )

    frame_w, frame_h = doc.width, doc.height
    title_style, h2, body, small = _styles()
    story: List = []

    # Cover
    cover_lines = []
    if months:
        cover_lines.append(f"Latest month: <b>{months[-1]}</b>")
    if window_str:
        cover_lines.append(window_str)
    story.append(Paragraph(report_title, title_style))
    for line in cover_lines:
        story.append(Paragraph(line, body))
    story.append(Spacer(1, 0.3 * inch))
    story.append(PageBreak())

    # Per-department pages
    depts = list(department_ind_dict.keys())
    for i, dept in enumerate(depts):
        pos_fig = pos_theme_figs_mpl.get(dept)
        neg_fig = neg_theme_figs_mpl.get(dept)

        story.extend(_dept_dashboard_grid_flowables(
            dept, months, monthly_data, dep_service, dep_wait,
            date_start=date_start, date_end=date_end,
            frame_w=frame_w, frame_h=frame_h,
            pos_topic_fig=pos_fig, neg_topic_fig=neg_fig
        ))

        if i < len(depts) - 1:
            story.append(PageBreak())

    # Build
    doc.build(
        story,
        onFirstPage=_page_decorators(report_title, window_str),
        onLaterPages=_page_decorators(report_title, window_str),
    )
    print(f"Saved PDF report → {output_pdf_path}")
