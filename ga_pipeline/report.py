# ga_pipeline/report.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import io

from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.graphics.shapes import Drawing, Line

from .graphs import page_for_department


# -------- Figure collector for the 4 dashboard plots (Matplotlib -> PNG bytes) --------
class _FigureCollector:
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
    return title, h2, body


# -------- page decorations (header/footer) --------
def _page_decorators():
    """
    No top header (per request).
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
    img.drawWidth  = iw * scale
    img.drawHeight = ih * scale

def _image_from_png_bytes(png_bytes: Optional[bytes], max_w: float, max_h: float) -> Optional[Image]:
    if not png_bytes:
        return None
    img = Image(io.BytesIO(png_bytes))
    _scale_image_to_box(img, max_w, max_h)
    return img

def _image_from_mpl_fig(fig, max_w: float, max_h: float) -> Optional[Image]:
    if fig is None:
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    img = Image(buf)
    _scale_image_to_box(img, max_w, max_h)
    return img


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
    frame_w: float,
    frame_h: float,
    pos_topic_fig,   # Matplotlib Figure (Positive) or None
    neg_topic_fig,   # Matplotlib Figure (Negative) or None
) -> List:
    """
    Layout:
      Title line: "<DEPT> — Monthly Dashboard (Window: ...)"
      Divider line
      ┌──────────────────────────────────────────────────────────────┐
      │ [Chart1]              │  [Chart3]                           │
      │ [Chart2]              │  [Chart4]                           │   <-- Top: two vertical stacks
      ├──────────────────────────────────────────────────────────────┤
      │ [ Positive topic map ]   [ Negative topic map ]              │   <-- Bottom: boxed
      └──────────────────────────────────────────────────────────────┘
    """
    _, h2, _ = _styles()

    # --- Header line text with window inline ---
    window_str = ""
    if date_start or date_end:
        s = date_start or "…"
        e = date_end or "…"
        window_str = f"(Window: {s} → {e})"

    heading = Paragraph(f"<b>{dept}</b> — Monthly Dashboard {window_str}", h2)

    # Divider under header
    divider = Drawing(frame_w, 0.2 * inch)
    divider.add(Line(0, 0, frame_w, 0, strokeColor=colors.HexColor("#94a3b8"), strokeWidth=1.1))

    # Collect the 4 dashboard plots (PNG bytes)
    collector = _FigureCollector()
    page_for_department(
        collector, dept, months, monthly_data, dep_service, dep_wait,
        date_start=date_start, date_end=date_end
    )
    charts = collector.images[:4] + [None] * max(0, 4 - len(collector.images))

    # --- compact vertical proportions to ensure single-page fit ---
    top_h_share    = 0.50       # top area
    bottom_h_share = 0.36       # bottom area
    v_gap          = 0.04 * inch
    col_gutter     = 0.14 * inch

    # --- TOP: two vertical stacks (each = two charts stacked) ---
    top_h = frame_h * top_h_share
    left_col_w = (frame_w - col_gutter) / 2.0
    right_col_w = left_col_w
    stack_gap = 0.04 * inch
    stack_row_h = (top_h - stack_gap) / 2.0

    left_stack = Table(
        [[_image_from_png_bytes(charts[0], left_col_w, stack_row_h)],
         [_image_from_png_bytes(charts[1], left_col_w, stack_row_h)]],
        colWidths=[left_col_w],
        rowHeights=[stack_row_h, stack_row_h],
        hAlign="LEFT",
        style=TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ])
    )

    right_stack = Table(
        [[_image_from_png_bytes(charts[2], right_col_w, stack_row_h)],
         [_image_from_png_bytes(charts[3], right_col_w, stack_row_h)]],
        colWidths=[right_col_w],
        rowHeights=[stack_row_h, stack_row_h],
        hAlign="LEFT",
        style=TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ])
    )

    top_row = Table(
        [[left_stack, right_stack]],
        colWidths=[left_col_w, right_col_w],
        rowHeights=[top_h],
        hAlign="LEFT",
        style=TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("LINEAFTER", (0, 0), (0, 0), 0.8, colors.HexColor("#cbd5e1")),  # vertical divider
        ])
    )

    # --- BOTTOM: two qualitative plots side-by-side (boxed) ---
    bottom_h = frame_h * bottom_h_share
    q_col_w = (frame_w - col_gutter) / 2.0
    q_row_h = bottom_h

    pos_img = _image_from_mpl_fig(pos_topic_fig, q_col_w, q_row_h)
    neg_img = _image_from_mpl_fig(neg_topic_fig, q_col_w, q_row_h)

    bottom_row = Table(
        [[pos_img, neg_img]],
        colWidths=[q_col_w, q_col_w],
        rowHeights=[q_row_h],
        hAlign="LEFT",
        style=TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("BOX", (0, 0), (0, 0), 1, colors.HexColor("#94a3b8")),
            ("BOX", (1, 0), (1, 0), 1, colors.HexColor("#94a3b8")),
        ])
    )

    flows: List = []
    flows.append(heading)
    flows.append(Spacer(1, 0.04 * inch))
    flows.append(divider)
    flows.append(Spacer(1, 0.08 * inch))
    flows.append(top_row)
    flows.append(Spacer(1, v_gap))
    flows.append(bottom_row)
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
    # Preferred new args: pass live Matplotlib figures for qualitative
    pos_theme_figs_mpl: Optional[Dict[str, Any]] = None,  # {dept: Matplotlib Figure}
    neg_theme_figs_mpl: Optional[Dict[str, Any]] = None,  # {dept: Matplotlib Figure}
    # Legacy args accepted (ignored here, kept for compatibility)
    pos_theme_df: Optional[Any] = None,
    neg_theme_df: Optional[Any] = None,
    pos_theme_figs: Optional[Dict[str, bytes]] = None,
    neg_theme_figs: Optional[Dict[str, bytes]] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
):
    pos_theme_figs_mpl = pos_theme_figs_mpl or {}
    neg_theme_figs_mpl = neg_theme_figs_mpl or {}

    # Smaller page margins to maximize space (landscape letter)
    pagesize = landscape(letter)
    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=pagesize,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.65 * inch, bottomMargin=0.6 * inch,
        title="Volunteer Rounding Report", author="MedStar Volunteer Analysis",
    )

    frame_w, frame_h = doc.width, doc.height
    title_style, _, body = _styles()
    story: List = []

    # Cover
    if months or (date_start or date_end):
        story.append(Paragraph("Monthly Volunteer Rounding Report", title_style))
        if months:
            story.append(Paragraph(f"Latest month: <b>{months[-1]}</b>", body))
        if date_start or date_end:
            s = date_start or "…"; e = date_end or "…"
            story.append(Paragraph(f"Window: {s} → {e}", body))
        story.append(Spacer(1, 0.24 * inch))
        story.append(PageBreak())

    # Per-department pages
    depts = list(department_ind_dict.keys())
    for i, dept in enumerate(depts):
        pos_fig = pos_theme_figs_mpl.get(dept)
        neg_fig = neg_theme_figs_mpl.get(dept)

        story.extend(_dept_page_flowables(
            dept, months, monthly_data, dep_service, dep_wait,
            date_start=date_start, date_end=date_end,
            frame_w=frame_w, frame_h=frame_h,
            pos_topic_fig=pos_fig, neg_topic_fig=neg_fig
        ))
        if i < len(depts) - 1:
            story.append(PageBreak())

    doc.build(
        story,
        onFirstPage=_page_decorators(),
        onLaterPages=_page_decorators(),
    )
    print(f"Saved PDF report → {output_pdf_path}")
