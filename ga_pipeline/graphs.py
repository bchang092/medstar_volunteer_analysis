# ga_pipeline/graphs.py
from typing import Dict, Any, List, Optional, Tuple
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# ----------- Visual defaults -----------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.prop_cycle":  plt.cycler("color", [cm.Blues(i) for i in np.linspace(0.35, 0.95, 6)]),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 140,
})

# ----------- Normalization knobs for heatmaps -----------
NORMALIZE_HEATMAPS = True
RATE_PER_SUBMISSIONS = 10.0  # patients per N submissions in heatmaps

# ----------- Constants / heuristics -----------
AVG_PATIENTS_PER_SUBMISSION = 8.0

# Your original “bin → midpoint” mappings (used in heatmaps)
SERVICE_BIN_PATIENTS_MAP = {
    "0": 0, "0 patients": 0, "0 patient": 0,
    "1 to 3 patients": 2, "1-3 patients": 2, "1–3 patients": 2, "1 – 3 patients": 2,
    "4 to 6 patients": 5, "4-6 patients": 5, "4–6 patients": 5, "4 – 6 patients": 5,
    "6+ patients": 9, "6 + patients": 9, "6 plus patients": 9, "6 or more patients": 9,
}

WAIT_BAND_MIDPOINT = {
    "0-25%": 0.125, "25-50%": 0.375, "50-75%": 0.625, "75-100%": 0.875,
}

# For the SERVICES SNAPSHOT (top-left), we’re going to bin per-submission quantities:
SERVICE_QTY_BANDS = ["0", "1–3", "4–6", "6+"]

# ----------- Utilities -----------
def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def _sum_counts(d: Dict[str, int]) -> int:
    return int(sum(d.values())) if d else 0

def _patients_per_submission(monthly_data: Dict[str, Any], month: str, dept: str) -> float:
    try:
        return float(monthly_data.get(month, {}).get("patients_per_shift", {}).get(dept, AVG_PATIENTS_PER_SUBMISSION))
    except Exception:
        return AVG_PATIENTS_PER_SUBMISSION

def _total_service_submissions(monthly_data: Dict[str, Any], month: str, dept: str) -> float:
    svc = monthly_data.get(month, {}).get("service", {}).get(dept, {}) or {}
    return float(sum(_sum_counts(qdict) for qdict in svc.values()))

def _total_wait_submissions(monthly_data: Dict[str, Any], month: str, dept: str) -> float:
    """
    Number of survey submissions for a department in a given month.
    Stored directly in processing.py as monthly_data[month]["submissions"][dept].
    """
    return float(monthly_data.get(month, {}).get("submissions", {}).get(dept, 0.0))

def _total_wait_patients(monthly_data: Dict[str, Any], month: str, dept: str) -> float:
    wt = monthly_data.get(month, {}).get("wait", {}).get(dept, {}) or {}
    return float(sum(_sum_counts(qdict) for qdict in wt.values()))

def _blues(n: int) -> List:
    return [cm.Blues(x) for x in np.linspace(0.35, 0.95, max(n, 1))]

def _service_bin_patients(label: str) -> Optional[int]:
    key = _norm_key(label)
    if key in SERVICE_BIN_PATIENTS_MAP:
        return SERVICE_BIN_PATIENTS_MAP[key]
    m = re.match(r"^(\d+)\s*[-–]\s*(\d+)\s*patients?$", key)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return int(round((lo + hi) / 2.0))
    if key.startswith("6"):
        return 9
    if key == "0":
        return 0
    return None

def _wait_band_midpoint(label: str) -> Optional[float]:
    key = _norm_key(label)
    if key in WAIT_BAND_MIDPOINT:
        return WAIT_BAND_MIDPOINT[key]
    m = re.match(r"^(\d+)\s*[-–]\s*(\d+)\s*%$", key)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return ((lo + hi) / 2.0) / 100.0
    return None

def _extract_first_int_or_none(val) -> Optional[int]:
    """Extract first integer from cell; return None if blank/unparseable."""
    if val is None:
        return None
    try:
        if isinstance(val, float) and np.isnan(val):
            return None
    except Exception:
        pass
    s = str(val).strip()
    if not s:
        return None
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else None

def _qty_to_band(qty: Optional[int]) -> Optional[str]:
    """
    Convert per-submission quantity to a band for top-left snapshot.
    None => not counted as a submission mentioning that service
    """
    if qty is None:
        return None
    if qty == 0:
        return "0"
    if 1 <= qty <= 3:
        return "1–3"
    if 4 <= qty <= 6:
        return "4–6"
    if qty >= 7:
        return "6+"
    return None

def _pick_snapshot_month(months: List[str], month_of_analysis: Optional[str]) -> Optional[str]:
    """
    Choose the month to show in the monthly (snapshot) charts.
    Prefer an explicit month_of_analysis if it exists in the data; otherwise
    fall back to the latest month available.
    """
    if not months:
        return None
    if month_of_analysis and month_of_analysis in months:
        return month_of_analysis
    return months[-1]

# ---------------------------------------------------------------------
# TOP-LEFT: Services snapshot (STACKED, horizontal; volunteer submissions)
# ---------------------------------------------------------------------
def _build_services_snapshot_submission_bands(
    svc_recent: Dict[str, Dict[str, int]],
    *,
    top_n: int = 10
) -> Dict[str, Dict[str, int]]:
    """
    Input svc_recent: monthly_data[month]["service"][dept] structure:
      svc_recent[service_name][raw_resp] = count_of_submissions_with_that_resp

    Output:
      out[service_name][band] = count_of_submissions (binned by qty band)
    """
    if not svc_recent:
        return {}

    # First compute totals (submissions mentioning that service, across non-blank parseable qty)
    totals: Dict[str, int] = {}
    for service_name, resp_counts in svc_recent.items():
        total = 0
        for raw_resp, c in (resp_counts or {}).items():
            qty = _extract_first_int_or_none(raw_resp)
            band = _qty_to_band(qty)
            if band is None:
                continue
            total += int(c)
        totals[service_name] = total

    # Top N services by total submission mentions
    top_services = [k for k, _ in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:top_n] if totals[k] > 0]

    out: Dict[str, Dict[str, int]] = {}
    for service_name in top_services:
        out[service_name] = {b: 0 for b in SERVICE_QTY_BANDS}
        resp_counts = svc_recent.get(service_name, {}) or {}
        for raw_resp, c in resp_counts.items():
            qty = _extract_first_int_or_none(raw_resp)
            band = _qty_to_band(qty)
            if band is None:
                continue
            out[service_name][band] += int(c)

    return out

def _plot_services_stacked_submissions(ax, svc_recent: Dict[str, Dict[str, int]]):
    """
    Plots services as submission counts, stacked by quantity band within a submission.
    """
    snapshot = _build_services_snapshot_submission_bands(svc_recent, top_n=10)
    if not snapshot:
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center")
        ax.set_axis_off()
        return

    services = list(snapshot.keys())
    bands_sorted = SERVICE_QTY_BANDS[:]  # fixed order
    colors = _blues(len(bands_sorted))

    y = np.arange(len(services))
    left = np.zeros(len(services), dtype=float)

    for color, b in zip(colors, bands_sorted):
        vals = np.array([float(snapshot.get(s, {}).get(b, 0)) for s in services], dtype=float)
        ax.barh(y, vals, left=left, label=b, color=color, edgecolor="white", linewidth=0.5)
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(services)
    ax.set_xlabel("Number of Volunteer Submissions")
    ax.set_title("Most Commonly Requested Services (submissions)")
    ax.legend(title="Qty in submission", loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    # Set xlim to total submissions that month if possible (approx: max bar length)
    xmax = float(np.max(left)) if len(left) else 0.0
    ax.set_xlim(0, max(1.0, xmax))

# ---------------------------------------------------------------------
# TOP-RIGHT: Wait snapshot (STACKED, horizontal; NUMBER OF PATIENTS)
# ---------------------------------------------------------------------
def _plot_wait_stacked_patients(ax, wt_recent: Dict[str, Dict[str, int]]):
    """
    Expects wt_recent: monthly_data[month]["wait"][dept] structure where:
      wt_recent[question][band] = TOTAL PATIENTS
    (your processing.py currently does += num_patients for wait responses)
    """
    if not wt_recent:
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center")
        ax.set_axis_off()
        return

    questions = list(wt_recent.keys())
    bands = list({str(b) for q in questions for b in (wt_recent.get(q, {}) or {}).keys() if b is not None and str(b).strip()})

    band_order = ["0-25%", "25-50%", "50-75%", "75-100%"]
    bands_sorted = [b for b in band_order if b in bands] + [b for b in bands if b not in band_order]
    colors = _blues(len(bands_sorted))

    y = np.arange(len(questions))
    left = np.zeros(len(questions), dtype=float)

    for color, b in zip(colors, bands_sorted):
        vals = np.array([float(wt_recent.get(q, {}).get(b, 0)) for q in questions], dtype=float)
        ax.barh(y, vals, left=left, label=b, color=color, edgecolor="white", linewidth=0.5)
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(questions)
    ax.set_xlabel("Number of Patients")
    ax.set_title("Wait Time Report (patients)")
    ax.legend(title="Wait band", loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    # x-axis max based on largest total patients across questions
    xmax = float(np.max(left)) if len(left) else 0.0
    ax.set_xlim(0, max(1.0, xmax))

# ----------- Bottom-left: Services heatmap -----------
def _compute_service_patients_monthly(months: List[str], monthly_data: Dict[str, Any], dept: str) -> Tuple[List[str], np.ndarray]:
    questions = sorted({q for m in months for q in monthly_data.get(m, {}).get("service", {}).get(dept, {}).keys()})
    if not questions:
        return [], np.zeros((0, len(months)), dtype=float)

    mat = np.zeros((len(questions), len(months)), dtype=float)
    for j, m in enumerate(months):
        svc = monthly_data.get(m, {}).get("service", {}).get(dept, {}) or {}
        den = _total_service_submissions(monthly_data, m, dept) if NORMALIZE_HEATMAPS else 1.0
        scale = (RATE_PER_SUBMISSIONS / den) if (NORMALIZE_HEATMAPS and den > 0) else 1.0
        for i, q in enumerate(questions):
            qdict = svc.get(q, {}) or {}
            pat = 0.0
            for b, c in qdict.items():
                w = _service_bin_patients(b)
                if w is not None:
                    pat += float(c) * float(w)
            mat[i, j] = pat * scale
    return questions, mat

def _plot_service_heatmap(ax, months: List[str], questions: List[str], mat: np.ndarray, title: str):
    if mat.size == 0:
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center")
        ax.set_axis_off()
        return
    im = ax.imshow(mat, aspect="auto", cmap="Blues")
    ax.set_xticks(np.arange(len(months)))
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(questions)))
    ax.set_yticklabels(questions)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"# patients (per {int(RATE_PER_SUBMISSIONS)} submissions)" if NORMALIZE_HEATMAPS else "# patients (estimated)")

# ----------- Bottom-right: Wait heatmap -----------
def _compute_wait_patients_monthly(months: List[str], monthly_data: Dict[str, Any], dept: str) -> Tuple[List[str], np.ndarray]:
    questions = sorted({q for m in months for q in monthly_data.get(m, {}).get("wait", {}).get(dept, {}).keys()})
    if not questions:
        return [], np.zeros((0, len(months)), dtype=float)

    mat = np.zeros((len(questions), len(months)), dtype=float)
    for j, m in enumerate(months):
        wt = monthly_data.get(m, {}).get("wait", {}).get(dept, {}) or {}
        # Scale to patients per RATE_PER_SUBMISSIONS submissions
        den = _total_wait_submissions(monthly_data, m, dept) if NORMALIZE_HEATMAPS else 1.0
        scale = (RATE_PER_SUBMISSIONS / den) if (NORMALIZE_HEATMAPS and den > 0) else 1.0
        for i, q in enumerate(questions):
            qdict = wt.get(q, {}) or {}
            pat = 0.0
            for band, count in qdict.items():
                mid = _wait_band_midpoint(band)
                if mid is not None:
                    # count is total patients reported in this band; midpoint gives % waiting
                    pat += float(count) * mid
            mat[i, j] = pat * scale
    return questions, mat

def _plot_wait_heatmap(ax, months: List[str], questions: List[str], mat: np.ndarray, title: str):
    if mat.size == 0:
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center")
        ax.set_axis_off()
        return
    im = ax.imshow(mat, aspect="auto", cmap="Blues")
    ax.set_xticks(np.arange(len(months)))
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(questions)))
    ax.set_yticklabels(questions)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"# patients waiting (per {int(RATE_PER_SUBMISSIONS)} submissions)" if NORMALIZE_HEATMAPS else "# patients waiting (estimated)")

# ----------- Page assembly: emit FOUR separate figures (wider & shorter) -----------
def page_for_department(
    pdf,
    dept: str,
    months: List[str],
    monthly_data: Dict[str, Any],
    dep_service: Dict[str, Any],
    dep_wait: Dict[str, Any],
    *,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    month_of_analysis: Optional[str] = None,
):
    """
    Render four independent figures (NOT a single 2×2 figure):
      1) Services snapshot (stacked bars)  -> SUBMISSIONS
      2) Wait snapshot                      -> PATIENTS
      3) Services heatmap (monthly)
      4) Wait heatmap (monthly)
    """
    recent = _pick_snapshot_month(months, month_of_analysis) if months else None

    svc_recent = monthly_data.get(recent, {}).get("service", {}).get(dept, {}) if recent else {}
    wt_recent  = monthly_data.get(recent, {}).get("wait", {}).get(dept, {}) if recent else {}

    FIGSIZE = (6.8, 2.9)

    # ---- 1) Services snapshot (SUBMISSIONS)
    fig1, ax1 = plt.subplots(figsize=FIGSIZE)
    _plot_services_stacked_submissions(ax1, svc_recent)
    plt.tight_layout(rect=[0, 0, 0.84, 1.0])
    pdf.savefig(fig1, bbox_inches="tight")
    plt.close(fig1)

    # ---- 2) Wait snapshot (PATIENTS)
    fig2, ax2 = plt.subplots(figsize=FIGSIZE)
    _plot_wait_stacked_patients(ax2, wt_recent)
    plt.tight_layout(rect=[0, 0, 0.84, 1.0])
    pdf.savefig(fig2, bbox_inches="tight")
    plt.close(fig2)

    # ---- 3) Services heatmap (monthly)
    svc_questions, svc_mat = _compute_service_patients_monthly(months, monthly_data, dept)
    fig3, ax3 = plt.subplots(figsize=FIGSIZE)
    _plot_service_heatmap(
        ax3, months, svc_questions, svc_mat,
        title="Monthly Service Request Trends (# patients)"
    )
    plt.tight_layout()
    pdf.savefig(fig3, bbox_inches="tight")
    plt.close(fig3)

    # ---- 4) Wait heatmap (monthly)
    wt_questions, wt_mat = _compute_wait_patients_monthly(months, monthly_data, dept)
    fig4, ax4 = plt.subplots(figsize=FIGSIZE)
    _plot_wait_heatmap(
        ax4, months, wt_questions, wt_mat,
        title="Monthly Patient Wait Time Trends (# patients)"
    )
    plt.tight_layout()
    pdf.savefig(fig4, bbox_inches="tight")
    plt.close(fig4)
