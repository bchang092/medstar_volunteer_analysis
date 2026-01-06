from typing import Dict, Any, List, Optional, Tuple
import re
import numpy as np
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

SERVICE_BIN_PATIENTS_MAP = {
    "0": 0, "0 patients": 0, "0 patient": 0,
    "1 to 3 patients": 2, "1-3 patients": 2, "1–3 patients": 2, "1 – 3 patients": 2,
    "4 to 6 patients": 5, "4-6 patients": 5, "4–6 patients": 5, "4 – 6 patients": 5,
    "6+ patients": 9, "6 + patients": 9, "6 plus patients": 9, "6 or more patients": 9,
}

WAIT_BAND_MIDPOINT = {
    "0-25%": 0.125, "25-50%": 0.375, "50-75%": 0.625, "75-100%": 0.875,
}

# ----------- Utilities -----------
def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

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
    wt = monthly_data.get(month, {}).get("wait", {}).get(dept, {}) or {}
    return float(sum(_sum_counts(qdict) for qdict in wt.values()))

def _blues(n: int) -> List:
    return [cm.Blues(x) for x in np.linspace(0.35, 0.95, max(n, 1))]

# ----------- Top-left: Services snapshot (stacked, horizontal; submissions) -----------
def _plot_services_stacked_submissions(ax, svc_recent: Dict[str, Dict[str, int]]):
    if not svc_recent:
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center"); ax.set_axis_off(); return

    questions = list(svc_recent.keys())
    bins = list({b for q in questions for b in svc_recent[q].keys()})
    order_hint = ["0", "0 Patients", "1 to 3 Patients", "1-3 Patients", "4 to 6 Patients", "4-6 Patients", "6+ Patients"]
    bins_sorted = [b for b in order_hint if b in bins] + [b for b in bins if b not in order_hint]
    colors = _blues(len(bins_sorted))

    y = np.arange(len(questions))
    left = np.zeros(len(questions), dtype=float)
    for color, b in zip(colors, bins_sorted):
        vals = np.array([float(svc_recent.get(q, {}).get(b, 0)) for q in questions], dtype=float)
        ax.barh(y, vals, left=left, label=b, color=color, edgecolor="white", linewidth=0.5)
        left += vals

    ax.set_yticks(y); ax.set_yticklabels(questions)
    ax.set_xlabel("Number of Volunteer Submissions")
    ax.set_title("Most Commonly Requested Services")
    ax.legend(title="Requests per shift", loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

# ----------- Top-right: Wait snapshot (stacked, horizontal; submissions) -----------
def _plot_wait_stacked_submissions(ax, wt_recent: Dict[str, Dict[str, int]]):
    if not wt_recent:
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center"); ax.set_axis_off(); return

    questions = list(wt_recent.keys())
    bands = list({b for q in questions for b in wt_recent[q].keys()})
    band_order = ["0-25%", "25-50%", "50-75%", "75-100%"]
    bands_sorted = [b for b in band_order if b in bands] + [b for b in bands if b not in band_order]
    colors = _blues(len(bands_sorted))

    y = np.arange(len(questions))
    left = np.zeros(len(questions), dtype=float)
    for color, b in zip(colors, bands_sorted):
        vals = np.array([float(wt_recent.get(q, {}).get(b, 0)) for q in questions], dtype=float)
        ax.barh(y, vals, left=left, label=b, color=color, edgecolor="white", linewidth=0.5)
        left += vals

    ax.set_yticks(y); ax.set_yticklabels(questions)
    ax.set_xlabel("Number of Patients")
    ax.set_title("Wait Time Report")
    ax.legend(title="Wait band", loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

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
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center"); ax.set_axis_off(); return
    im = ax.imshow(mat, aspect="auto", cmap="Blues")
    ax.set_xticks(np.arange(len(months))); ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(questions))); ax.set_yticklabels(questions)
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
        pps = _patients_per_submission(monthly_data, m, dept)
        den = _total_wait_submissions(monthly_data, m, dept) if NORMALIZE_HEATMAPS else 1.0
        scale = (RATE_PER_SUBMISSIONS / den) if (NORMALIZE_HEATMAPS and den > 0) else 1.0
        for i, q in enumerate(questions):
            qdict = wt.get(q, {}) or {}
            pat = 0.0
            for band, count in qdict.items():
                mid = _wait_band_midpoint(band)
                if mid is not None:
                    pat += float(count) * mid * pps
            mat[i, j] = pat * scale
    return questions, mat

def _plot_wait_heatmap(ax, months: List[str], questions: List[str], mat: np.ndarray, title: str):
    if mat.size == 0:
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center"); ax.set_axis_off(); return
    im = ax.imshow(mat, aspect="auto", cmap="Blues")
    ax.set_xticks(np.arange(len(months))); ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(questions))); ax.set_yticklabels(questions)
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
):
    """
    Render four independent figures (NOT a single 2×2 figure):
      1) Services snapshot (stacked bars)
      2) Wait snapshot
      3) Services heatmap
      4) Wait heatmap

    Slightly shorter to help single-page layout.
    """
    recent = months[-1] if months else None
    svc_recent = monthly_data.get(recent, {}).get("service", {}).get(dept, {}) if recent else {}
    wt_recent  = monthly_data.get(recent, {}).get("wait", {}).get(dept, {}) if recent else {}

    # Wider & shorter figures (width, height inches)
    FIGSIZE = (6.8, 2.9)  # ↓ shorter than before

    # ---- 1) Services snapshot
    fig1, ax1 = plt.subplots(figsize=FIGSIZE)
    _plot_services_stacked_submissions(ax1, svc_recent)
    plt.tight_layout(rect=[0, 0, 0.84, 1.0])
    pdf.savefig(fig1, bbox_inches="tight")
    plt.close(fig1)

    # ---- 2) Wait snapshot
    fig2, ax2 = plt.subplots(figsize=FIGSIZE)
    _plot_wait_stacked_submissions(ax2, wt_recent)
    plt.tight_layout(rect=[0, 0, 0.84, 1.0])
    pdf.savefig(fig2, bbox_inches="tight")
    plt.close(fig2)

    # ---- 3) Services heatmap (monthly)
    svc_questions, svc_mat = _compute_service_patients_monthly(months, monthly_data, dept)
    fig3, ax3 = plt.subplots(figsize=FIGSIZE)
    _plot_service_heatmap(ax3, months, svc_questions, svc_mat,
                          title="Monthly Service Request Trends (# patients)")
    plt.tight_layout()
    pdf.savefig(fig3, bbox_inches="tight")
    plt.close(fig3)

    # ---- 4) Wait heatmap (monthly)
    wt_questions, wt_mat = _compute_wait_patients_monthly(months, monthly_data, dept)
    fig4, ax4 = plt.subplots(figsize=FIGSIZE)
    _plot_wait_heatmap(ax4, months, wt_questions, wt_mat,
                       title="Monthly Patient Wait Time Trends (# patients)")
    plt.tight_layout()
    pdf.savefig(fig4, bbox_inches="tight")
    plt.close(fig4)
