# ga_pipeline/graphs.py
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.facecolor"] = "whitesmoke"
plt.rcParams["axes.facecolor"]   = "white"
plt.rcParams["axes.prop_cycle"]  = plt.cycler("color", plt.cm.tab10.colors)

def plot_clustered_bars(ax, data: dict, title: str, legend_str: str):
    """
    data: { question_label: { response_label: frequency, … }, … }
    """
    if not data or all(len(d) == 0 for d in data.values()):
        ax.text(0.5, 0.5, "No data to plot",
                ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        return

    questions = list(data.keys())
    responses = sorted({r for q in questions for r in data[q]})
    n_q, n_r = len(questions), len(responses)

    width = 0.8 / max(n_r, 1)
    x = np.arange(n_q)

    for i, resp in enumerate(responses):
        freqs = [data[q].get(resp, 0) for q in questions]
        positions = x + i * width
        ax.bar(positions, freqs, width, label=resp)

    ax.set_xticks(x + width * (len(responses) - 1) / 2)
    ax.set_xticklabels(questions, rotation=45, ha="right")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend(title=legend_str, bbox_to_anchor=(1.02, 1), loc="upper left")

def page_for_department(pdf, dept: str,
                        months: List[str],
                        monthly_data: Dict[str, Any],
                        dep_service: Dict[str, Any],
                        dep_wait: Dict[str, Any]):

    recent = months[-1] if months else None
    svc_recent = monthly_data.get(recent, {}).get("service", {}).get(dept, {}) if recent else {}
    wt_recent  = monthly_data.get(recent, {}).get("wait", {}).get(dept, {}) if recent else {}

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axs.flatten()

    header = f"Monthly Report for {recent}: {dept}" if recent else f"Department: {dept}"
    fig.text(0.5, 0.99, header, ha="center", va="top",
             fontsize=20, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="skyblue", alpha=0.5))

    # Top row: bar charts
    plot_clustered_bars(ax1, svc_recent, "Most Commonly Requested Services", "Requests per shift")
    plot_clustered_bars(ax2, wt_recent, "Wait Time Report", "% of patients waiting (per shift)")

    # Bottom-left: service trend (most frequent bin per question)
    delta = 0.05
    svc_questions = list(dep_service.get(dept, {}).keys())
    svc_offsets = {q: i * delta for i, q in enumerate(svc_questions)}
    svc_bins = sorted({
        b
        for m in months
        for qdict in monthly_data.get(m, {}).get("service", {}).get(dept, {}).values()
        for b in qdict
    })
    svc_map = {b: i for i, b in enumerate(svc_bins)}

    for question in svc_questions:
        y = []
        for m in months:
            qdict = monthly_data.get(m, {}).get("service", {}).get(dept, {}).get(question, {})
            if qdict:
                max_count = max(qdict.values())
                top_bins = [b for b, c in qdict.items() if c == max_count]
                chosen = sorted(top_bins)[0]
                y.append(svc_map[chosen] + svc_offsets[question])
            else:
                y.append(np.nan)
        ax3.plot(months, y, marker="o", label=question)

    ax3.set_yticks(range(len(svc_bins)))
    ax3.set_yticklabels(svc_bins)
    ax3.set_xticks(range(len(months)))
    ax3.set_xticklabels(months, rotation=45, ha="right")
    ax3.set_ylabel("Service request bin")
    ax3.set_title("Monthly Service Request Trends")
    ax3.legend(title="Question", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax3.set_ylim(-0.5, (len(svc_bins) - 1) + max(svc_offsets.values(), default=0.0) + 0.5)

    # Bottom-right: wait trend (most frequent bin per question)
    wt_questions = list(dep_wait.get(dept, {}).keys())
    wt_offsets = {q: i * delta for i, q in enumerate(wt_questions)}
    wt_bins = sorted({
        b
        for m in months
        for qdict in monthly_data.get(m, {}).get("wait", {}).get(dept, {}).values()
        for b in qdict
    })
    wt_map = {b: i for i, b in enumerate(wt_bins)}

    for question in wt_questions:
        y = []
        for m in months:
            qdict = monthly_data.get(m, {}).get("wait", {}).get(dept, {}).get(question, {})
            if qdict:
                max_count = max(qdict.values())
                top_bins = [b for b, c in qdict.items() if c == max_count]
                chosen = sorted(top_bins)[0]
                y.append(wt_map[chosen] + wt_offsets[question])
            else:
                y.append(np.nan)
        ax4.plot(months, y, marker="o", label=question)

    ax4.set_yticks(range(len(wt_bins)))
    ax4.set_yticklabels(wt_bins)
    ax4.set_xticks(range(len(months)))
    ax4.set_xticklabels(months, rotation=45, ha="right")
    ax4.set_ylabel("Wait time bin")
    ax4.set_title("Monthly Patient Wait Time Trends")
    ax4.legend(title="Question", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax4.set_ylim(-0.5, (len(wt_bins) - 1) + max(wt_offsets.values(), default=0.0) + 0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
