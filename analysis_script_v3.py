#mount + import data
from itertools import chain
import math
import random
import pandas as pd
from pathlib import Path
import numpy as np
from bertopic import BERTopic #ML analysis of qualitative reviews
from tqdm import tqdm 
import sys
import os
from collections import defaultdict #dictionary package

from itertools import chain
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN
#report assembly:
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import Counter

################## USER INPUT HERE ##########################
#input name of excel file
data_name = "Guest Ambassador Rounding Survey.xlsx"
output_pdf_name = "08302025_results.pdf"
df = pd.read_excel(data_name) #importing data 
bootstrap_quant = 1000
process_data = True
bootstrap = True
run_analysis = True

#######################################################
################# CODE################################

#processing data 
if process_data:
     #turn column into numbers
     # --- Insert Name column before index 4 (between Email and f_name/l_name) ---
    if "First Name" in df.columns and "Last Name" in df.columns:
        df.insert(4, "Name", (df["First Name"].fillna("").astype(str) + " " + df["Last Name"].fillna("").astype(str)).str.strip())
    else:
        df.insert(4, "Name", "")

    # --- Ensure num_patients (col index 8) is integer, else 0 ---
    col_patients = "Number of Patients Visited (e.g., 10)"
    if col_patients in df.columns:
        df["num_patients"] = pd.to_numeric(df[col_patients], errors="coerce")
        df["num_patients"] = df["num_patients"].where(df["num_patients"] % 1 == 0, 0)
        df["num_patients"] = df["num_patients"].fillna(0).astype(int)


#common items asked: -- list is just for references - update if original column headers get changed
form_dict = {
    0: "Id",
    1: "Start time",
    2: "Completion time",
    3: "Email",
    4: "Name",
    5: "First Name",
    6: "Last Name",
    7: "Shift Date",
    8: "Number of Patients Visited (e.g., 10)",
    9: "Shift Floor",
    10: "How many patients did you help use the patient portal?",
    11: "What requests did patients make? Please select the supplies/assistance asked for and the quantity. .Blankets",
    12: "What requests did patients make? Please select the supplies/assistance asked for and the quantity. .Water",
    13: "What requests did patients make? Please select the supplies/assistance asked for and the quantity. .Bathing Supplies",
    14: "What requests did patients make? Please select the supplies/assistance asked for and the quantity. .Snacks",
    15: "What requests did patients make? Please select the supplies/assistance asked for and the quantity. .Ice",
    16: "What requests did patients make? Please select the supplies/assistance asked for and the quantity. .Technology Instructions",
    17: "What requests did patients make? Please select the supplies/assistance asked for and the quantity. .Company",
    18: "What requests did patients make? Please select the supplies/assistance asked for and the quantity. .Straightening out the Room",
    19: "What requests did patients make? Please select the supplies/assistance asked for and the quantity. .Moving belongings into reach",
    20: "What percentage of patients feel like they waited to the following services?.Medication",
    21: "What percentage of patients feel like they waited to the following services?.Food",
    22: "What percentage of patients feel like they waited to the following services?.Doctor updates",
    23: "What percentage of patients feel like they waited to the following services?.Call Bell",
    24: "What percentage of patients feel like they waited for the following services?.Bed/Linen",
    25: "Select the most positive event that was shared with you regarding interactions with team members (identifying specific roles if possible)? Leave response blank if none.",
    26: "Select the most negative event that was shared with you regarding interactions with team members (identifying specific roles if possible)? Leave response blank if none."
}

#processed form header dict
processed_form_dict = {
    0: "Id",
    1: "form_start_time",
    2: "form_end_time",
    3: "Email",
    4: "Name",
    5: "f_name",
    6: "l_name",
    7: "date",
    8: "num_patients",
    9: "floor",
    10: "num_patient_portal",
    # Supplies/assistance asked for 
    11: "Blankets",
    12: "Water",
    13: "Bathing Supplies",
    14: "Snacks",
    15: "Ice",
    16: "Technology Instructions",
    17: "Company",
    18: "Straightening out the Room",
    19: "Moving belongings into reach",
    # Wait time for services
    20: "Medication",
    21: "Food",
    22: "Doctor updates",
    23: "Call Bell",
    24: "Bed_linen",
    #most interesting events
    25: "pos_exp",
    26: "neg_exp"
}
#### IMPORTANT: Might need to process the inputs furhter; ie: the departments are still a little messed up notation wise
#processed dataframe headers
df.columns = [processed_form_dict[i] for i in range(len(df.columns))] #changing table headers


##########################################################################################################################################
####################### Review Separation by Department############################################################################################
##########################################################################################################################################

"""
Make dictionaries with department separated reviews.
"""

def _concat_theme_dict(theme_map: dict) -> pd.DataFrame:
    frames = []
    for dept, df in theme_map.items():
        if df is None or getattr(df, "empty", False):
            continue
        dfc = df.copy()
        if "department" not in dfc.columns:
            dfc.insert(0, "department", dept)
        else:
            dfc["department"] = dept
        frames.append(dfc)
    if not frames:
        return pd.DataFrame(columns=["department"]) # empty-safe
    return pd.concat(frames, ignore_index=True)
def as_flat_str_list(docs):
    flat = []
    for d in docs:
        if isinstance(d, list):            # flatten nested lists
            flat.extend(d)
        else:
            flat.append(d)
    cleaned = []
    for d in flat:
        if d is None:
            continue
        if isinstance(d, float) and (math.isnan(d) or math.isinf(d)):
            continue
        s = str(d).strip()
        if s:  # keep non-empty
            cleaned.append(s)
    return cleaned


def cycle_fill(texts, target):
    """
    Uniformly repeat items in `texts` until length == target.
    If target < len(texts), just truncate.
    """
    if not texts or target <= 0:
        return []
    n = len(texts)
    out = []
    while len(out) < target:
        need = target - len(out)
        # if need >= n, take whole list
        if need >= n:
            out.extend(texts)
        else:
            out.extend(texts[:need])
    return out

# 1) Build a mapping from the *original* long headers to your cleaned names
col_map = {
    form_dict[i]: processed_form_dict[i]
    for i in form_dict
    if form_dict[i] in df.columns
}

# 2) Rename in place
df.rename(columns=col_map, inplace=True)

# 3) Sanity‐check: do we still have any unmapped columns?
missing = set(df.columns) - set(processed_form_dict.values())
if missing:
    print("Warning: these columns weren’t remapped:", missing)

department_ind_dict = {}  #{9th floor: {0,2,3,}, 8th floor: {1,4,5}}
for i in range(len(df)):
  if df.iloc[i,9] not in department_ind_dict.keys():
    department_ind_dict[df.iloc[i,9]] = [i]
  else:
    department_ind_dict[df.iloc[i,9]].append(i)

num_departments = len(department_ind_dict)

#######################################################################################################################################
############################################## Service and wait time Dictionary extraction #####################################################################
##########################################################################################################################################

# Correct 3-level nested initialization – stores the {department: {service: {response:frequency}}
# helper to make a `{ item : { response : count }}` dict
def nested_dict():
    return defaultdict(lambda: defaultdict(int))

# monthly_data will look like:
# { "2025-04": {
#       "service": { dept: { service: { response: count } } },
#       "wait":    { dept: { service: { response: count } } }
#   },
#   "2025-05": { … },
#   …
# }
pos_rev_themes  = defaultdict(nested_dict)
neg_rev_themes  = defaultdict(nested_dict)
dep_service = defaultdict(nested_dict)
dep_wait = defaultdict(nested_dict)
monthly_data = defaultdict(lambda: {
    "service": defaultdict(nested_dict),
    "wait":    defaultdict(nested_dict),
})
# now in your loop, grab the month and write into monthly_data[...] as well:
for department, indices in tqdm(department_ind_dict.items(), desc="Processing departments" ):
    tqdm.write(f"Now processing: {department}")
    pos_reviews = []
    neg_reviews = []
    for idx in indices:
        row = df.iloc[idx]
        # extract month as "YYYY-MM"
        month = row["date"].strftime("%Y-%m")

        # ─── SERVICE COUNTS (change if items get added/removed)───────────────────────────────────────────────
        for col in range(11, 20):
            col_name = processed_form_dict[col]
            resp = row.iloc[col]
            if pd.notnull(resp):
                resp = str(resp).strip()
                # month-by-month tallies
                dep_service[department][col_name][resp] += 1
                monthly_data[month]["service"][department][col_name][resp] += 1

        # ─── WAIT TIME RESPONSES #change if more items get added ──────────────────────────────────────────
        num_patients = row.iloc[8]
        for col in range(20, 25):
            col_name = processed_form_dict[col]
            resp = row.iloc[col]
            # if pd.notnull(resp):
            #     npat = 0 
            #     resp = str(resp).strip()
            #     try:
            #         npat = int(num_patients)
            #     except (ValueError, TypeError):
            #         npat = 0
            dep_wait[department][col_name][resp] += num_patients
            monthly_data[month]["wait"][department][col_name][resp] += num_patients

        # ─── POS / NEG REVIEWS (change if more items get added) ───────────────────────────────────────────
        pos_text = row.iloc[25]
        neg_text = row.iloc[26]
        if pd.notnull(pos_text) and str(pos_text).strip():
            pos_reviews.append(str(pos_text).strip())
        if pd.notnull(neg_text) and str(neg_text).strip():
            neg_reviews.append(str(neg_text).strip())

    # (you can still run BERTopic here if you like, 
    #  and store monthly themes analogously in monthly_data[month])

    # At the end, `monthly_data` is populated by month:
    # e.g. monthly_data["2025-04"]["service"]["Cardiology"]["Blankets"]["2"] == how many “2 blankets” responses in April for Cardiology.

        # --------- Run BERTopic on Positive Reviews ---------
    if run_analysis:
        pos_boot_add, neg_boot_add = [], []

        if bootstrap:
            if pos_reviews:
                pos_boot = cycle_fill(pos_reviews, bootstrap_quant)
                pos_boot_add = pos_reviews + pos_boot  # include originals + uniform additions
            if neg_reviews:
                neg_boot = cycle_fill(neg_reviews, bootstrap_quant)
                neg_boot_add = neg_reviews + neg_boot
        else:
            pos_boot_add = list(pos_reviews)
            neg_boot_add = list(neg_reviews)

        pos_boot_add = as_flat_str_list(pos_boot_add)
        neg_boot_add = as_flat_str_list(neg_boot_add)
        
        vectorizer = CountVectorizer(
            ngram_range=(2, 3),        # allow bi+trigrams
            min_df=1,                  # keep rare phrases
            stop_words=None,           # don’t drop stopwords (helps form phrases)
            token_pattern=r"(?u)\b\w+\b"  # keep 1-char tokens if present
        )
        if pos_reviews:
            try:
                pos_model = BERTopic(vectorizer_model=vectorizer)
                pos_model.fit(pos_boot_add)
                pos_rev_themes[department] = pos_model.get_topic_info()

            except Exception as e:
                print(f"[{department}] POS review BERTopic failed: {e}")
                pos_rev_themes[department] = None  # or {} if you want a dict

        if neg_reviews:
            try:
                neg_model = BERTopic(vectorizer_model=vectorizer)
                neg_model.fit(neg_boot_add)
                neg_rev_themes[department] = neg_model.get_topic_info()

            except Exception as e:
                print(f"[{department}] NEG review BERTopic failed: {e}")
                neg_rev_themes[department] = None



############## export ################
        out_dir = Path(".") # change if needed
        pos_df = _concat_theme_dict(pos_rev_themes)
        neg_df = _concat_theme_dict(neg_rev_themes)


        pos_df.to_csv(out_dir / "pos_rev_themes.csv", index=False)
        neg_df.to_csv(out_dir / "neg_rev_themes.csv", index=False)


        print(f"Saved pos_rev_themes.csv with {len(pos_df)} rows")
        print(f"Saved neg_rev_themes.csv with {len(neg_df)} rows")

        

   
    ### BE SURE TO SCALE THEMES BY # Patients visited – VERY APPLICABLE AND IMPORTANT FOR THIS ANALYSIS

##############################################################################################################################
#################################### PDF CREATION ########################################################################
##############################################################################################################################

# print(monthly_data)
from pprint import pprint

# with open("my_dict.txt", "w") as f:
#     pprint(monthly_data, stream=f, indent=4, width=120)
# exit()
#display quantities: 1) number of volunteers, number of reviews per month, 4 pdfs: 2 bar charts, a historical chart, thematic analysis
plt.rcParams["figure.facecolor"] = "whitesmoke"
plt.rcParams["axes.facecolor"]   = "white"
plt.rcParams["axes.prop_cycle"]  = plt.cycler("color", plt.cm.tab10.colors)


def plot_clustered_bars(ax, data: dict, title: str, legend_str: str):

    """
    ax    : a matplotlib Axes
    data  : { question_label: { response_label: frequency, … }, … }
    title : chart title
    """
    if not data or all(len(d)==0 for d in data.values()):
        ax.text(0.5, 0.5, "No data to plot",
                ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        return
    questions = list(data.keys())
    # collect all possible responses so clusters align
    responses = sorted({r for q in questions for r in data[q]})
    n_q, n_r = len(questions), len(responses)

    width = 0.8 / n_r
    x = range(n_q)

    for i, resp in enumerate(responses):
        freqs = [ data[q].get(resp, 0) for q in questions ]
        positions = [ xi + i * width for xi in x ]
        ax.bar(positions, freqs, width, label=resp)

    # center tick labels under clusters
    ax.set_xticks([ xi + width*(n_r-1)/2 for xi in x ])
    ax.set_xticklabels(questions, rotation=45, ha="right")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend(title=legend_str, bbox_to_anchor=(1.02,1), loc="upper left")

# ── make sure your date column is datetime ────────────────────────────────
df['date'] = pd.to_datetime(df['date'])

# ── figure out the most recent month key ─────────────────────────────────
# if monthly_data was built from df['date'].dt.to_period("M").strftime("%Y-%m")
months = sorted(monthly_data.keys())
recent_month = months[-1]

with PdfPages(output_pdf_name) as pdf:
    for dept in department_ind_dict:

        # ─── top row: clustered–bar charts for most recent month ─────────
        recent = months[-1]
        svc_recent = monthly_data[recent]["service"].get(dept, {})
        wt_recent  = monthly_data[recent]["wait"].get(dept,   {})


        # ─── make 2×2 grid ────────────────────────────────────────────
        fig, axs = plt.subplots(2, 2, figsize=(14,10))
        # fig.suptitle(f"{dept} — {recent}", fontsize=18, y=0.98)
        ax1, ax2, ax3, ax4 = axs.flatten()

        #--------------------------------- HEADER---------------------------------------
        header = f"Monthly Report for {recent}: {dept}"
        fig.text(
            0.5, 0.99, header,
            ha="center", va="top",
            fontsize=20, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="skyblue", alpha=0.5)
        )

        plot_clustered_bars(ax1, svc_recent, "Most Commonly Requested Services", "Patients requesting this service per shift")
        plot_clustered_bars(ax2, wt_recent, "Wait Time Report", "% of patients waiting per shift")

        # ─── bottom-left: service trend as “most-freq bin” lines ──────
        # build global category list for this dept’s service bins
        # --- bottom-left: service trend with small vertical offsets -------------------
        delta = .05  # small, visible nudge
        # incremental offsets: 0, +δ, +2δ, ...
        svc_questions = list(dep_service[dept].keys())
        svc_offsets = {q: i * delta for i, q in enumerate(svc_questions)}

        # (optional) symmetric around 0 to reduce upward bias:
        # svc_offsets = {q: (i - (len(svc_questions)-1)/2) * delta for i, q in enumerate(svc_questions)}

        # build global category list for this dept’s service bins
        svc_bins = sorted({
            b
            for m in months
            for qdict in monthly_data[m]["service"].get(dept, {}).values()
            for b in qdict
        })
        svc_map = {b: i for i, b in enumerate(svc_bins)}

        for question in svc_questions:
            y = []
            for m in months:
                qdict = monthly_data[m]["service"].get(dept, {}).get(question, {})
                if qdict:
                    # pick one of the bins with max count
                    max_count = max(qdict.values())
                    top_bins = [b for b, c in qdict.items() if c == max_count]
                    chosen = sorted(top_bins)[0]
                    # add tiny per-series offset so overlapping lines separate
                    y.append(svc_map[chosen] + svc_offsets[question])  # why: visual separation only
                else:
                    y.append(np.nan)
            ax3.plot(months, y, marker="o", label=question)

        ax3.set_yticks(range(len(svc_bins)))
        ax3.set_yticklabels(svc_bins)
        ax3.set_xticks(months)
        ax3.set_xticklabels(months, rotation=45, ha="right")
        ax3.set_ylabel("Service requests per shift")
        ax3.set_title("Monthly Service Request Trends")
        ax3.legend(title="Question", bbox_to_anchor=(1.02, 1), loc="upper left")

        # ensure highest-offset series isn't clipped
        svc_max_off = max(svc_offsets.values(), default=0.0)
        ax3.set_ylim(-0.5, (len(svc_bins) - 1) + svc_max_off + 0.5)

        # --- bottom-right: wait trend with small vertical offsets ---------------------
        wt_questions = list(dep_wait[dept].keys())
        wt_offsets = {q: i * delta for i, q in enumerate(wt_questions)}
        # wt_offsets = {q: (i - (len(wt_questions)-1)/2) * delta for i, q in enumerate(wt_questions)}  # symmetric option

        wt_bins = sorted({
            b
            for m in months
            for qdict in monthly_data[m]["wait"].get(dept, {}).values()
            for b in qdict
        })
        wt_map = {b: i for i, b in enumerate(wt_bins)}

        for question in wt_questions:
            y = []
            for m in months:
                qdict = monthly_data[m]["wait"].get(dept, {}).get(question, {})
                if qdict:
                    max_count = max(qdict.values())
                    top_bins = [b for b, c in qdict.items() if c == max_count]
                    chosen = sorted(top_bins)[0]
                    y.append(wt_map[chosen] + wt_offsets[question])  # why: visual separation only
                else:
                    y.append(np.nan)
            ax4.plot(months, y, marker="o", label=question)

        ax4.set_yticks(range(len(wt_bins)))
        ax4.set_yticklabels(wt_bins)
        ax4.set_xticks(months)
        ax4.set_xticklabels(months, rotation=45, ha="right")
        ax4.set_ylabel("Avg. % of patients waiting per shift")
        ax4.set_title("Monthly Patient Wait Time Trends")
        ax4.legend(title="Question", bbox_to_anchor=(1.02, 1), loc="upper left")

        wt_max_off = max(wt_offsets.values(), default=0.0)
        ax4.set_ylim(-0.5, (len(wt_bins) - 1) + wt_max_off + 0.5)

        # --- finalize and save page ---------------------------------------------------
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 5) (Optional) Thematic analysis
        # if you have pos_rev_themes / neg_rev_themes dicts filled in, e.g. DataFrame with
        # columns ["Name","Count"], you could do:
        # if department in pos_rev_themes:
        #     top5 = pos_rev_themes[department].nlargest(5, "Count")
        #     fig, ax = plt.subplots(figsize=(8, 5))
        #     ax.bar(top5["Name"], top5["Count"])
        #     ax.set_xticklabels(top5["Name"], rotation=45, ha="right")
        #     ax.set_ylabel("Frequency")
        #     ax.set_title(f"{department} — Top Positive Themes")
        #     pdf.savefig(fig, bbox_inches="tight")
        #     plt.close(fig)






