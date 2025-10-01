# ga_pipeline/processing.py
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np
import re

def _nested_dict():
    return defaultdict(lambda: defaultdict(int))

def _as_month_key(ts) -> str:
    return pd.to_datetime(ts).to_period("M").strftime("%Y-%m")

# original -> processed header maps
FORM_DICT = {
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

PROCESSED_DICT = {
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
    11: "Blankets",
    12: "Water",
    13: "Bathing Supplies",
    14: "Snacks",
    15: "Ice",
    16: "Technology Instructions",
    17: "Company",
    18: "Straightening out the Room",
    19: "Moving belongings into reach",
    20: "Medication",
    21: "Food",
    22: "Doctor updates",
    23: "Call Bell",
    24: "Bed_linen",
    25: "pos_exp",
    26: "neg_exp"
}

def _insert_full_name(df: pd.DataFrame) -> pd.DataFrame:
    if "First Name" in df.columns and "Last Name" in df.columns:
        df.insert(4, "Name",
                  (df["First Name"].fillna("").astype(str) + " " +
                   df["Last Name"].fillna("").astype(str)).str.strip())
    else:
        df.insert(4, "Name", "")
    return df

def _fix_num_patients(df: pd.DataFrame) -> pd.DataFrame:
    col_patients = "Number of Patients Visited (e.g., 10)"
    if col_patients in df.columns:
        df["num_patients"] = pd.to_numeric(df[col_patients], errors="coerce")
        df["num_patients"] = df["num_patients"].where(df["num_patients"] % 1 == 0, 0)
        df["num_patients"] = df["num_patients"].fillna(0).astype(int)
    return df

def _rename_headers(df: pd.DataFrame) -> pd.DataFrame:
    # Set to processed headers by ordinal if shape matches
    if len(df.columns) >= len(PROCESSED_DICT):
        df.columns = [PROCESSED_DICT.get(i, df.columns[i]) for i in range(len(df.columns))]
    # Also map any original names still present → processed
    col_map = {FORM_DICT[i]: PROCESSED_DICT[i] for i in FORM_DICT if FORM_DICT[i] in df.columns}
    df = df.rename(columns=col_map)
    return df

def _build_department_index(df: pd.DataFrame) -> Dict[str, list]:
    dept_idx = {}
    for i in range(len(df)):
        dept = df.iloc[i]["floor"]
        dept_idx.setdefault(dept, []).append(i)
    return dept_idx

def extract_first_int(val):
    if pd.isna(val):
        return 0
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        try:
            return int(val)
        except Exception:
            pass
    m = re.search(r"\d+", str(val))
    return int(m.group(0)) if m else 0

def run_processing(excel_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = pd.read_excel(excel_path)
    df = _insert_full_name(df)
    df = _fix_num_patients(df)
    df = _rename_headers(df)

    # Normalize date dtype
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    department_ind_dict = _build_department_index(df)

    # Aggregation stores
    dep_service = defaultdict(_nested_dict)  # dept -> question -> response -> count
    dep_wait    = defaultdict(_nested_dict)  # dept -> question -> response -> sum(num_patients)
    monthly_data = defaultdict(lambda: {"service": defaultdict(_nested_dict),
                                        "wait":    defaultdict(_nested_dict)})

    # Store texts per dept for qualitative step
    pos_texts_by_dept = defaultdict(list)
    neg_texts_by_dept = defaultdict(list)

    for dept, indices in department_ind_dict.items():
        for idx in indices:
            row   = df.iloc[idx]
            month = _as_month_key(row["date"]) if pd.notnull(row["date"]) else "unknown"
            num_patients = extract_first_int(row.get("num_patients", 0))


            # Services (cols 11..19)
            for col in range(11, 20):
                col_name = PROCESSED_DICT[col]
                resp = row.get(col_name, None)
                if pd.notnull(resp):
                    resp = str(resp).strip()
                    dep_service[dept][col_name][resp] += 1
                    monthly_data[month]["service"][dept][col_name][resp] += 1

            # Wait time (cols 20..24) accumulate people affected
            for col in range(20, 24 + 1):
                col_name = PROCESSED_DICT[col]
                resp = row.get(col_name, None)
                # keep raw value as key; you were binning by response string in your plotter
                dep_wait[dept][col_name][resp] += num_patients
                monthly_data[month]["wait"][dept][col_name][resp] += num_patients

            # Qual texts
            pos_text = row.get("pos_exp", None)
            neg_text = row.get("neg_exp", None)
            if pd.notnull(pos_text) and str(pos_text).strip():
                pos_texts_by_dept[dept].append(str(pos_text).strip())
            if pd.notnull(neg_text) and str(neg_text).strip():
                neg_texts_by_dept[dept].append(str(neg_text).strip())

    months = sorted(monthly_data.keys())

    artifacts = {
        "department_ind_dict": department_ind_dict,
        "monthly_data": monthly_data,
        "dep_service": dep_service,
        "dep_wait": dep_wait,
        "pos_texts_by_dept": pos_texts_by_dept,
        "neg_texts_by_dept": neg_texts_by_dept,
        "months": months
    }
    return df, artifacts
