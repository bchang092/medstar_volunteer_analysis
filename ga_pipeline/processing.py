# ga_pipeline/processing.py
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, Any, Optional, List
import logging
import re

import numpy as np
import pandas as pd
from tqdm import tqdm


# ----------------------------
# Column header maps (original → processed)
# ----------------------------
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
    26: "Select the most negative event that was shared with you regarding interactions with team members (identifying specific roles if possible)? Leave response blank if none.",
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
    8: "num_patients",          # original numeric-ish column name
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
    26: "neg_exp",
}


# ----------------------------
# Small helpers
# ----------------------------
def _nested_dict():
    """level-2 defaultdict[int] pattern: q -> response -> int count"""
    return defaultdict(lambda: defaultdict(int))

def _as_month_key(ts) -> str:
    if pd.isna(ts):
        return "unknown"
    return pd.to_datetime(ts).to_period("M").strftime("%Y-%m")

def _parse_date_or_none(x: Optional[str]):
    if x is None:
        return None
    try:
        return pd.to_datetime(x).normalize()
    except Exception:
        return None

def extract_first_int(val) -> int:
    """
    Robustly coerce a cell to int:
    - numeric → int(val)
    - string with numbers → first number (e.g., '20 (both wings)' -> 20, '10–12 patients' -> 10)
    - otherwise → 0
    """
    if pd.isna(val):
        return 0
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        try:
            return int(val)
        except Exception:
            pass
    m = re.search(r"\d+", str(val))
    return int(m.group(0)) if m else 0

def _insert_full_name(df: pd.DataFrame) -> pd.DataFrame:
    if "First Name" in df.columns and "Last Name" in df.columns:
        df.insert(
            4,
            "Name",
            (df["First Name"].fillna("").astype(str) + " " + df["Last Name"].fillna("").astype(str)).str.strip(),
        )
    elif "Name" not in df.columns:
        df.insert(4, "Name", "")
    return df

def _fix_num_patients_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a clean integer column 'num_patients_clean'.
    Prefer the survey column 'Number of Patients Visited (e.g., 10)' if present,
    else fall back to 'num_patients' if present.
    """
    # try canonical survey header
    survey_col_candidates: List[str] = [c for c in df.columns if "Number of Patients Visited" in str(c)]
    src_col = survey_col_candidates[0] if survey_col_candidates else None

    if src_col is None and "num_patients" in df.columns:
        src_col = "num_patients"

    if src_col is not None:
        df["num_patients_clean"] = df[src_col].apply(extract_first_int)
    else:
        # If truly nothing to use, create zeros so downstream code stays happy.
        df["num_patients_clean"] = 0

    return df

def _rename_headers(df: pd.DataFrame) -> pd.DataFrame:
    # If the file has the same number/order of columns, set by ordinal:
    if len(df.columns) >= len(PROCESSED_DICT):
        try:
            df.columns = [PROCESSED_DICT.get(i, df.columns[i]) for i in range(len(df.columns))]
        except Exception:
            # fall back to name-based mapping below if shapes mismatch
            pass

    # Name-based mapping from any FORM_DICT names still present to PROCESSED_DICT
    name_map = {FORM_DICT[i]: PROCESSED_DICT[i] for i in FORM_DICT if FORM_DICT[i] in df.columns}
    if name_map:
        df = df.rename(columns=name_map)

    # Normalize some expected keys if they still carry original names
    if "Shift Date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"Shift Date": "date"})
    if "Shift Floor" in df.columns and "floor" not in df.columns:
        df = df.rename(columns={"Shift Floor": "floor"})

    return df

def _build_department_index(df: pd.DataFrame) -> Dict[str, list]:
    dept_idx: Dict[str, list] = {}
    floor_col = "floor" if "floor" in df.columns else ("Shift Floor" if "Shift Floor" in df.columns else None)
    if floor_col is None:
        # No department info; single bucket
        dept_idx["(unknown department)"] = list(range(len(df)))
        return dept_idx

    for i in range(len(df)):
        dept = df.iloc[i][floor_col]
        dept_idx.setdefault(dept, []).append(i)
    return dept_idx


# ----------------------------
# Public entry point
# ----------------------------
def run_processing(
    excel_path: Path,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load the survey Excel, normalize headers, build 'num_patients_clean',
    optionally filter by an inclusive [date_start, date_end] window,
    and aggregate per-department monthly service/wait dictionaries and review texts.

    Returns:
        df (pd.DataFrame): cleaned dataframe (includes 'num_patients_clean', 'date')
        artifacts (dict): {
            'department_ind_dict', 'monthly_data', 'dep_service', 'dep_wait',
            'pos_texts_by_dept', 'neg_texts_by_dept', 'months'
        }
    """
    log = logging.getLogger("processing")
    log.info(f"Loading Excel: {excel_path}")

    df = pd.read_excel(excel_path)

    # Basic canonicalization
    df = _insert_full_name(df)
    df = _rename_headers(df)

    # Ensure date is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        # Create empty date col if missing (prevents downstream KeyErrors)
        df["date"] = pd.NaT

    # Clean num_patients
    df = _fix_num_patients_column(df)

    # ---- Date window filter (inclusive) ----
    start_ts = _parse_date_or_none(date_start)
    end_ts   = _parse_date_or_none(date_end)

    if start_ts is not None or end_ts is not None:
        mask = pd.Series(True, index=df.index)
        if start_ts is not None:
            mask &= (df["date"] >= start_ts)
        if end_ts is not None:
            mask &= (df["date"] <= end_ts)  # inclusive end
        before = len(df)
        df = df[mask].copy()
        log.info(f"Date filter applied: start={start_ts}, end={end_ts} → kept {len(df)}/{before} rows")

    # If nothing left after filtering, return empty artifacts
    if df.empty:
        log.warning("No rows to process after filtering. Returning empty artifacts.")
        return df, {
            "department_ind_dict": {},
            "monthly_data": {},
            "dep_service": {},
            "dep_wait": {},
            "pos_texts_by_dept": {},
            "neg_texts_by_dept": {},
            "months": [],
        }

    # Build department index
    department_ind_dict = _build_department_index(df)
    depts = list(department_ind_dict.keys())
    log.info(f"Found {len(depts)} departments.")

    # Aggregation stores
    dep_service: Dict[str, Any] = defaultdict(_nested_dict)  # dept -> question -> response -> count
    dep_wait: Dict[str, Any]    = defaultdict(_nested_dict)  # dept -> question -> response -> sum(num_patients_clean)
    monthly_data: Dict[str, Any] = defaultdict(
        lambda: {"service": defaultdict(_nested_dict), "wait": defaultdict(_nested_dict)}
    )

    pos_texts_by_dept: Dict[str, List[str]] = defaultdict(list)
    neg_texts_by_dept: Dict[str, List[str]] = defaultdict(list)

    # Main aggregation loop
    for dept in tqdm(depts, desc="Processing departments", unit="dept"):
        indices = department_ind_dict[dept]
        for idx in indices:
            row = df.iloc[idx]
            month = _as_month_key(row.get("date"))

            # patients visited (prefer cleaned)
            num_patients = row.get("num_patients_clean", row.get("num_patients", 0))
            num_patients = extract_first_int(num_patients)

            # --- Services (cols 11..19 in PROCESSED_DICT) ---
            for col in range(11, 20):
                col_name = PROCESSED_DICT[col]
                if col_name in df.columns:
                    resp = row.get(col_name, None)
                    if pd.notnull(resp):
                        resp = str(resp).strip()
                        dep_service[dept][col_name][resp] += 1
                        monthly_data[month]["service"][dept][col_name][resp] += 1

            # --- Wait times (cols 20..24) accumulate affected patients ---
            for col in range(20, 25):
                col_name = PROCESSED_DICT[col]
                if col_name in df.columns:
                    resp = row.get(col_name, None)
                    dep_wait[dept][col_name][resp] += num_patients
                    monthly_data[month]["wait"][dept][col_name][resp] += num_patients

            # --- Qualitative texts ---
            pos_text = row.get("pos_exp", None)
            neg_text = row.get("neg_exp", None)
            if pd.notnull(pos_text) and str(pos_text).strip():
                pos_texts_by_dept[dept].append(str(pos_text).strip())
            if pd.notnull(neg_text) and str(neg_text).strip():
                neg_texts_by_dept[dept].append(str(neg_text).strip())

    months = sorted(monthly_data.keys())
    log.info(f"Aggregated {len(months)} month buckets.")

    artifacts = {
        "department_ind_dict": department_ind_dict,
        "monthly_data": monthly_data,
        "dep_service": dep_service,
        "dep_wait": dep_wait,
        "pos_texts_by_dept": pos_texts_by_dept,
        "neg_texts_by_dept": neg_texts_by_dept,
        "months": months,
    }
    return df, artifacts
