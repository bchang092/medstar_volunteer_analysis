# ga_pipeline/qualitative.py
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import math

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic

def _as_flat_str_list(docs: List[Any]) -> List[str]:
    flat = []
    for d in docs:
        if isinstance(d, list):
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
        if s:
            cleaned.append(s)
    return cleaned

def _cycle_fill(texts: List[str], target: int) -> List[str]:
    if not texts or target <= 0:
        return []
    out = []
    n = len(texts)
    while len(out) < target:
        need = target - len(out)
        if need >= n:
            out.extend(texts)
        else:
            out.extend(texts[:need])
    return out

def _concat_theme_dict(theme_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for dept, df in theme_map.items():
        if df is None or getattr(df, "empty", False):
            continue
        dfc = df.copy()
        dfc.insert(0, "department", dept)
        frames.append(dfc)
    if not frames:
        return pd.DataFrame(columns=["department"])
    return pd.concat(frames, ignore_index=True)

def run_qualitative(
    pos_texts_by_dept: Dict[str, List[str]],
    neg_texts_by_dept: Dict[str, List[str]],
    bootstrap: bool,
    bootstrap_quant: int,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:

    output_dir.mkdir(parents=True, exist_ok=True)

    vectorizer = CountVectorizer(
        ngram_range=(2, 3),
        min_df=1,
        stop_words=None,
        token_pattern=r"(?u)\b\w+\b",
    )

    pos_themes_by_dept = {}
    neg_themes_by_dept = {}

    for dept, texts in pos_texts_by_dept.items():
        docs = _as_flat_str_list(texts)
        if bootstrap and docs:
            docs = docs + _cycle_fill(docs, bootstrap_quant)
        if docs:
            try:
                model = BERTopic(vectorizer_model=vectorizer)
                model.fit(docs)
                pos_themes_by_dept[dept] = model.get_topic_info()
            except Exception as e:
                print(f"[{dept}] POS BERTopic failed: {e}")
                pos_themes_by_dept[dept] = None
        else:
            pos_themes_by_dept[dept] = None

    for dept, texts in neg_texts_by_dept.items():
        docs = _as_flat_str_list(texts)
        if bootstrap and docs:
            docs = docs + _cycle_fill(docs, bootstrap_quant)
        if docs:
            try:
                model = BERTopic(vectorizer_model=vectorizer)
                model.fit(docs)
                neg_themes_by_dept[dept] = model.get_topic_info()
            except Exception as e:
                print(f"[{dept}] NEG BERTopic failed: {e}")
                neg_themes_by_dept[dept] = None
        else:
            neg_themes_by_dept[dept] = None

    pos_df = _concat_theme_dict(pos_themes_by_dept)
    neg_df = _concat_theme_dict(neg_themes_by_dept)

    # Save CSVs
    pos_path = output_dir / "pos_rev_themes.csv"
    neg_path = output_dir / "neg_rev_themes.csv"
    pos_df.to_csv(pos_path, index=False)
    neg_df.to_csv(neg_path, index=False)
    print(f"Saved {pos_path} ({len(pos_df)} rows)")
    print(f"Saved {neg_path} ({len(neg_df)} rows)")

    return pos_df, neg_df, {
        "pos_themes_by_dept": pos_themes_by_dept,
        "neg_themes_by_dept": neg_themes_by_dept,
    }
