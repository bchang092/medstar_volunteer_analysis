# ga_pipeline/qualitative.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# Parameters for analysis
DEFAULT_MIN_DOCS = 1
DEFAULT_NGRAM_RANGE = (1, 3)
DEFAULT_MAX_FEATURES = 8000
DEFAULT_TOP_PHRASES = 6


def _as_flat_str_list(docs: List[Any]) -> List[str]:
    flat: List[str] = []
    for d in docs or []:
        if isinstance(d, list):
            flat.extend(d)
        else:
            flat.append(d)

    out: List[str] = []
    for s in flat:
        if s is None:
            continue
        s = str(s).strip()
        if s:
            out.append(s)
    return out


def _default_stopwords(extra: Optional[List[str]] = None) -> set:
    domain = {"hospital", "clinic", "medstar", "hopkins", "department", "dept", "room", "unit", "area", "floor"}
    sw = set(ENGLISH_STOP_WORDS) | domain
    if extra:
        sw |= {w.lower() for w in extra}
    return sw


def _filter_and_extract_texts(
    docs: List[Any],
    date_start: Optional[str],
    date_end: Optional[str],
    keep_undated_when_filtering: bool = True,
) -> List[str]:
    if date_start is None and date_end is None:
        return _as_flat_str_list(docs)

    s = pd.to_datetime(date_start) if date_start else None
    e = pd.to_datetime(date_end) if date_end else None

    texts: List[str] = []
    for item in (docs or []):
        if item is None:
            continue
        if isinstance(item, dict):
            t = str(item.get("text", "")).strip()
            if not t:
                continue
            d_raw = item.get("date", None)
            if d_raw is not None:
                try:
                    d = pd.to_datetime(d_raw)
                except Exception:
                    if keep_undated_when_filtering:
                        texts.append(t)
                    continue
                if (s is not None and d < s) or (e is not None and d > e):
                    continue
                texts.append(t)
            else:
                if keep_undated_when_filtering:
                    texts.append(t)
        else:
            t = str(item).strip()
            if t and keep_undated_when_filtering:
                texts.append(t)
    return texts


def _extract_top_phrases(
    docs: List[str],
    *,
    stopwords: set,
    ngram_range: Tuple[int, int],
    max_features: int,
    top_k: int,
) -> List[str]:
    if not docs:
        return []

    vect = TfidfVectorizer(
        lowercase=True,
        stop_words=list(stopwords),
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=1,
        max_df=1.0,
        norm="l2",
    )
    X = vect.fit_transform(docs)
    if X.shape[1] == 0:
        return []

    scores = np.asarray(X.sum(axis=0)).ravel()
    feats = np.array(vect.get_feature_names_out())

    # boost longer phrases slightly
    lengths = np.array([f.count(" ") + 1 for f in feats])
    boost = np.where(lengths >= 3, 1.25, np.where(lengths == 2, 1.15, 1.0))
    scores = scores * boost

    idx = np.argsort(-scores)

    phrases: List[str] = []
    seen = set()
    for i in idx:
        ph = feats[i].strip()
        if len(ph) < 3:
            continue
        if ph in seen:
            continue
        seen.add(ph)
        phrases.append(ph)
        if len(phrases) >= top_k * 2:
            break

    # remove strict substrings of longer phrases
    keep: List[str] = []
    for ph in phrases:
        longer_exists = any((other != ph) and (ph in other) and (len(other) > len(ph)) for other in phrases)
        if longer_exists:
            continue
        keep.append(ph)

    return keep[:top_k]


def _tldr_sentence_from_phrases(phrases: List[str]) -> str:
    """
    Guaranteed sentence output (not a list dump).
    """
    phrases = [p.strip() for p in (phrases or []) if p and p.strip()]
    if not phrases:
        return ""

    core = phrases[:5]
    if len(core) == 1:
        s = f"Feedback primarily centers on {core[0]}."
    elif len(core) == 2:
        s = f"Feedback most often mentions {core[0]} and {core[1]}."
    elif len(core) == 3:
        s = f"Feedback most often mentions {core[0]}, {core[1]}, and {core[2]}."
    else:
        s = (
            f"Feedback most often mentions {core[0]}, {core[1]}, and {core[2]}, "
            f"with additional comments about {core[3]} and {core[4]}."
        )
    return s


def run_qualitative(
    pos_texts_by_dept: Dict[str, List[Any]],
    neg_texts_by_dept: Dict[str, List[Any]],
    output_dir: Path,
    *,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    min_docs: int = DEFAULT_MIN_DOCS,
    ngram_range: Tuple[int, int] = DEFAULT_NGRAM_RANGE,
    max_features: int = DEFAULT_MAX_FEATURES,
    top_k_phrases: int = DEFAULT_TOP_PHRASES,
    extra_stopwords: Optional[List[str]] = None,
    keep_undated_when_filtering: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    TF-IDF-only fallback summaries (no LLMs).
    Returns pos_df / neg_df with columns ['department','summary','method'].
    """
    log = logging.getLogger("qualitative")
    output_dir.mkdir(parents=True, exist_ok=True)

    stopwords = _default_stopwords(extra_stopwords)

    pos_rows: List[Dict[str, Any]] = []
    neg_rows: List[Dict[str, Any]] = []
    pos_summaries_by_dept: Dict[str, str] = {}
    neg_summaries_by_dept: Dict[str, str] = {}

    def summarize_one(texts: List[str]) -> Tuple[str, str]:
        phrases = _extract_top_phrases(
            texts,
            stopwords=stopwords,
            ngram_range=ngram_range,
            max_features=max_features,
            top_k=top_k_phrases,
        )
        sent = _tldr_sentence_from_phrases(phrases)
        return sent, ("tfidf_tldr" if sent else "empty")

    for dept, raw_docs in tqdm(pos_texts_by_dept.items(), desc="Qual TLDR (positive)", unit="dept"):
        texts = _filter_and_extract_texts(raw_docs, date_start, date_end, keep_undated_when_filtering)
        summary, method = summarize_one(texts)
        pos_summaries_by_dept[dept] = summary
        pos_rows.append({"department": dept, "summary": summary, "method": method})
        log.info("[Qual TLDR][positive] dept=%r n_docs=%d method=%s", dept, len(texts), method)

    for dept, raw_docs in tqdm(neg_texts_by_dept.items(), desc="Qual TLDR (negative)", unit="dept"):
        texts = _filter_and_extract_texts(raw_docs, date_start, date_end, keep_undated_when_filtering)
        summary, method = summarize_one(texts)
        neg_summaries_by_dept[dept] = summary
        neg_rows.append({"department": dept, "summary": summary, "method": method})
        log.info("[Qual TLDR][negative] dept=%r n_docs=%d method=%s", dept, len(texts), method)

    pos_df = pd.DataFrame(pos_rows, columns=["department", "summary", "method"])
    neg_df = pd.DataFrame(neg_rows, columns=["department", "summary", "method"])

    pos_df.to_csv(output_dir / "pos_rev_themes.csv", index=False)
    neg_df.to_csv(output_dir / "neg_rev_themes.csv", index=False)

    return pos_df, neg_df, {
        "pos_summaries_by_dept": pos_summaries_by_dept,
        "neg_summaries_by_dept": neg_summaries_by_dept,
    }
