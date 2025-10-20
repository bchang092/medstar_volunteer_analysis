# ga_pipeline/qualitative.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import math
import json
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

# Gensim (LDA)
try:
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel
    from gensim.utils import simple_preprocess
    from gensim.parsing.preprocessing import STOPWORDS as GENSIM_STOPWORDS
except Exception as e:
    raise ImportError("Please install gensim: pip install gensim") from e


# ---------------- helpers ----------------
def _as_flat_str_list(docs: List[Any]) -> List[str]:
    flat, cleaned = [], []
    for d in docs:
        if isinstance(d, list):
            flat.extend(d)
        else:
            flat.append(d)
    for s in flat:
        if s is None:
            continue
        if isinstance(s, float):
            try:
                if np.isnan(s) or np.isinf(s):
                    continue
            except Exception:
                pass
        s = str(s).strip()
        if s:
            cleaned.append(s)
    return cleaned


def _default_stopwords(extra: Optional[List[str]] = None) -> set:
    # Light domain additions; keep informative words like patient/nurse/care.
    domain = {"hospital", "clinic", "medstar", "hopkins", "department", "dept", "room", "unit"}
    sw = set(GENSIM_STOPWORDS) | domain
    if extra:
        sw |= {w.lower() for w in extra}
    return sw


def _filter_and_extract_texts(
    docs: List[Any],
    date_start: Optional[str],
    date_end: Optional[str],
    keep_undated_when_filtering: bool = True,
) -> List[str]:
    """
    Accept list of plain strings OR dicts {'text': str, 'date': <parseable>}.
    If date_* provided, include items within [start, end]; keep undated if flag True.
    """
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


def _choose_k(n_texts: int, k_max: int = 8) -> int:
    if n_texts <= 3:
        return 1
    k = int(round(math.sqrt(n_texts / 2.0)))
    return max(2, min(k, k_max))


def _prep_tokens(texts: List[str], stopwords: set, bigrams: bool = False) -> List[List[str]]:
    tokens = [
        [t for t in simple_preprocess(doc, deacc=True, min_len=2) if t not in stopwords]
        for doc in texts
    ]
    if bigrams:
        from gensim.models.phrases import Phrases, Phraser
        phrases = Phrases(tokens, min_count=2, threshold=10.0)
        phraser = Phraser(phrases)
        tokens = [phraser[tok] for tok in tokens]
    return tokens


def _corpus_from_tokens(tokens: List[List[str]], no_below: int, no_above: float, keep_n: Optional[int]):
    if not tokens or all(len(t) == 0 for t in tokens):
        return None, None
    dictionary = Dictionary(tokens)
    try:
        dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    except ValueError:
        pass
    if len(dictionary) == 0:
        dictionary = Dictionary(tokens)
    corpus = [dictionary.doc2bow(t) for t in tokens]
    if all(len(bow) == 0 for bow in corpus):
        return None, None
    return dictionary, corpus


def _lda_topics_for_dept(
    texts: List[str],
    *,
    n_topics: int,
    n_top_terms: int,
    random_state: int,
    stopwords: set,
    no_below: int,
    no_above: float,
    keep_n: Optional[int],
    bigrams: bool,
    min_docs: int,
) -> str:
    # Enforce minimum doc count
    if len(texts) < min_docs:
        return ""

    tokens = _prep_tokens(texts, stopwords=stopwords, bigrams=bigrams)

    # Try normal pruning then relax
    for nb, na, kn in [(no_below, no_above, keep_n), (1, 0.95, keep_n), (1, 1.0, None)]:
        dictionary, corpus = _corpus_from_tokens(tokens, nb, na, kn)
        if dictionary is not None and corpus is not None:
            break
    if dictionary is None or corpus is None:
        return ""

    n_docs = len(texts)
    n_terms = len(dictionary)
    k = max(1, min(n_topics, n_docs, n_terms))
    if k == 0:
        return ""

    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        random_state=random_state,
        passes=10,
        iterations=200,
        alpha="auto",
        eta="auto",
        eval_every=None,
    )

    topic_terms = []
    for ti in range(k):
        terms = [w for w, _p in lda.show_topic(ti, topn=n_top_terms)]
        topic_terms.append(terms)

    topic_sizes = [0] * k
    exemplars = [""] * k
    best_prob = [-1.0] * k

    for doc_idx, bow in enumerate(corpus):
        dist = lda.get_document_topics(bow, minimum_probability=0.0)
        if not dist:
            continue
        tids, probs = zip(*dist)
        top_tid = int(np.argmax(probs))
        topic_sizes[top_tid] += 1
        if probs[top_tid] > best_prob[top_tid]:
            best_prob[top_tid] = probs[top_tid]
            ex = texts[doc_idx].replace("\n", " ").strip()
            exemplars[top_tid] = (ex[:217] + "...") if len(ex) > 220 else ex

    parts = []
    for ti in range(k):
        terms = ", ".join(topic_terms[ti])
        size = topic_sizes[ti]
        ex = exemplars[ti]
        parts.append(f"Topic {ti} (n={size}): {terms}" + (f" | exemplar: {ex}" if ex else ""))

    return "  ||  ".join(parts)


# ---------------- public entry point ----------------
def run_qualitative(
    pos_texts_by_dept: Dict[str, List[Any]],
    neg_texts_by_dept: Dict[str, List[Any]],
    bootstrap: bool,
    bootstrap_quant: int,
    output_dir: Path,
    *,
    # NEW: accept the knobs your control.py passes
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    min_docs: int = 5,
    k_max: int = 8,
    n_top_terms: int = 8,
    random_state: int = 42,
    extra_stopwords: Optional[List[str]] = None,
    # kept for signature compatibility with earlier examples (ignored by LDA)
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int = 6000,
    min_df: int = 2,
    max_df: float = 1.0,
    # internal LDA-specific pruning
    no_below: int = 2,
    no_above: float = 0.90,
    keep_n: Optional[int] = 10000,
    bigrams: bool = False,
    keep_undated_when_filtering: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:

    """
    Thematic analysis via Gensim LDA, per department.
    Accepts date filtering and skips departments with < min_docs reviews.
    """

    log = logging.getLogger("qualitative")
    output_dir.mkdir(parents=True, exist_ok=True)

    stopwords = _default_stopwords(extra_stopwords)

    pos_rows: List[Dict[str, Any]] = []
    neg_rows: List[Dict[str, Any]] = []
    pos_summaries_by_dept: Dict[str, str] = {}
    neg_summaries_by_dept: Dict[str, str] = {}

    # Optional: emit a quick preview of counts after filtering
    def _count_after_filter(d: Dict[str, List[Any]]) -> Dict[str, int]:
        out = {}
        for dept, items in d.items():
            texts = _filter_and_extract_texts(items, date_start, date_end, keep_undated_when_filtering)
            out[dept] = len(texts)
        return out

    # Uncomment to debug counts:
    # print("POS counts:", _count_after_filter(pos_texts_by_dept))
    # print("NEG counts:", _count_after_filter(neg_texts_by_dept))

    # Positive
    for dept, raw_docs in tqdm(pos_texts_by_dept.items(), desc="Thematic (positive)", unit="dept"):
        texts = _filter_and_extract_texts(raw_docs, date_start, date_end, keep_undated_when_filtering)
        k = _choose_k(len(texts), k_max=k_max)
        summary = _lda_topics_for_dept(
            texts=texts,
            n_topics=k,
            n_top_terms=n_top_terms,
            random_state=random_state,
            stopwords=stopwords,
            no_below=no_below,
            no_above=no_above,
            keep_n=keep_n,
            bigrams=bigrams,
            min_docs=min_docs,
        )
        method = "gensim_lda" if summary else "skipped_insufficient_docs"
        pos_summaries_by_dept[dept] = summary
        pos_rows.append({"department": dept, "summary": summary, "method": method})

    # Negative
    for dept, raw_docs in tqdm(neg_texts_by_dept.items(), desc="Thematic (negative)", unit="dept"):
        texts = _filter_and_extract_texts(raw_docs, date_start, date_end, keep_undated_when_filtering)
        k = _choose_k(len(texts), k_max=k_max)
        summary = _lda_topics_for_dept(
            texts=texts,
            n_topics=k,
            n_top_terms=n_top_terms,
            random_state=random_state,
            stopwords=stopwords,
            no_below=no_below,
            no_above=no_above,
            keep_n=keep_n,
            bigrams=bigrams,
            min_docs=min_docs,
        )
        method = "gensim_lda" if summary else "skipped_insufficient_docs"
        neg_summaries_by_dept[dept] = summary
        neg_rows.append({"department": dept, "summary": summary, "method": method})

    pos_df = pd.DataFrame(pos_rows, columns=["department", "summary", "method"])
    neg_df = pd.DataFrame(neg_rows, columns=["department", "summary", "method"])

    # Save CSVs (same names as before)
    pos_path = output_dir / "pos_rev_themes.csv"
    neg_path = output_dir / "neg_rev_themes.csv"
    pos_df.to_csv(pos_path, index=False)
    neg_df.to_csv(neg_path, index=False)
    log.info(f"Saved {pos_path} ({len(pos_df)} rows)")
    log.info(f"Saved {neg_path} ({len(neg_df)} rows)")

    return pos_df, neg_df, {
        "pos_summaries_by_dept": pos_summaries_by_dept,
        "neg_summaries_by_dept": neg_summaries_by_dept,
    }
