# ga_pipeline/qualitative.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import math
import io
import re
import unicodedata
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA  # used for robust fallback & small cohorts

# --- Gensim (LDA) ---
try:
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel
    from gensim.utils import simple_preprocess
    from gensim.parsing.preprocessing import STOPWORDS as GENSIM_STOPWORDS
except Exception as e:
    raise ImportError("Please install gensim: pip install gensim") from e

# --- UMAP (preferred) or PCA fallback ---
try:
    import umap.umap_ as umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False


# ---------------- utilities ----------------
def _safe_slug(s: str, maxlen: int = 96) -> str:
    """
    Turn an arbitrary string into a filesystem-safe slug.
    - Normalize unicode, lower-case
    - Replace anything non [A-Za-z0-9._-] with underscore
    - Collapse repeats and trim length
    """
    if s is None:
        return "unnamed"
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^\w.\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s or "unnamed")[:maxlen]


def _as_flat_str_list(docs: List[Any]) -> List[str]:
    flat, cleaned = [], []
    for d in docs:
        flat.extend(d if isinstance(d, list) else [d])
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
    # Light domain extras; keep content words like patient/nurse/care by default.
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
    Accept list[str] OR list[{'text': str, 'date': <parseable>}].
    If dates provided, include items within [start, end]; keep undated if flag True.
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


def _prep_tokens(texts: List[str], stopwords: set, bigrams: bool = True
                 ) -> Tuple[List[List[str]], Optional[Any]]:
    """
    Tokenize and learn phrases: bigrams then trigrams.
    Returns (tokens, phraser).
    """
    tokens = [
        [t for t in simple_preprocess(doc, deacc=True, min_len=2) if t not in stopwords]
        for doc in texts
    ]
    phraser = None
    if bigrams:
        from gensim.models.phrases import Phrases, Phraser
        # learn bigrams
        bigram = Phrases(tokens, min_count=2, threshold=10.0)
        bigram_phraser = Phraser(bigram)
        tokens = [list(bigram_phraser[tok]) for tok in tokens]
        # learn trigrams
        trigram = Phrases(tokens, min_count=2, threshold=10.0)
        phraser = Phraser(trigram)
        tokens = [list(phraser[tok]) for tok in tokens]
    return tokens, phraser


def _corpus_from_tokens(tokens: List[List[str]], no_below: int, no_above: float, keep_n: Optional[int]):
    if not tokens or all(len(t) == 0 for t in tokens):
        return None, None, None
    dictionary = Dictionary(tokens)
    try:
        dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    except ValueError:
        pass
    if len(dictionary) == 0:
        dictionary = Dictionary(tokens)
    corpus = [dictionary.doc2bow(t) for t in tokens]
    if all(len(bow) == 0 for bow in corpus):
        return None, None, None
    return dictionary, corpus, tokens


def _three_word_labels_from_topic(lda: LdaModel, topic_id: int, topn: int = 18) -> List[str]:
    """
    Build three-word labels for a topic.
    Preference order:
      1) Terms that are trigrams already (contain two underscores)
      2) Otherwise, combine bigram + single
      3) Otherwise, combine three singles in order
    Underscores are rendered as spaces.
    """
    terms = [w for w, _ in lda.show_topic(topic_id, topn=topn)]
    labels, seen = [], set()

    # true trigrams first
    for w in terms:
        if w.count("_") >= 2:
            lbl = w.replace("_", " ")
            if lbl not in seen:
                labels.append(lbl); seen.add(lbl)
        if len(labels) >= 3:
            break

    # bigram + single
    if len(labels) < 3:
        bigrams = [w for w in terms if w.count("_") == 1]
        singles = [w for w in terms if "_" not in w]
        for bg in bigrams:
            for s in singles:
                lbl = f"{bg.replace('_',' ')} {s}"
                if lbl not in seen:
                    labels.append(lbl); seen.add(lbl)
                if len(labels) >= 3:
                    break
            if len(labels) >= 3:
                break

    # three singles
    if len(labels) < 3:
        singles = [w for w in terms if "_" not in w]
        for i in range(max(0, len(singles) - 2)):
            lbl = f"{singles[i]} {singles[i+1]} {singles[i+2]}"
            if lbl not in seen:
                labels.append(lbl); seen.add(lbl)
            if len(labels) >= 3:
                break

    return labels or ["topic theme summary"]


def _legend_label_from_candidates(cands: List[str]) -> str:
    return cands[0]


def _cov_ellipse(X: np.ndarray) -> Tuple[float, float, float]:
    """
    Return ellipse width, height, angle from 2D covariance of X.
    """
    cov = np.cov(X.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    # protect against tiny negative numerical noise
    vals = np.maximum(vals, 1e-12)
    # 2 std dev ~ covers ~95% if Gaussian-ish
    width, height = 2 * 2 * np.sqrt(vals)
    return float(width), float(height), float(theta)


# ---------- robust 2-D embedding (UMAP with safe init; PCA fallback) ----------
def _embed_2d(theta: np.ndarray, n_docs: int, random_state: int):
    """
    Robust 2-D embedding:

    - Handles tiny shapes safely:
        * n_docs == 0  -> empty (0,2)
        * n_docs == 1  -> [[0, 0]]
        * n_features == 0 -> zeros
        * n_features == 1 -> use that 1D as x, zeros for y
    - For small cohorts: PCA with dynamic n_components (1 or 2), pad to 2D if needed.
    - For normal cohorts: try UMAP (init='random'); on any error, fall back to PCA.
    """
    n_docs = int(theta.shape[0])
    n_feat = int(theta.shape[1]) if theta.ndim == 2 else 0

    # --- trivial/degenerate cases ---
    if n_docs == 0:
        return np.zeros((0, 2), dtype=float)
    if n_docs == 1:
        return np.zeros((1, 2), dtype=float)
    if n_feat == 0:
        return np.zeros((n_docs, 2), dtype=float)
    if n_feat == 1:
        x = theta[:, 0]
        # center to avoid all points collapsed in a corner
        x = x - float(x.mean()) if np.std(x) > 0 else np.zeros_like(x)
        return np.column_stack([x, np.zeros(n_docs, dtype=float)])

    # --- helper: safe PCA to 1 or 2 dims, then pad to 2D if needed ---
    def _safe_pca(x: np.ndarray, dims: int) -> np.ndarray:
        m = max(1, min(dims, x.shape[0], x.shape[1]))  # 1 or 2
        xp = PCA(n_components=m, random_state=random_state).fit_transform(x)
        if m == 1:
            return np.column_stack([xp.ravel(), np.zeros(x.shape[0], dtype=float)])
        return xp  # m == 2

    # For very small cohorts, PCA is more stable than UMAP
    if not _HAS_UMAP or n_docs < 5:
        return _safe_pca(theta, 2)

    # --- try UMAP; fall back to PCA on any hiccup ---
    try:
        n_nbrs = max(2, min(10, n_docs - 1))
        reducer = umap.UMAP(
            n_neighbors=n_nbrs,
            n_components=2,
            min_dist=0.1,
            metric="cosine",
            random_state=random_state,
            init="random",  # avoids eigsh(k>=N) spectral init on tiny N
        )
        return reducer.fit_transform(theta)
    except Exception:
        return _safe_pca(theta, 2)


# ---------------- figure builders ----------------
def _topic_map_figure(
    X2d: np.ndarray,
    labels: np.ndarray,
    topic_labels: List[str],
    title: str,
    cmap_name: str = "Blues",
):
    """
    Build a 'datamap' style scatter:
      - background points (light gray)
      - colored clusters
      - soft covariance ellipse per cluster
      - centroid text label (3-word phrase)
    """
    k = int(np.max(labels)) + 1
    cmap = cm.get_cmap(cmap_name, k + 3)
    colors = [cmap(i + 2) for i in range(k)]
    fig, ax = plt.subplots(figsize=(10, 10))

    # faint background
    ax.scatter(X2d[:, 0], X2d[:, 1], s=6, color=(0.6, 0.6, 0.6, 0.25), linewidths=0)

    # clusters + blobs + labels
    for tid in range(k):
        mask = labels == tid
        if not np.any(mask):
            continue
        pts = X2d[mask, :]
        ax.scatter(pts[:, 0], pts[:, 1], s=14, color=colors[tid], alpha=0.85, linewidths=0)

        try:
            w, h, ang = _cov_ellipse(pts)
            cx, cy = pts.mean(axis=0)
            e = Ellipse((cx, cy), width=w, height=h, angle=ang,
                        facecolor=colors[tid], edgecolor="none", alpha=0.18, zorder=0)
            ax.add_patch(e)
        except Exception:
            pass

        cx, cy = pts.mean(axis=0)
        ax.text(cx, cy, topic_labels[tid],
                fontsize=10, fontweight="semibold",
                ha="center", va="center",
                color="black",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))

    ax.set_title(title, fontsize=14, pad=14)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return fig


def _render_topic_map_png(
    X2d: np.ndarray,
    labels: np.ndarray,
    topic_labels: List[str],
    title: str,
    cmap_name: str = "Blues",
) -> bytes:
    """
    Convenience: build figure then return PNG bytes.
    """
    if X2d is None or labels is None or topic_labels is None:
        return b""
    fig = _topic_map_figure(X2d, labels, topic_labels, title, cmap_name=cmap_name)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ---------------- LDA + 2-D topic map per department ----------------
def _lda_and_map_for_dept(
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
) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
    """
    Returns:
      summary_str,
      X2d (num_docs x 2),
      labels (num_docs,),
      topic_labels (list[str] of length K; 3-word phrases)
    """
    if len(texts) < min_docs:
        return "", None, None, None

    tokens, _phraser = _prep_tokens(texts, stopwords=stopwords, bigrams=bigrams)

    # Try normal pruning then relax.
    dictionary = corpus = toks = None
    for nb, na, kn in [(no_below, no_above, keep_n), (1, 0.95, keep_n), (1, 1.0, None)]:
        dictionary, corpus, toks = _corpus_from_tokens(tokens, nb, na, kn)
        if dictionary is not None and corpus is not None:
            break
    if dictionary is None or corpus is None:
        return "", None, None, None

    n_docs = len(texts)
    n_terms = len(dictionary)
    k = max(1, min(n_topics, n_docs, n_terms))

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

    # 3-word topic labels (and longer label lines for summary)
    topic_labels_three: List[str] = []
    topic_label_lines: List[str] = []
    for ti in range(k):
        cands = _three_word_labels_from_topic(lda, ti, topn=max(n_top_terms, 18))
        legend_label = _legend_label_from_candidates(cands)
        topic_labels_three.append(legend_label)
        topic_label_lines.append(", ".join(cands))

    # Per-doc distributions -> dense (n_docs x k)
    theta = np.zeros((n_docs, k), dtype=float)
    for i, bow in enumerate(corpus):
        dist = lda.get_document_topics(bow, minimum_probability=0.0)
        for tid, p in dist:
            theta[i, tid] = p
    labels = np.argmax(theta, axis=1)

    # Robust 2-D map
    X2d = _embed_2d(theta, n_docs=n_docs, random_state=random_state)

    # exemplar + sizes for textual summary
    topic_sizes = [0] * k
    exemplars = [""] * k
    best_prob = [-1.0] * k
    for doc_idx in range(n_docs):
        top_tid = int(labels[doc_idx])
        topic_sizes[top_tid] += 1
        p = theta[doc_idx, top_tid]
        if p > best_prob[top_tid]:
            best_prob[top_tid] = p
            ex = texts[doc_idx].replace("\n", " ").strip()
            exemplars[top_tid] = (ex[:217] + "...") if len(ex) > 220 else ex

    parts = []
    for ti in range(k):
        terms = topic_label_lines[ti]
        size = topic_sizes[ti]
        ex = exemplars[ti]
        parts.append(f"Topic {ti} (n={size}): {terms}" + (f" | exemplar: {ex}" if ex else ""))

    summary = "  ||  ".join(parts)
    # return short labels for plotting
    return summary, X2d, labels, [f"{topic_labels_three[i]}" for i in range(k)]


# ---------------- public entry point ----------------
def run_qualitative(
    pos_texts_by_dept: Dict[str, List[Any]],
    neg_texts_by_dept: Dict[str, List[Any]],
    bootstrap: bool,
    bootstrap_quant: int,
    output_dir: Path,
    *,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    min_docs: int = 5,
    k_max: int = 8,
    n_top_terms: int = 8,
    random_state: int = 42,
    extra_stopwords: Optional[List[str]] = None,
    # ---- LDA pruning knobs ----
    no_below: int = 2,
    no_above: float = 0.90,
    keep_n: Optional[int] = 10000,
    bigrams: bool = True,  # emphasize multiword themes
    keep_undated_when_filtering: bool = True,
    # ---- LEGACY / IGNORED (accepted to avoid TypeError from old control.py) ----
    ngram_range: Optional[tuple[int, int]] = None,
    max_features: Optional[int] = None,
    min_df: Optional[int] = None,
    max_df: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Thematic analysis via Gensim LDA, per department, with optional date filtering.
    Produces PNG topic-map figures (for archival) and also returns live Matplotlib Figure
    objects for direct placement into a PDF.
    """

    log = logging.getLogger("qualitative")
    output_dir.mkdir(parents=True, exist_ok=True)

    stopwords = _default_stopwords(extra_stopwords)

    pos_rows: List[Dict[str, Any]] = []
    neg_rows: List[Dict[str, Any]] = []
    pos_summaries_by_dept: Dict[str, str] = {}
    neg_summaries_by_dept: Dict[str, str] = {}
    pos_figs_by_dept: Dict[str, bytes] = {}
    neg_figs_by_dept: Dict[str, bytes] = {}
    pos_figs_mpl: Dict[str, Any] = {}   # Matplotlib Figure objects
    neg_figs_mpl: Dict[str, Any] = {}   # Matplotlib Figure objects

    # Positive
    for dept, raw_docs in tqdm(pos_texts_by_dept.items(), desc="Thematic (positive)", unit="dept"):
        texts = _filter_and_extract_texts(raw_docs, date_start, date_end, keep_undated_when_filtering)
        k = _choose_k(len(texts), k_max=k_max)
        summary, X2d, labels, topic_labels = _lda_and_map_for_dept(
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

        if X2d is not None:
            # produce fig + bytes
            fig = _topic_map_figure(X2d, labels, topic_labels, title=f"Positive Themes — {dept}", cmap_name="Blues")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
            buf.seek(0)
            pos_figs_by_dept[dept] = buf.getvalue()
            pos_figs_mpl[dept] = fig
            plt.close(fig)

    # Negative
    for dept, raw_docs in tqdm(neg_texts_by_dept.items(), desc="Thematic (negative)", unit="dept"):
        texts = _filter_and_extract_texts(raw_docs, date_start, date_end, keep_undated_when_filtering)
        k = _choose_k(len(texts), k_max=k_max)
        summary, X2d, labels, topic_labels = _lda_and_map_for_dept(
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

        if X2d is not None:
            fig = _topic_map_figure(X2d, labels, topic_labels, title=f"Negative Themes — {dept}", cmap_name="PuBu")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
            buf.seek(0)
            neg_figs_by_dept[dept] = buf.getvalue()
            neg_figs_mpl[dept] = fig
            plt.close(fig)

    pos_df = pd.DataFrame(pos_rows, columns=["department", "summary", "method"])
    neg_df = pd.DataFrame(neg_rows, columns=["department", "summary", "method"])

    # Save CSVs (same names as before)
    pos_path = output_dir / "pos_rev_themes.csv"
    neg_path = output_dir / "neg_rev_themes.csv"
    pos_df.to_csv(pos_path, index=False)
    neg_df.to_csv(neg_path, index=False)
    log.info(f"Saved {pos_path} ({len(pos_df)} rows)")
    log.info(f"Saved {neg_path} ({len(neg_df)} rows)")

    # Optional: write PNGs for sanity check (not used by report.py when using MPL figs)
    fig_dir = output_dir / "qualitative_maps"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for dept, png_bytes in pos_figs_by_dept.items():
        safe = _safe_slug(f"positive_{dept}")
        with open(fig_dir / f"{safe}.png", "wb") as f:
            f.write(png_bytes)
    for dept, png_bytes in neg_figs_by_dept.items():
        safe = _safe_slug(f"negative_{dept}")
        with open(fig_dir / f"{safe}.png", "wb") as f:
            f.write(png_bytes)

    print(f"[qualitative] Saved {len(pos_figs_by_dept)} positive and {len(neg_figs_by_dept)} negative maps to {fig_dir}")

    return pos_df, neg_df, {
        "pos_summaries_by_dept": pos_summaries_by_dept,
        "neg_summaries_by_dept": neg_summaries_by_dept,
        # PNG bytes for archival / optional uses
        "pos_figs_by_dept": pos_figs_by_dept,
        "neg_figs_by_dept": neg_figs_by_dept,
        # LIVE Matplotlib figures for direct PDF placement
        "pos_figs_mpl": pos_figs_mpl,
        "neg_figs_mpl": neg_figs_mpl,
    }
