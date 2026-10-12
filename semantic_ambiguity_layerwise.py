#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
semantic_ambiguity_layerwise.py
===============================
Multiple MEANINGS vs multiple SENSES, homonymy vs polysemy, and word frequency
across the layers of decoder-only LLMs.

RESEARCH QUESTION
  Do LLMs distinguish multiple relatively unrelated meanings from multiple
  related senses, can homonymy-like ambiguity be distinguished from
  polysemy-like structure, and how do these effects and word-frequency effects
  vary across transformer depth?

AUTHORITATIVE SOURCES
  ambiguity_procedure_spec.md   Rodd, Gaskell & Marslen-Wilson (2002) procedure,
                                two-stage design, cautions, data dictionary
  Lexical_Variable.pptx         depth hypothesis, measure choice, Route 2
                                manipulations, screening decisions
  ambiguity_items_combined.csv  stimulus table (2,035 rows, 35 columns)
  ldt_fresh.py                  reused only for model inventory, loading,
                                logit-lens readout verification and statistics

CONSTRUCTS AND THEIR STATUS (see CONSTRUCTS below; printed in REPORT.md)
  multiple meanings     Stage 1: published ambiguity factor (Wordsmyth entries)
                        Stage 2: gcide_etymologies — computed PROXY
  multiple senses       Stage 1: published few/many factor (B, C); Appendix A
                        uses wn_n_all as in the article's regression
                        Stage 2: wn_n_all — computed PROXY
  homonymy vs polysemy  relatedness of a word's interpretations. The supplied
                        files contain NO validated sense-pair or context
                        annotations, so the primary analysis is UNSUPPORTED
                        unless --context_file supplies human-validated
                        meaning_id / sense_id labels; WordNet similarity
                        columns are exploratory proxies only
  word frequency        log_subtlex = log10(SUBTLEX count + 1), used as given

DESIGN
  Stage 1  published Rodd et al. items: Appendix A regression (Exp. 1, visual);
           Appendices B (Exp. 2, visual) and C (Exp. 3, auditory → text here)
           as separate meanings × senses 2 × 2 designs. Never pooled.
  Stage 2  WordNet-derived pool, source-defined central range, per layer
               y_k ~ gcide_etymologies + wn_n_all + log_subtlex + length + concreteness
           a generalisation analysis on proxies, never a replication.

WHAT IS MEASURED
  Logit-lens surprisal (bits) of the target word at every block output:
  log_softmax(W_U · norm_f(h_l)); the final block is the model's actual output
  (verified per model against its own logits); intermediate blocks are
  logit-lens ESTIMATES, not the model's intermediate decisions. Target subtoken
  t is scored from position t−1. In the isolated-item protocol every item
  shares the same prefix, so the FIRST subtoken is scored from a hidden state
  that is identical for all items: its layerwise profile shows where the
  model's expectation for the upcoming word forms, not processing of the word.
  Representations of the word itself are read AT the target positions.
  Surprisal is not a reaction time and no word/nonword decision is simulated
  (the pool has no nonword foils and none are generated).

INFERENCE
  Prespecified confirmatory hypotheses (Stage 1 sign pattern, Stage 2 proxies,
  meanings-vs-senses separation, frequency-vs-ambiguity depth slopes) are
  evaluated on depth-region means and the output layer with a design-preserving
  bootstrap and Holm adjustment, and classified supported / contradicted /
  inconclusive. Layerwise profiles are reported in full as exploratory.

INSTALL
  pip install "torch>=2.2" "transformers>=4.53" "tokenizers>=0.19" accelerate numpy pandas scipy matplotlib
  pip install statsmodels scikit-learn      # optional: reference checks in --self_test
  huggingface-cli login                     # Llama models are gated: accept the licence first
  (optional) pdflatex + booktabs            # lets --self_test compile the LaTeX tables and figures

OUTPUTS (under --output_dir)
  REPORT.md                 every result, limitation and not-run analysis, with reasons
  tables/*.csv              all result tables (machine-readable)
  tables/latex/*.tex        booktabs table* floats that fit the ACL text width (16 cm)
  figures/*.pdf|.png        figures at the ACL text width, embedded TrueType text,
                            300-dpi PNG and a _caption.txt per figure
  item_scores/, cache/, logs/, models/, prompts/   per-item scores, caches, provenance

USAGE
  python semantic_ambiguity_layerwise.py --self_test
  python semantic_ambiguity_layerwise.py --audit_only
  python semantic_ambiguity_layerwise.py --smoke_test --models SmolLM2-360M Qwen2.5-1.5B-Instruct
  python semantic_ambiguity_layerwise.py --models all --dtype bf16 --continue_on_model_error
  python semantic_ambiguity_layerwise.py --resume --models Llama-3.1-8B     # add a model
  python semantic_ambiguity_layerwise.py --analysis_only --resume           # re-run analyses
  python semantic_ambiguity_layerwise.py --list_models
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import shutil
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_NAME = 'semantic_ambiguity_layerwise.py'
SCRIPT_VERSION = '2.0.0'
DEFAULT_DATA_CSV = 'ambiguity_items_combined.csv'
DEFAULT_OUTPUT_DIR = 'semantic_ambiguity_results'
MIN_TRANSFORMERS = (4, 53, 0)          # Qwen3 / SmolLM3 need >= 4.53
LN2 = math.log(2.0)

logger = logging.getLogger('semamb')


# ════════════════════════════════════════════════════════════════════════════
# LAZY HEAVY IMPORTS (audit-only and the statistical self-tests run without torch)
# ════════════════════════════════════════════════════════════════════════════
def _torch():
    import torch
    return torch


def _transformers():
    import transformers
    return transformers


def torch_available() -> bool:
    try:
        _torch()
        _transformers()
        return True
    except Exception:                                                # noqa: BLE001
        return False


def _version_tuple(v: str) -> tuple:
    parts = []
    for p in str(v).split('+')[0].split('.')[:3]:
        digits = ''.join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple((parts + [0, 0, 0])[:3])


# ════════════════════════════════════════════════════════════════════════════
# GENERIC UTILITIES
# ════════════════════════════════════════════════════════════════════════════
def safe_name(name: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in '-._' else '_' for ch in str(name))


def sha256_text(s: str) -> str:
    return hashlib.sha256(str(s).encode('utf-8')).hexdigest()


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def _clean_json(o):
    """Replace NaN/inf by None recursively so the JSON is standard."""
    if isinstance(o, dict):
        return {str(k): _clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_json(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    return o


def sha256_json(obj) -> str:
    return sha256_text(json.dumps(_clean_json(obj), sort_keys=True, default=_json_default))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: str, text: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(text)
    os.replace(tmp, path)


def atomic_write_json(path: str, obj):
    atomic_write_text(path, json.dumps(_clean_json(obj), indent=2, default=_json_default))


def read_json(path: str):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def write_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def seed_everything(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch_available():
        torch = _torch()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CriticalError(RuntimeError):
    """A failure that invalidates the run (non-zero exit)."""


class ChatTemplateError(RuntimeError):
    pass


# ════════════════════════════════════════════════════════════════════════════
# MODEL INVENTORY (from ldt_fresh.py; BASE and INSTRUCT kept separately labelled)
# ════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ModelSpec:
    alias: str
    repo_id: str
    family: str
    variant: str                   # 'base' | 'instruct'
    base_of: str | None = None     # instruct variants: alias of the base counterpart
    revision: str = 'main'
    test_model: str | None = None  # offline random tiny model kind (pipeline tests only)

    @property
    def chat(self) -> bool:
        return self.variant == 'instruct'


MODEL_REGISTRY: list[ModelSpec] = [
    ModelSpec('SmolLM2-360M', 'HuggingFaceTB/SmolLM2-360M', 'smollm', 'base'),
    ModelSpec('SmolLM3-3B-Base', 'HuggingFaceTB/SmolLM3-3B-Base', 'smollm', 'base'),
    ModelSpec('SmolLM2-360M-Instruct', 'HuggingFaceTB/SmolLM2-360M-Instruct', 'smollm', 'instruct',
              'SmolLM2-360M'),
    ModelSpec('Qwen2.5-1.5B', 'Qwen/Qwen2.5-1.5B', 'qwen', 'base'),
    ModelSpec('Qwen2.5-3B', 'Qwen/Qwen2.5-3B', 'qwen', 'base'),
    ModelSpec('Qwen3-4B-Base', 'Qwen/Qwen3-4B-Base', 'qwen', 'base'),
    ModelSpec('Qwen3-8B-Base', 'Qwen/Qwen3-8B-Base', 'qwen', 'base'),
    ModelSpec('Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct', 'qwen', 'instruct', 'Qwen2.5-1.5B'),
    ModelSpec('Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-3B-Instruct', 'qwen', 'instruct', 'Qwen2.5-3B'),
    ModelSpec('Llama-3.2-1B', 'meta-llama/Llama-3.2-1B', 'llama', 'base'),
    ModelSpec('Llama-3.2-3B', 'meta-llama/Llama-3.2-3B', 'llama', 'base'),
    ModelSpec('Llama-3.1-8B', 'meta-llama/Llama-3.1-8B', 'llama', 'base'),
    ModelSpec('Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.2-1B-Instruct', 'llama', 'instruct', 'Llama-3.2-1B'),
    ModelSpec('Llama-3.2-3B-Instruct', 'meta-llama/Llama-3.2-3B-Instruct', 'llama', 'instruct', 'Llama-3.2-3B'),
    ModelSpec('Llama-3.1-8B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct', 'llama', 'instruct', 'Llama-3.1-8B'),
]
# Offline tiny RANDOM-WEIGHT models whose tokenizers reproduce each family's
# pre-tokenisation rules and chat-template style. They exist only to test the
# pipeline without network access; their outputs are never results.
TEST_MODEL_REGISTRY: list[ModelSpec] = [
    ModelSpec('tiny-llama-TEST', 'local/tiny-llama-TEST', 'test', 'base', test_model='llama'),
    ModelSpec('tiny-qwen2-TEST', 'local/tiny-qwen2-TEST', 'test', 'base', test_model='qwen2'),
    ModelSpec('tiny-llama-chat-TEST', 'local/tiny-llama-chat-TEST', 'test', 'instruct',
              'tiny-llama-TEST', test_model='llama_chat'),
    ModelSpec('tiny-qwen2-chat-TEST', 'local/tiny-qwen2-chat-TEST', 'test', 'instruct',
              'tiny-qwen2-TEST', test_model='qwen2_chat'),
]


def resolve_models(tokens: list[str]) -> list[ModelSpec]:
    """Aliases, groups ('all', 'base', 'instruct', 'tests') or repository IDs
    ('org/name' or 'org/name@revision'). The variant of an unregistered repo is
    inferred from its name and logged; append '::base' or '::instruct' to force it."""
    known = {m.alias: m for m in MODEL_REGISTRY + TEST_MODEL_REGISTRY}
    by_repo = {m.repo_id: m for m in MODEL_REGISTRY}
    out: list[ModelSpec] = []
    flat = [t.strip() for tok in tokens for t in str(tok).split(',') if t.strip()]
    for t in flat:
        if t == 'all':
            out += MODEL_REGISTRY
        elif t in ('base', 'instruct'):
            out += [m for m in MODEL_REGISTRY if m.variant == t]
        elif t == 'tests':
            out += TEST_MODEL_REGISTRY
        elif t in known:
            out.append(known[t])
        else:
            variant = None
            if '::' in t:
                t, variant = t.split('::', 1)
                if variant not in ('base', 'instruct'):
                    raise ValueError(f"variant suffix must be ::base or ::instruct, got {variant!r}")
            repo, _, rev = t.partition('@')
            if '/' not in repo:
                raise ValueError(f"unknown model alias {t!r}; see --list_models")
            if repo in by_repo and variant is None:
                m = by_repo[repo]
                out.append(ModelSpec(m.alias if not rev else f"{m.alias}@{rev}", m.repo_id, m.family,
                                     m.variant, m.base_of, rev or 'main'))
                continue
            if variant is None:
                variant = 'instruct' if re.search(r'(instruct|chat|-it\b)', repo, re.I) else 'base'
                logger.warning(f"{repo}: not in the registry; variant inferred from its name as "
                               f"{variant!r} (force with ::base / ::instruct)")
            alias = repo.split('/')[-1] + (f"@{rev}" if rev else '')
            out.append(ModelSpec(alias, repo, repo.split('/')[0].lower(), variant, None, rev or 'main'))
    seen, uniq = set(), []
    for m in out:
        if m.alias not in seen:
            seen.add(m.alias)
            uniq.append(m)
    return uniq


# ════════════════════════════════════════════════════════════════════════════
# STATISTICS (pure numpy / scipy; unit-tested by --self_test)
# ════════════════════════════════════════════════════════════════════════════
def p_two_sided_t(t, df):
    t = np.asarray(t, float)
    with np.errstate(invalid='ignore'):
        return 2.0 * stats.t.sf(np.abs(t), df)


def p_normal(est, se) -> float:
    """Two-sided normal-approximation p from an estimate and its (bootstrap) SE."""
    if not (np.isfinite(est) and np.isfinite(se)) or se <= 0:
        return np.nan
    return float(2.0 * stats.norm.sf(abs(est) / se))


def p_tost_normal(est, se, delta) -> float:
    """Two one-sided tests (Schuirmann, 1987) for |θ| < delta, normal approximation."""
    if not (np.isfinite(est) and np.isfinite(se)) or se <= 0:
        return np.nan
    p_low = stats.norm.sf((est + delta) / se)          # H0: θ <= −delta
    p_up = stats.norm.cdf((est - delta) / se)          # H0: θ >= +delta
    return float(max(p_low, p_up))


def bh_fdr(p) -> np.ndarray:
    """Benjamini–Hochberg (1995) step-up adjusted p-values, NaN-tolerant."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.clip(adj, 0, 1)
    out[ok] = res
    return out


def holm(p) -> np.ndarray:
    """Holm (1979) step-down family-wise adjusted p-values, NaN-tolerant."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    adj = np.maximum.accumulate(pv[order] * (m - np.arange(m)))
    res = np.empty(m)
    res[order] = np.clip(adj, 0, 1)
    out[ok] = res
    return out


def zscore_cols(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, float)
    mu = A.mean(0)
    sd = A.std(0, ddof=1) if A.shape[0] > 1 else np.ones(A.shape[1])
    sd = np.where(sd > 0, sd, np.nan)
    return (A - mu) / sd


def ols_multi(X: np.ndarray, Y: np.ndarray) -> dict:
    """OLS of every column of Y (n x L) on the same design X (n x p, intercept
    included). Classical and HC3 (MacKinnon & White, 1985) standard errors.
    Returns arrays of shape (p, L) plus df and R² (L,)."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    n, p = X.shape
    L = Y.shape[1]
    nan = np.full((p, L), np.nan)
    res = {'beta': nan.copy(), 'se': nan.copy(), 'se_hc3': nan.copy(), 'df': n - p, 'r2': np.full(L, np.nan)}
    if np.linalg.matrix_rank(X) < p or n - p < 1:
        return res
    XtX_inv = np.linalg.inv(X.T @ X)
    B = XtX_inv @ (X.T @ Y)
    E = Y - X @ B
    df = n - p
    rss = (E ** 2).sum(0)
    se = np.sqrt(np.outer(np.diag(XtX_inv), rss / df))
    A = X @ XtX_inv
    h = (A * X).sum(1)
    w = E ** 2 / np.clip(1.0 - h, 1e-12, None)[:, None] ** 2
    se_hc3 = np.sqrt((A ** 2).T @ w)
    tss = ((Y - Y.mean(0)) ** 2).sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        r2 = 1.0 - rss / tss
    res.update({'beta': B, 'se': se, 'se_hc3': se_hc3, 'df': df, 'r2': r2})
    return res


def ols_cluster(X: np.ndarray, Y: np.ndarray, groups) -> dict:
    """OLS with cluster-robust (CR1, Liang & Zeger, 1986) standard errors for
    every column of Y; df = G − 1 (number of clusters minus one)."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    n, p = X.shape
    g_codes, g_index = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    G = len(g_codes)
    L = Y.shape[1]
    res = {'beta': np.full((p, L), np.nan), 'se_cluster': np.full((p, L), np.nan), 'df': G - 1, 'n_clusters': G}
    if np.linalg.matrix_rank(X) < p or G < 2 or n <= p:
        return res
    XtX_inv = np.linalg.inv(X.T @ X)
    B = XtX_inv @ (X.T @ Y)
    E = Y - X @ B
    scale = G / (G - 1) * (n - 1) / (n - p)
    se = np.empty((p, L))
    for l in range(L):
        U = np.zeros((G, p))
        np.add.at(U, g_index, X * E[:, [l]])
        V = XtX_inv @ (U.T @ U) @ XtX_inv * scale
        se[:, l] = np.sqrt(np.diag(V))
    res.update({'beta': B, 'se_cluster': se})
    return res


def ols_beta_only(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, Y, rcond=None)[0]


def vif_table(X: np.ndarray, names: list[str]) -> dict:
    """Variance inflation factors of the predictor columns (no intercept in X)."""
    X = np.asarray(X, float)
    out = {}
    for j, nm in enumerate(names):
        y = X[:, j]
        A = np.column_stack([np.ones(len(y)), np.delete(X, j, axis=1)])
        r = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        tss = ((y - y.mean()) ** 2).sum()
        r2 = 1 - (r @ r) / tss if tss > 0 else np.nan
        out[nm] = float(1.0 / (1.0 - r2)) if np.isfinite(r2) and r2 < 1 else float('inf')
    return out


def partial_corr(x, y, Z) -> tuple[float, float, int]:
    """Pearson correlation of x and y after residualising both on Z (+ intercept);
    returns (r, two-sided p with df = n − 2 − k, n)."""
    x, y, Z = np.asarray(x, float), np.asarray(y, float), np.asarray(Z, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(Z).all(1)
    x, y, Z = x[ok], y[ok], Z[ok]
    n, k = len(x), Z.shape[1]
    if n - 2 - k < 1:
        return np.nan, np.nan, n
    A = np.column_stack([np.ones(n), Z])
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    if rx.std() == 0 or ry.std() == 0:
        return np.nan, np.nan, n
    r = float(np.corrcoef(rx, ry)[0, 1])
    df = n - 2 - k
    t = r * np.sqrt(df / max(1e-15, 1 - r * r))
    return r, float(p_two_sided_t(t, df)), n


def percentile_ci(boot, alpha=0.05, axis=0):
    b = np.asarray(boot, float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return (np.nanpercentile(b, 100 * alpha / 2, axis=axis),
                np.nanpercentile(b, 100 * (1 - alpha / 2), axis=axis))


def auroc_columns(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUROC (Mann–Whitney, average ranks) of each column of `scores`; labels may
    be one vector or one vector per column (permutation nulls)."""
    S = np.asarray(scores, float)
    if S.ndim == 1:
        S = S[:, None]
    Ylab = np.asarray(y)
    if Ylab.ndim == 1:
        Ylab = np.repeat(Ylab[:, None], S.shape[1], axis=1)
    R = stats.rankdata(S, axis=0)
    pos = Ylab == 1
    n_pos = pos.sum(0)
    n_neg = (~pos).sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return ((R * pos).sum(0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def r2_columns(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    P = np.asarray(pred, float)
    Yv = np.asarray(y, float)
    if P.ndim == 1:
        P = P[:, None]
    if Yv.ndim == 1:
        Yv = np.repeat(Yv[:, None], P.shape[1], axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return 1.0 - ((Yv - P) ** 2).sum(0) / ((Yv - Yv.mean(0)) ** 2).sum(0)


def standardized_mean_difference(x_t, x_c, sd_ref=None) -> float:
    """(mean_t − mean_c)/SD; SD = pooled pre-matching SD when given (Austin, 2009)."""
    x_t = np.asarray(x_t, float)
    x_t = x_t[np.isfinite(x_t)]
    x_c = np.asarray(x_c, float)
    x_c = x_c[np.isfinite(x_c)]
    if len(x_t) < 2 or len(x_c) < 2:
        return np.nan
    sd = sd_ref if (sd_ref is not None and np.isfinite(sd_ref)) else \
        np.sqrt((x_t.var(ddof=1) + x_c.var(ddof=1)) / 2)
    return float((x_t.mean() - x_c.mean()) / sd) if sd > 0 else 0.0


def greedy_caliper_match(feat_t, feat_c, caliper_sd: float, pooled_sd):
    """1:1 nearest-neighbour matching without replacement with a per-covariate
    caliper |Δx_j| ≤ caliper·SD_j (Austin, 2011); adapted from ldt_fresh.py."""
    ft, fc = np.asarray(feat_t, float), np.asarray(feat_c, float)
    sd = np.where(np.asarray(pooled_sd, float) > 0, pooled_sd, 1.0)
    zt, zc = ft / sd, fc / sd
    if len(zt) == 0 or len(zc) == 0:
        return np.array([], int), np.array([], int), len(zt)
    D = ((zt[:, None, :] - zc[None, :, :]) ** 2).sum(-1)
    ok = np.all(np.abs(zt[:, None, :] - zc[None, :, :]) <= caliper_sd + 1e-12, axis=-1)
    D = np.where(ok, D, np.inf)
    used = np.zeros(len(zc), bool)
    mt, mc, rejected = [], [], 0
    for i in np.lexsort((np.arange(len(zt)), D.min(1))):
        d = np.where(used, np.inf, D[i])
        j = int(np.argmin(d))
        if not np.isfinite(d[j]):
            rejected += 1
            continue
        used[j] = True
        mt.append(int(i))
        mc.append(j)
    return np.asarray(mt, int), np.asarray(mc, int), rejected


def make_group_folds(groups, k: int, seed: int, labels=None) -> np.ndarray:
    """Fold index per row such that all rows of a group (word) share a fold.
    With binary labels, groups are dealt round-robin within class (stratified).
    Labels must be constant within a group."""
    groups = np.asarray(groups).astype(str)
    rng = np.random.default_rng(seed)
    fold_of = {}
    if labels is not None:
        lab = pd.Series(np.asarray(labels)).groupby(groups)
        if (lab.nunique() > 1).any():
            raise ValueError("labels differ within a group")
        glabel = lab.first()
        for cls in sorted(glabel.unique(), key=str):
            g = np.array(sorted(glabel.index[glabel == cls]))
            for i, gg in enumerate(g[rng.permutation(len(g))]):
                fold_of[gg] = i % k
    else:
        uniq = np.array(sorted(set(groups)))
        for i, gg in enumerate(uniq[rng.permutation(len(uniq))]):
            fold_of[gg] = i % k
    return np.array([fold_of[g] for g in groups], int)


def ridge_oof(X: np.ndarray, Ycols: np.ndarray, folds: np.ndarray, lam_scale: float) -> np.ndarray:
    """Out-of-fold predictions of a linear ridge probe with unpenalised intercept,
    in dual (kernel) form so that many label vectors (observed + permutations)
    share one fit per fold. Standardisation is fitted inside each training fold;
    λ = lam_scale · mean(diag K_train), fixed a priori."""
    X = np.asarray(X, np.float64)
    Ycols = np.asarray(Ycols, np.float64)
    if Ycols.ndim == 1:
        Ycols = Ycols[:, None]
    pred = np.full(Ycols.shape, np.nan)
    for f in np.unique(folds):
        te, tr = folds == f, folds != f
        mu = X[tr].mean(0)
        sd = X[tr].std(0)
        sd = np.where(sd > 0, sd, 1.0)
        Xtr = (X[tr] - mu) / sd
        K = Xtr @ Xtr.T
        lam = max(lam_scale * float(np.trace(K)) / max(1, K.shape[0]), 1e-8)
        ym = Ycols[tr].mean(0)
        A = np.linalg.solve(K + lam * np.eye(K.shape[0]), Ycols[tr] - ym)
        pred[te] = (((X[te] - mu) / sd) @ Xtr.T) @ A + ym
    return pred


# ════════════════════════════════════════════════════════════════════════════
# CONSTRUCTS — what each variable can and cannot stand for
# ════════════════════════════════════════════════════════════════════════════
CONSTRUCTS = [
    {'construct': 'multiple meanings', 'stage': 'Stage 1', 'variable': 'ambiguity',
     'operationalisation': 'published factor: ambiguous = two or more Wordsmyth dictionary entries (Rodd et al., 2002)',
     'status': 'published design factor', 'limitation': 'binary; meaning relatedness ratings of the article not supplied'},
    {'construct': 'multiple meanings', 'stage': 'Stage 2', 'variable': 'gcide_etymologies',
     'operationalisation': 'distinct etymologies across GCIDE (Webster 1913) headword blocks',
     'status': 'computed PROXY',
     'limitation': ('r = .55 with the published factor on Appendices B+C; no citation of its own; 0 = no etymology '
                    'recorded (zero handling configured); century-old etymology; 171 words absent from GCIDE')},
    {'construct': 'multiple senses', 'stage': 'Stage 1', 'variable': 'senses (B, C); wn_n_all (A)',
     'operationalisation': ('published few/many factor in Appendices B and C; Appendix A had no sense manipulation '
                            'and uses the WordNet sense count, as the article\'s Exp. 1 regression did'),
     'status': 'published design factor (B, C); computed WordNet 3.0 analogue of the article\'s count (A)',
     'limitation': 'WordNet 3.0 runs slightly higher than the 1998 counts the article used'},
    {'construct': 'multiple senses', 'stage': 'Stage 2', 'variable': 'wn_n_all',
     'operationalisation': 'WordNet 3.0 synset count, all parts of speech', 'status': 'computed PROXY',
     'limitation': 'r = .67 with the published factor; correlates with frequency; inflated for inflected forms'},
    {'construct': 'homonymy vs polysemy (relatedness of interpretations)', 'stage': 'both',
     'variable': 'none in the supplied files',
     'operationalisation': ('requires human-validated annotations of which usages share a meaning / a sense '
                            '(--context_file with meaning_id and sense_id)'),
     'status': 'UNSUPPORTED without such annotations',
     'limitation': ('sim_top2 / min_sim / mean_sim are WordNet Wu–Palmer similarities (r ≈ 0 with both published '
                    'factors) and ambiguity_class / pool_condition are thresholds on them: exploratory proxies only')},
    {'construct': 'word frequency', 'stage': 'both', 'variable': 'log_subtlex',
     'operationalisation': 'log10(SUBTLEX count + 1), used exactly as supplied (never re-logged)',
     'status': 'corpus norm',
     'limitation': 'Rodd et al. matched on CELEX frequency; subtoken count and length are audited as confounds'},
]
TERM_CONSTRUCT = {'ambiguity': ('multiple meanings', 'published design factor'),
                  'senses': ('multiple senses', 'published design factor'),
                  'ambiguity:senses': ('meanings × senses interaction', 'published design factors'),
                  'wn_n_all': ('multiple senses', 'computed proxy (WordNet sense count)'),
                  'gcide_etymologies': ('multiple meanings', 'computed proxy (GCIDE etymologies)'),
                  'log_subtlex': ('word frequency', 'corpus norm'),
                  'sim_top2': ('sense relatedness', 'unvalidated proxy (exploratory)'),
                  'min_sim': ('sense relatedness', 'unvalidated proxy (exploratory)'),
                  'mean_sim': ('sense relatedness', 'unvalidated proxy (exploratory)'),
                  'n_subtokens': ('tokenisation confound', 'model-specific count'),
                  'length': ('control', 'letters'), 'concreteness': ('control', 'Brysbaert et al. norm'),
                  'ortho_n': ('control', 'computed Coltheart N'), 'n_syll': ('control', 'MRC norm')}


# ════════════════════════════════════════════════════════════════════════════
# DATA: schema, validation, stages, exclusions, audit
# ════════════════════════════════════════════════════════════════════════════
EXPECTED_COLUMNS = [
    'word', 'provenance', 'source_set', 'pool_condition', 'appendix', 'experiment', 'modality',
    'ambiguity', 'senses', 'length', 'n_syll', 'n_phon', 'ortho_n', 'subtlex', 'log_subtlex',
    'gcide_entries', 'gcide_etymologies', 'wn_n_all', 'wn_n_noun', 'wn_n_verb', 'wn_n_adj',
    'wn_n_adv', 'wn_n_pos', 'wn_domains_all', 'wn_domains_noun', 'sim_top2', 'min_sim',
    'mean_sim', 'ambiguity_class', 'concreteness', 'conc_sd', 'imageability', 'familiarity',
    'meaningfulness', 'aoa']
TEXT_COLUMNS = ['word', 'provenance', 'source_set', 'pool_condition', 'appendix', 'modality',
                'ambiguity', 'senses', 'ambiguity_class']
NUMERIC_COLUMNS = [c for c in EXPECTED_COLUMNS if c not in TEXT_COLUMNS]
KEY_PREDICTORS = ['log_subtlex', 'subtlex', 'length', 'concreteness', 'gcide_etymologies',
                  'gcide_entries', 'wn_n_all', 'wn_n_noun', 'ortho_n', 'n_syll', 'n_phon',
                  'sim_top2', 'min_sim', 'mean_sim', 'familiarity', 'imageability',
                  'meaningfulness', 'aoa', 'conc_sd']
BALANCE_COVARIATES = ['log_subtlex', 'length', 'concreteness', 'ortho_n', 'n_syll', 'familiarity',
                      'wn_n_all', 'gcide_etymologies']
APPENDIX_DESIGN = {'A': (1, 'visual'), 'B': (2, 'visual'), 'C': (3, 'auditory')}
STAGE2_RANGE = {'length': (4, 6), 'log_subtlex': (2.0, 4.0)}
GCIDE_ZERO_POLICIES = ('drop', 'recode_to_one', 'retain_as_zero')
# Numbers stated in ambiguity_procedure_spec.md / Lexical_Variable.pptx. They are
# RE-COMPUTED from the CSV and compared, never used as expected results.
SOURCE_STATED = {
    'published_rows': 402, 'published_words': 309, 'derived_rows': 1633,
    'appendix_A_rows': 182, 'appendix_A_ambiguous': 124, 'appendix_A_unambiguous': 58,
    'appendix_B_rows': 128, 'appendix_B_per_cell': 32, 'appendix_C_rows': 92, 'appendix_C_per_cell': 23,
    'derived_with_gcide': 1464, 'stage2_matched_both_predictors': 596,
    'stage2_gcide_distribution': {'1': 359, '2': 129, '3': 34, '4+': 20},
    'derived_in_range_without_gcide_filter': 646, 'in_range_homonym_candidates': 548,
    'in_range_polyseme_candidates': 98, 'in_range_plus_concreteness_3.5_5': 489,
    'collinearity_stage2_pool': {'gcide_etymologies~wn_n_all': 0.33,
                                 'gcide_etymologies~log_subtlex': 0.22,
                                 'wn_n_all~log_subtlex': 0.45},
    'validation_BC': {'r_gcide_meanings': 0.554, 'r_gcide_senses': -0.061,
                      'r_wn_n_all_senses': 0.666, 'r_wn_n_all_meanings': 0.047},
    'gcide_zero_rows_whole_file': 251, 'words_missing_from_gcide': 171,
    'article_exp1_regression_n': 184, 'article_table2_unambiguous_n': 60,
}


def load_and_validate(path: str) -> tuple[pd.DataFrame, dict]:
    """Read the stimulus CSV, validate schema and design labels, add a stable row
    identifier. Raises CriticalError on violations that would invalidate the stages."""
    if not os.path.exists(path):
        raise CriticalError(f"data CSV not found: {path}")
    raw = pd.read_csv(path, dtype={c: str for c in TEXT_COLUMNS}, keep_default_na=True)
    rep = {'path': os.path.abspath(path), 'sha256': sha256_file(path), 'n_rows': int(len(raw)),
           'n_columns': int(raw.shape[1]), 'checks': {}}
    missing = [c for c in EXPECTED_COLUMNS if c not in raw.columns]
    rep['columns_missing'] = missing
    rep['columns_extra'] = [c for c in raw.columns if c not in EXPECTED_COLUMNS]
    if missing:
        raise CriticalError(f"CSV is missing expected columns: {missing}")
    df = raw.copy()
    for c in TEXT_COLUMNS:
        # plain object dtype with NaN for missing, so comparisons give plain bools
        vals = [v.strip() if isinstance(v, str) else v for v in raw[c].astype(object).tolist()]
        df[c] = pd.Series([np.nan if (v is None or (isinstance(v, float) and np.isnan(v)) or v == '') else v
                           for v in vals], index=raw.index, dtype=object)
    df['word'] = df['word'].map(lambda v: v.lower() if isinstance(v, str) else v)
    for c in NUMERIC_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
    ck = rep['checks']
    critical = []

    def check(name, ok, detail='', is_critical=True):
        ck[name] = {'pass': bool(ok), 'detail': detail}
        if is_critical and not ok:
            critical.append(name)

    check('word_lowercase_alpha', bool(df['word'].map(lambda v: isinstance(v, str) and
                                                      re.fullmatch(r'[a-z]+', v) is not None).all()),
          'every word matches ^[a-z]+$')
    prov_vals = set(df['provenance'].dropna().unique())
    check('provenance_values', prov_vals <= {'published', 'wordnet-derived'} and bool(df['provenance'].notna().all()),
          f"values={sorted(prov_vals)}")
    pub = df['provenance'] == 'published'
    der = df['provenance'] == 'wordnet-derived'
    ok_app = bool(df.loc[pub, 'appendix'].isin(list(APPENDIX_DESIGN)).all())
    exp_ok = all(((df.loc[pub & (df['appendix'] == a), 'experiment'] == e).all() and
                  (df.loc[pub & (df['appendix'] == a), 'modality'] == m).all())
                 for a, (e, m) in APPENDIX_DESIGN.items())
    check('published_appendix_experiment_modality', ok_app and exp_ok, 'A→Exp1 visual, B→Exp2 visual, C→Exp3 auditory')
    check('published_ambiguity_values', bool(df.loc[pub, 'ambiguity'].isin(['ambiguous', 'unambiguous']).all()))
    sen_bc = df.loc[pub & df['appendix'].isin(['B', 'C']), 'senses'].isin(['few', 'many']).all()
    sen_a = df.loc[pub & (df['appendix'] == 'A'), 'senses'].isna().all()
    check('published_senses_values', bool(sen_bc and sen_a), 'few/many in B and C; empty in A')
    check('derived_design_columns_empty',
          bool(df.loc[der, ['appendix', 'experiment', 'modality', 'ambiguity', 'senses']].isna().all().all()))
    check('derived_pool_condition_values',
          bool(df.loc[der, 'pool_condition'].isin(['homonym-candidate', 'polyseme-candidate']).all() and
               df.loc[pub, 'pool_condition'].isna().all()), is_critical=False)
    check('length_equals_word_length', bool((df['word'].str.len() == df['length']).all()), is_critical=False)
    d = (np.log10(df['subtlex'] + 1) - df['log_subtlex']).abs()
    max_dev = float(np.nanmax(d.values)) if d.notna().any() else np.nan
    check('log_subtlex_is_log10_subtlex_plus_1', bool(np.isfinite(max_dev) and max_dev < 0.0015),
          f"max |log10(subtlex+1) − log_subtlex| = {max_dev:.5f}; log_subtlex is used AS IS and never re-logged")
    df['row_id'] = df['provenance'].astype(str) + '|' + df['source_set'].astype(str) + '|' + df['word'].astype(str)
    dup = int(df['row_id'].duplicated().sum())
    check('row_id_unique', dup == 0, f"{dup} duplicated provenance|source_set|word keys")
    overlap = set(df.loc[pub, 'word']) & set(df.loc[der, 'word'])
    check('published_derived_words_disjoint', len(overlap) == 0, f"{len(overlap)} shared words", is_critical=False)
    rep['critical_failures'] = critical
    if critical:
        raise CriticalError(f"schema/design validation failed: {critical}")
    return df, rep


def apply_gcide_policy(df: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """gcide_etymologies == 0 means 'in GCIDE, no etymology recorded' (not zero
    meanings). The procedure says: drop or recode to 1, and say which. A blank
    (word absent from GCIDE) is never imputed. Returns (rows, dropped rows)."""
    if policy not in GCIDE_ZERO_POLICIES:
        raise ValueError(f"gcide_zero_policy must be one of {GCIDE_ZERO_POLICIES}")
    out = df.copy()
    zero = out['gcide_etymologies'] == 0
    if policy == 'drop':
        return out[~zero], out[zero]
    if policy == 'recode_to_one':
        out.loc[zero, 'gcide_etymologies'] = 1.0
    return out, out.iloc[0:0]


def load_screening_resources(cfg) -> dict:
    """Optional, user-supplied screening resources. Without them the screens are
    NOT_AVAILABLE: the CSV has no morphological or valence field and none is invented."""
    res = {'inflected': {'status': 'NOT_AVAILABLE',
                         'reason': 'no morphological/lemma field in the CSV and no --screen_inflected_file supplied',
                         'words': set()},
           'emotional_extremes': {'status': 'NOT_AVAILABLE',
                                  'reason': 'no valence field in the CSV and no --screen_valence_file supplied',
                                  'words': set()}}
    if cfg.screen_inflected_file:
        t = pd.read_csv(cfg.screen_inflected_file)
        if not {'word', 'is_inflected'} <= set(t.columns):
            raise CriticalError("--screen_inflected_file needs columns word,is_inflected")
        flag = t['is_inflected'].astype(str).str.lower().isin(['1', 'true', 'yes'])
        res['inflected'] = {'status': 'APPLIED', 'source': os.path.abspath(cfg.screen_inflected_file),
                            'sha256': sha256_file(cfg.screen_inflected_file),
                            'rule': 'exclude words with is_inflected == true (Stage 2); flag only in Stage 1',
                            'words': set(t.loc[flag, 'word'].astype(str).str.lower())}
    if cfg.screen_valence_file:
        t = pd.read_csv(cfg.screen_valence_file)
        if not {'word', 'valence'} <= set(t.columns):
            raise CriticalError("--screen_valence_file needs columns word,valence")
        v = pd.to_numeric(t['valence'], errors='coerce')
        flag = ((v - v.mean()) / v.std(ddof=1)).abs() > cfg.valence_extreme_sd
        res['emotional_extremes'] = {
            'status': 'APPLIED', 'source': os.path.abspath(cfg.screen_valence_file),
            'sha256': sha256_file(cfg.screen_valence_file),
            'rule': (f"exclude |z(valence)| > {cfg.valence_extreme_sd} (CONFIGURATION choice, not prescribed by the "
                     f"source) in Stage 2; flag only in Stage 1"),
            'words': set(t.loc[flag, 'word'].astype(str).str.lower())}
    return res


@dataclass
class Samples:
    stage1: pd.DataFrame             # published rows (appendix membership preserved)
    stage2_base: pd.DataFrame        # derived rows in the central range, BEFORE the zero policy
    stage2: dict                     # policy -> rows after zero policy
    funnel: list
    exclusions: pd.DataFrame         # one row per excluded CSV row, with the step that excluded it
    words_to_score: list
    subset: bool
    screening: dict


def build_samples(df: pd.DataFrame, cfg, screening: dict) -> Samples:
    funnel, excl = [], []

    def drop(frame, keep, stage, step):
        gone = frame[~keep]
        excl.extend({'scope': 'data', 'stage': stage, 'step': step, 'row_id': r, 'word': w}
                    for r, w in zip(gone['row_id'], gone['word']))
        out = frame[keep]
        funnel.append({'stage': stage, 'step': step, 'rows': len(out), 'unique_words': int(out['word'].nunique()),
                       'excluded_rows': len(gone)})
        return out

    pub = df[df['provenance'] == 'published'].copy()
    funnel.append({'stage': 'stage1', 'step': 'provenance == "published"', 'rows': len(pub),
                   'unique_words': int(pub['word'].nunique()), 'excluded_rows': 0})
    for a in 'ABC':
        m = pub['appendix'] == a
        funnel.append({'stage': 'stage1', 'step': f'appendix == {a} (Exp. {APPENDIX_DESIGN[a][0]}, '
                                                  f'{APPENDIX_DESIGN[a][1]})', 'rows': int(m.sum()),
                       'unique_words': int(pub.loc[m, 'word'].nunique()), 'excluded_rows': 0})
    s = df[df['provenance'] == 'wordnet-derived'].copy()
    funnel.append({'stage': 'stage2', 'step': 'provenance == "wordnet-derived"', 'rows': len(s),
                   'unique_words': int(s['word'].nunique()), 'excluded_rows': 0})
    s = drop(s, s['gcide_etymologies'].notna(), 'stage2', 'gcide_etymologies missing (word absent from GCIDE)')
    s = drop(s, s['wn_n_all'].notna(), 'stage2', 'wn_n_all missing')
    s = drop(s, s['length'].between(*STAGE2_RANGE['length']), 'stage2', 'length outside 4–6')
    s = drop(s, s['log_subtlex'].between(*STAGE2_RANGE['log_subtlex']), 'stage2',
             'log_subtlex outside 2.0–4.0 or missing')
    for key in ('inflected', 'emotional_extremes'):
        sc = screening[key]
        if sc['status'] == 'APPLIED':
            s = drop(s, ~s['word'].isin(sc['words']), 'stage2', f'screen {key}')
            pub[f'flag_{key}'] = pub['word'].isin(sc['words'])
        else:
            funnel.append({'stage': 'stage2', 'step': f'screen {key}: NOT_AVAILABLE (nothing excluded)',
                           'rows': len(s), 'unique_words': int(s['word'].nunique()), 'excluded_rows': 0})
    subset = bool(cfg.max_items)
    if subset:
        rng = np.random.default_rng(cfg.seed)

        def pick(words, k):
            words = np.array(sorted(words))
            if len(words) <= k:
                return set(words)
            by_len = sorted(words, key=lambda w: (len(w), w))
            keep = {by_len[0], by_len[-1]}            # shortest and longest for tokenisation diagnostics
            rest = [w for w in words[rng.permutation(len(words))] if w not in keep]
            return keep | set(rest[:max(0, k - len(keep))])
        k1 = max(1, cfg.max_items // 2)
        keep1, keep2 = pick(set(pub['word']), k1), pick(set(s['word']), max(1, cfg.max_items - k1))
        pub = drop(pub, pub['word'].isin(keep1), 'stage1', f'not in --max_items {cfg.max_items} DEBUG subset')
        s = drop(s, s['word'].isin(keep2), 'stage2', f'not in --max_items {cfg.max_items} DEBUG subset')
    stage2 = {}
    for pol in GCIDE_ZERO_POLICIES:
        rows, gone = apply_gcide_policy(s, pol)
        stage2[pol] = rows
        tag = 'PRIMARY (configured)' if pol == cfg.gcide_zero_policy else 'sensitivity'
        funnel.append({'stage': 'stage2', 'step': f'gcide_etymologies == 0 policy = {pol} [{tag}]',
                       'rows': len(rows), 'unique_words': int(rows['word'].nunique()), 'excluded_rows': len(gone)})
        excl.extend({'scope': 'data', 'stage': 'stage2', 'step': f'gcide_etymologies == 0 under policy {pol}',
                     'row_id': r, 'word': w} for r, w in zip(gone['row_id'], gone['word']))
    words = sorted(set(pub['word']) | set(s['word']))
    return Samples(stage1=pub, stage2_base=s.copy(), stage2=stage2, funnel=funnel,
                   exclusions=pd.DataFrame(excl, columns=['scope', 'stage', 'step', 'row_id', 'word']),
                   words_to_score=words, subset=subset, screening=screening)


def _vc(s: pd.Series) -> dict:
    return {('<NA>' if pd.isna(k) else str(k)): int(v) for k, v in s.value_counts(dropna=False).items()}


def _r(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else np.nan


def stage1_balance(pub: pd.DataFrame) -> pd.DataFrame:
    """Covariate balance of the published contrasts (cell means and SMDs)."""
    rows = []
    for app in 'ABC':
        r = pub[pub['appendix'] == app]
        contrasts = [('ambiguity', r['ambiguity'] == 'ambiguous', r['ambiguity'] == 'unambiguous')]
        if app != 'A':
            contrasts.append(('senses', r['senses'] == 'many', r['senses'] == 'few'))
        for c in BALANCE_COVARIATES:
            for name, t, u in contrasts:
                rows.append({'appendix': app, 'contrast': f'{name} (' + ('ambiguous − unambiguous' if name ==
                             'ambiguity' else 'many − few') + ')', 'covariate': c,
                             'mean_level_1': float(r.loc[t, c].mean()), 'mean_level_2': float(r.loc[u, c].mean()),
                             'n_level_1': int(r.loc[t, c].notna().sum()), 'n_level_2': int(r.loc[u, c].notna().sum()),
                             'smd': standardized_mean_difference(r.loc[t, c], r.loc[u, c])})
    return pd.DataFrame(rows)


def homonymy_polysemy_support(df: pd.DataFrame, context_info: dict) -> dict:
    """Is a primary homonymy-vs-polysemy analysis testable with the supplied
    resources? It needs human-validated annotations of which usages of a word
    share a meaning / a sense. Nothing is inferred from etymology or counts."""
    pair_cols = [c for c in df.columns if re.search(r'(sense_pair|same_sense|relatedness_rating|usage)', c)]
    out = {'csv_sense_pair_annotation_columns': pair_cols,
           'csv_proxies_present': ['sim_top2', 'min_sim', 'mean_sim', 'ambiguity_class', 'pool_condition'],
           'proxy_status': 'WordNet-derived, not human-validated: exploratory only'}
    if context_info.get('has_meaning_ids') and context_info.get('all_human_validated'):
        out['status'] = 'SUPPORTED (human-validated meaning_id / sense_id annotations in --context_file)'
    elif context_info.get('has_meaning_ids'):
        out['status'] = ('EXPLORATORY ONLY: --context_file has meaning_id annotations that are not all marked '
                         'validation_status = human-validated')
    else:
        out['status'] = ('UNSUPPORTED: the supplied files contain no validated sense-pair or usage annotations; '
                         'supply --context_file with human-validated meaning_id and sense_id per context')
    return out


def data_audit(df: pd.DataFrame, rep: dict, samples: Samples, cfg) -> dict:
    pub = df[df['provenance'] == 'published']
    der = df[df['provenance'] == 'wordnet-derived']
    audit = {'file': {k: rep[k] for k in ('path', 'sha256', 'n_rows', 'n_columns')},
             'schema': {'columns_missing': rep['columns_missing'], 'columns_extra': rep['columns_extra'],
                        'checks': rep['checks']},
             'constructs': CONSTRUCTS}
    audit['row_counts'] = {c: _vc(df[c]) for c in ('provenance', 'source_set', 'appendix', 'experiment', 'modality',
                                                   'ambiguity', 'senses', 'pool_condition', 'ambiguity_class')}
    cells = (pub.assign(senses=pub['senses'].fillna('NA'))
             .groupby(['appendix', 'experiment', 'modality', 'ambiguity', 'senses']).size().reset_index(name='rows'))
    audit['stage1_cells'] = cells.to_dict('records')
    audit['unique_words'] = {
        'overall': int(df['word'].nunique()),
        'by_provenance': {k: int(v) for k, v in df.groupby('provenance')['word'].nunique().items()},
        'by_source_set': {k: int(v) for k, v in df.groupby('source_set')['word'].nunique().items()}}
    audit['missingness'] = {c: {'all_rows': int(df[c].isna().sum()), 'published': int(pub[c].isna().sum()),
                                'derived': int(der[c].isna().sum()),
                                'stage2_pool': int(samples.stage2_base[c].isna().sum()),
                                'stage2_pool_n': int(len(samples.stage2_base))} for c in KEY_PREDICTORS}
    membership = pub.groupby('word')['appendix'].agg(lambda s: ''.join(sorted(s)))
    in_b, in_c = membership.str.contains('B'), membership.str.contains('C')
    audit['repeated_words'] = {
        'row_id': 'provenance|source_set|word (unique)',
        'published_words_in_multiple_appendices': int((membership.str.len() > 1).sum()),
        'appendix_membership_patterns': {k: int(v) for k, v in membership.value_counts().items()},
        'words_in_both_B_and_C': int((in_b & in_c).sum()),
        'dependence': ('Each appendix is analysed separately (a word occurs once per appendix). Because '
                       f'{int((in_b & in_c).sum())} words occur in both B and C, the B and C results are not '
                       'independent evidence; no test pools or contrasts them. Probes are fitted per appendix '
                       'with word-grouped folds.')}
    z_all = df['gcide_etymologies'] == 0
    z_pool = samples.stage2_base['gcide_etymologies'] == 0
    audit['gcide_etymologies_zero'] = {
        'meaning': 'word present in GCIDE but no entry carries an <ety> field; NOT zero meanings',
        'rows_whole_file': int(z_all.sum()), 'stage2_pool_rows': int(z_pool.sum()),
        'stage2_pool_zero_words': sorted(samples.stage2_base.loc[z_pool, 'word'].tolist()),
        'primary_policy': cfg.gcide_zero_policy,
        'policy_status': ('UNRESOLVED IN SOURCE: the procedure says "drop or recode to 1, and say which" without '
                          'choosing. The primary policy is a configuration choice; all policies are reported.'),
        'n_stage2_by_policy': {p: int(len(r)) for p, r in samples.stage2.items()}}
    audit['screening'] = {k: {kk: vv for kk, vv in v.items() if kk != 'words'} for k, v in samples.screening.items()}
    audit['stage_counts'] = {'funnel': samples.funnel, 'subset_debug_run': samples.subset,
                             'words_to_score': len(samples.words_to_score),
                             'excluded_rows_by_step': {k: int(v) for k, v in
                                                       samples.exclusions['step'].value_counts().items()},
                             'stage1_and_stage2_disjoint': bool(not (set(samples.stage1['word']) &
                                                                    set(samples.stage2_base['word'])))}
    audit['stage1_balance'] = stage1_balance(pub).to_dict('records')
    audit['correlations'] = {
        **{f'appendix_{a}': pub.loc[pub['appendix'] == a, BALANCE_COVARIATES].corr().round(3).to_dict()
           for a in 'ABC'},
        'stage2_pool': samples.stage2[cfg.gcide_zero_policy][BALANCE_COVARIATES].corr().round(3).to_dict()}
    # ── recompute every number stated in the source documents ────────────────
    full_der = der[der['gcide_etymologies'].notna() & der['wn_n_all'].notna()]
    in_range = der[der['length'].between(4, 6) & der['log_subtlex'].between(2.0, 4.0)]
    pool596 = in_range[in_range['gcide_etymologies'].notna() & in_range['wn_n_all'].notna()]
    gdist = pool596['gcide_etymologies'].value_counts()
    bc = pub[pub['appendix'].isin(['B', 'C'])]
    amb = (bc['ambiguity'] == 'ambiguous').astype(float)
    sen = (bc['senses'] == 'many').astype(float)

    def coll(frame):
        return {'gcide_etymologies~wn_n_all': _r(frame['gcide_etymologies'], frame['wn_n_all']),
                'gcide_etymologies~log_subtlex': _r(frame['gcide_etymologies'], frame['log_subtlex']),
                'wn_n_all~log_subtlex': _r(frame['wn_n_all'], frame['log_subtlex'])}
    recomputed = {
        'published_rows': len(pub), 'published_words': int(pub['word'].nunique()), 'derived_rows': len(der),
        'appendix_A_rows': int((pub['appendix'] == 'A').sum()),
        'appendix_A_ambiguous': int(((pub['appendix'] == 'A') & (pub['ambiguity'] == 'ambiguous')).sum()),
        'appendix_A_unambiguous': int(((pub['appendix'] == 'A') & (pub['ambiguity'] == 'unambiguous')).sum()),
        'appendix_B_rows': int((pub['appendix'] == 'B').sum()),
        'appendix_B_per_cell': sorted(set(pub[pub['appendix'] == 'B'].groupby(['ambiguity', 'senses']).size())),
        'appendix_C_rows': int((pub['appendix'] == 'C').sum()),
        'appendix_C_per_cell': sorted(set(pub[pub['appendix'] == 'C'].groupby(['ambiguity', 'senses']).size())),
        'derived_with_gcide': int(der['gcide_etymologies'].notna().sum()),
        'stage2_matched_both_predictors': len(pool596),
        'stage2_gcide_distribution': {'0': int(gdist.get(0.0, 0)), '1': int(gdist.get(1.0, 0)),
                                      '2': int(gdist.get(2.0, 0)), '3': int(gdist.get(3.0, 0)),
                                      '4+': int(gdist[gdist.index >= 4].sum())},
        'derived_in_range_without_gcide_filter': len(in_range),
        'in_range_homonym_candidates': int((in_range['pool_condition'] == 'homonym-candidate').sum()),
        'in_range_polyseme_candidates': int((in_range['pool_condition'] == 'polyseme-candidate').sum()),
        'in_range_plus_concreteness_3.5_5': int(in_range['concreteness'].between(3.5, 5.0).sum()),
        'collinearity_stage2_pool': coll(pool596), 'collinearity_all_derived_with_gcide': coll(full_der),
        'validation_BC': {'r_gcide_meanings': _r(bc['gcide_etymologies'], amb),
                          'r_gcide_senses': _r(bc['gcide_etymologies'], sen),
                          'r_wn_n_all_senses': _r(bc['wn_n_all'], sen), 'r_wn_n_all_meanings': _r(bc['wn_n_all'], amb)},
        'gcide_zero_rows_whole_file': int(z_all.sum()),
        'words_missing_from_gcide': int(df.loc[df['gcide_etymologies'].isna(), 'word'].nunique()),
        'article_exp1_regression_n': int((pub['appendix'] == 'A').sum()),
        'article_table2_unambiguous_n': int(((pub['appendix'] == 'A') & (pub['ambiguity'] == 'unambiguous')).sum())}
    checks = []

    def cmp(key, stated, got, tol):
        if isinstance(stated, dict):
            for k in stated:
                cmp(f"{key}.{k}", stated[k], got.get(k) if isinstance(got, dict) else None, tol)
            return
        match = (got == [stated]) if isinstance(got, list) else \
            (got is not None and np.isfinite(float(got)) and abs(float(got) - float(stated)) <= tol)
        checks.append({'quantity': key, 'stated_in_source': stated, 'recomputed': got, 'match': bool(match)})
    for k, v in SOURCE_STATED.items():
        cmp(k, v, recomputed.get(k), 0.0105 if k == 'collinearity_stage2_pool' else
            (0.0015 if k == 'validation_BC' else 0.0))
    cs, ca = recomputed['collinearity_stage2_pool'], recomputed['collinearity_all_derived_with_gcide']
    notes = []
    if not all(abs(cs[k] - SOURCE_STATED['collinearity_stage2_pool'][k]) <= 0.0105 for k in cs):
        notes.append(
            "The collinearity values stated for the Stage 2 pool (.33/.22/.45) do NOT reproduce on that pool "
            f"(recomputed {cs['gcide_etymologies~wn_n_all']:.2f}/{cs['gcide_etymologies~log_subtlex']:.2f}/"
            f"{cs['wn_n_all~log_subtlex']:.2f}); they reproduce on ALL {len(full_der)} derived words with a GCIDE "
            f"value ({ca['gcide_etymologies~wn_n_all']:.2f}/{ca['gcide_etymologies~log_subtlex']:.2f}/"
            f"{ca['wn_n_all~log_subtlex']:.2f}). In-pool correlations and VIFs are used here.")
    if recomputed['stage2_gcide_distribution']['0'] > 0:
        notes.append(f"The stated gcide distribution of the 596-word pool (359/129/34/20 = 542) omits the "
                     f"{recomputed['stage2_gcide_distribution']['0']} words with gcide_etymologies == 0 that the 596 "
                     f"count includes.")
    notes.append("Appendix A prints 182 items (124/58); the article reports n = 184 for the Exp. 1 regression and "
                 "N = 60 unambiguous in Table 2. The discrepancy is in the original; the CSV's 182 rows are used.")
    audit['source_cross_checks'] = {'comparisons': checks, 'notes': notes, 'recomputed_on_subset': samples.subset}
    vifs = {}
    for pol, rows in samples.stage2.items():
        cols = ['gcide_etymologies', 'wn_n_all', 'log_subtlex', 'length', 'concreteness']
        cc = rows[cols].dropna()
        if len(cc) > len(cols) + 2:
            vifs[pol] = {'n': int(len(cc)), 'vif': vif_table(cc.values, cols)}
    audit['stage2_collinearity'] = vifs
    audit['homonymy_polysemy'] = homonymy_polysemy_support(df, cfg.context_info)
    audit['notes'] = {
        'wn_n_noun_zero_stage2_pool': int((samples.stage2_base['wn_n_noun'] == 0).sum()),
        'modality': 'Appendix C (auditory) items are given to the LLM as text; spoken-word processing is not simulated',
        'nonword_foils': 'NOT_AVAILABLE in the pool and not generated (inventing pseudowords is not allowed)',
        'dominance': 'NOT_AVAILABLE unless --context_file supplies validated dominance labels'}
    return audit


def audit_to_flat(audit: dict) -> pd.DataFrame:
    rows = []

    def walk(prefix, o):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(o, list) and o and isinstance(o[0], dict):
            for i, v in enumerate(o):
                walk(f"{prefix}[{i}]", v)
        else:
            rows.append({'key': prefix, 'value': json.dumps(_clean_json(o), default=_json_default)
                         if isinstance(o, (list, dict)) else o})
    walk('', audit)
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# PROMPT CONDITIONS
# ════════════════════════════════════════════════════════════════════════════
ITEM_SLOT = '{ITEM_LIST}'
TEXT_SLOT = '{TEXT}'
TARGET_SLOT = '{TARGET}'
# Verbatim from ambiguity_procedure_spec.md, "LLM prompt template".
SOURCE_LD_TEMPLATE = (
    "You are a participant in a lexical decision experiment.\n"
    "You will see one letter string at a time.\n"
    "For each one, decide as quickly and accurately as you can\n"
    "whether it is a real English word.\n\n"
    "Respond in this exact format, one line per item:\n"
    "<item> | <WORD or NONWORD> | <ease 1-7>\n\n"
    "\"ease\" is how easy the decision felt: 1 = very hard, 7 = very easy.\n"
    "Do not explain. Do not revise earlier answers.\n\n"
    "Items:\n" + ITEM_SLOT)
# AUTHORED HERE: the slides ask for a relatedness-judgement instruction but give
# no wording. Modelled on Rodd et al.'s 7-point meaning-relatedness pretest; it
# keeps the source prompt's structure, so the text right before the target
# ("Items:\n") is identical and only the task instruction differs.
AUTHORED_RELATEDNESS_TEMPLATE = (
    "You are a participant in a word meaning experiment.\n"
    "You will see one English word at a time.\n"
    "For each one, think about the different meanings the word can have\n"
    "and judge how related those meanings are to each other.\n\n"
    "Respond in this exact format, one line per item:\n"
    "<item> | <relatedness 1-7>\n\n"
    "\"relatedness\" is how related the meanings are: 1 = completely unrelated, 7 = closely related.\n"
    "Do not explain. Do not revise earlier answers.\n\n"
    "Items:\n" + ITEM_SLOT)
# AUTHORED HERE: one neutral frame shared by every context condition AND its
# no-context control (the bare word in the same frame), so a context effect is
# never confounded with a change of task instruction.
CONTEXT_FRAME_TEMPLATE = "Read the following text.\n\nText: " + TEXT_SLOT

# Item-specific labels that must never reach a model input.
LEAK_RE = re.compile(r'\b(ambiguous|unambiguous|homonym\w*|polysem\w*|etymolog\w*|wordnet|synsets?|subtlex|'
                     r'gcide|pool_condition|ambiguity_class|few senses|many senses)\b', re.I)
TEMPLATE_LEAK_RE = re.compile(LEAK_RE.pattern[:-3] + r'|frequency|frequent)\b', re.I)


@dataclass
class PromptCondition:
    name: str
    template: str
    protocol: str            # 'isolated' | 'list' | 'context_bare' | 'context_sentence'
    role: str
    wording_source: str
    status: str              # 'confirmatory' | 'exploratory' | 'secondary'

    @property
    def template_sha256(self) -> str:
        return sha256_text(self.template)


def check_template(name: str, tpl: str, slot: str):
    if tpl.count(slot) != 1 or not tpl.endswith(slot):
        raise CriticalError(f"prompt condition {name!r}: {slot} must occur exactly once and end the template")
    m = TEMPLATE_LEAK_RE.search(tpl.replace(slot, ''))
    if m:
        raise CriticalError(f"prompt condition {name!r} contains an analysis label ({m.group(0)!r}); labels and "
                            f"covariates must never enter model inputs")


def build_conditions(cfg) -> dict[str, PromptCondition]:
    conds = {
        'ld_isolated': PromptCondition(
            'ld_isolated', SOURCE_LD_TEMPLATE, 'isolated', 'primary',
            'ambiguity_procedure_spec.md, LLM prompt template (verbatim); one item per forward pass', 'confirmatory'),
        'relatedness_isolated': PromptCondition(
            'relatedness_isolated', AUTHORED_RELATEDNESS_TEMPLATE, 'isolated', 'task_instruction',
            'AUTHORED HERE (no source wording); modelled on Rodd et al. (2002) relatedness pretest', 'exploratory'),
        'ld_list': PromptCondition(
            'ld_list', SOURCE_LD_TEMPLATE, 'list', 'secondary_list',
            'source template with a randomised multi-item list; words only (the pool has no nonword foils)',
            'secondary'),
    }
    if cfg.task_prompts_json:
        for name, d in read_json(cfg.task_prompts_json).items():
            conds[name] = PromptCondition(name, d['template'], 'isolated', 'task_instruction',
                                          d.get('wording_source', f'user file {cfg.task_prompts_json}'),
                                          'exploratory')
    for c in conds.values():
        check_template(c.name, c.template, ITEM_SLOT)
    if cfg.context_file:
        check_template('context frame', CONTEXT_FRAME_TEMPLATE, TEXT_SLOT)
        status = 'confirmatory-eligible' if cfg.context_info.get('all_human_validated') else 'exploratory'
        for proto in ('context_bare', 'context_sentence'):
            conds[proto] = PromptCondition(
                proto, CONTEXT_FRAME_TEMPLATE, proto, 'context',
                'AUTHORED neutral frame shared by the bare-word control and the user-supplied contexts', status)
    if cfg.primary_condition not in conds or conds[cfg.primary_condition].protocol != 'isolated':
        raise CriticalError(f"--primary_condition {cfg.primary_condition!r} must name an isolated-item condition")
    return conds


def active_conditions(cfg, conds: dict) -> list[str]:
    names = [cfg.primary_condition]
    if cfg.run_prompt_manipulations:
        names += [n for n, c in conds.items() if c.protocol == 'isolated' and n != cfg.primary_condition]
    if cfg.run_list_protocol:
        names += [n for n, c in conds.items() if c.protocol == 'list']
    names += [n for n, c in conds.items() if c.protocol.startswith('context_')]
    return names


def render_items(template: str, words: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Template text with the item list (one item per line, each followed by
    '\\n'); returns the text and each item's character span."""
    buf, spans = template.split(ITEM_SLOT)[0], []
    for w in words:
        spans.append((len(buf), len(buf) + len(w)))
        buf += w + '\n'
    return buf, spans


def load_context_file(path: str) -> tuple[pd.DataFrame, dict]:
    """User-supplied contexts. Required: word, context_id, text (with {TARGET} once).
    Optional annotations: meaning_id, sense_id (which meaning / sense the context
    expresses), dominance (dominant/subordinate), frame_id (shared syntactic
    frame), source, validation_status. Nothing here is generated or inferred."""
    t = pd.read_csv(path, dtype=str).fillna('')
    need = {'word', 'context_id', 'text'}
    if not need <= set(t.columns):
        raise CriticalError(f"--context_file needs columns {sorted(need)}")
    t['word'] = t['word'].str.strip().str.lower()
    if not t['word'].str.fullmatch(r'[a-z]+').all():
        raise CriticalError("--context_file: every word must be lowercase alphabetic")
    bad = t[t['text'].str.count(re.escape(TARGET_SLOT)) != 1]
    if len(bad):
        raise CriticalError(f"--context_file: {len(bad)} rows do not contain {TARGET_SLOT} exactly once")
    leak = t[t['text'].map(lambda x: bool(LEAK_RE.search(x)))]
    if len(leak):
        raise CriticalError(f"--context_file: {len(leak)} contexts contain analysis labels "
                            f"(e.g. {leak['text'].iloc[0]!r})")
    if t.duplicated(['word', 'context_id']).any():
        raise CriticalError("--context_file: (word, context_id) must be unique")
    for c in ('meaning_id', 'sense_id', 'dominance', 'frame_id', 'source', 'validation_status'):
        if c not in t.columns:
            t[c] = ''
    pre = t['text'].str.split(re.escape(TARGET_SLOT), regex=True).str[0]
    t['n_pre_target_words'] = pre.map(lambda s: len(re.findall(r'[A-Za-z]+', s)))
    t['n_words'] = t['text'].str.replace(TARGET_SLOT, 'x', regex=False).map(lambda s: len(re.findall(r'[A-Za-z]+', s)))
    multi = t.groupby('word')['context_id'].transform('size') >= 2
    info = {'path': os.path.abspath(path), 'sha256': sha256_file(path), 'n_contexts': int(len(t)),
            'n_words': int(t['word'].nunique()), 'n_words_with_2plus_contexts': int(t.loc[multi, 'word'].nunique()),
            'has_meaning_ids': bool(len(t)) and bool((t.loc[multi, 'meaning_id'] != '').all()) and bool(multi.any()),
            'has_sense_ids': bool(len(t)) and bool((t.loc[multi, 'sense_id'] != '').all()) and bool(multi.any()),
            'has_dominance_labels': bool(t['dominance'].str.lower().isin(['dominant', 'subordinate']).any()),
            'has_frame_ids': bool((t['frame_id'] != '').any()),
            'all_human_validated': bool(len(t)) and bool((t['validation_status'].str.lower() == 'human-validated').all()),
            'n_contexts_without_pre_target_words': int((t['n_pre_target_words'] == 0).sum())}
    return t, info


# ════════════════════════════════════════════════════════════════════════════
# SCORING UNITS (one forward pass each) AND TARGET ALIGNMENT
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Unit:
    unit_id: str
    text: str                     # BASE: whole input; INSTRUCT: user-message content
    targets: list                 # [(word, char_start, char_end)] in `text`
    meta: list                    # per target: dict


@dataclass
class EncodedUnit:
    unit_id: str
    input_ids: list
    full_text: str                # exact text that was tokenised (rendered chat for INSTRUCT)
    targets: list                 # per target: alignment dict (see align_target)


def build_units(cond: PromptCondition, cfg, samples: Samples, context_df: pd.DataFrame | None) -> list[Unit]:
    units = []
    if cond.protocol == 'isolated':
        for w in samples.words_to_score:
            text, spans = render_items(cond.template, [w])
            units.append(Unit(f"{cond.name}|{w}", text, [(w, *spans[0])], [{'word': w}]))
    elif cond.protocol == 'list':
        # Stage 1 and Stage 2 words never share a list (the stages are never mixed).
        pools = {'stage1': sorted(set(samples.stage1['word'])), 'stage2': sorted(set(samples.stage2_base['word']))}
        for rep in range(cfg.list_repeats):
            for pname, pw in pools.items():
                rng = np.random.default_rng([cfg.seed, rep, 0 if pname == 'stage1' else 1])
                order = [pw[i] for i in rng.permutation(len(pw))]
                for li in range(0, len(order), cfg.list_size):
                    chunk = order[li:li + cfg.list_size]
                    text, spans = render_items(cond.template, chunk)
                    units.append(Unit(f"{cond.name}|r{rep}|{pname}|{li // cfg.list_size}", text,
                                      [(w, *sp) for w, sp in zip(chunk, spans)],
                                      [{'word': w, 'list_position': j, 'repeat': rep, 'pool': pname,
                                        'previous_word': chunk[j - 1] if j else ''} for j, w in enumerate(chunk)]))
    else:
        frame = cond.template.split(TEXT_SLOT)[0]
        if cond.protocol == 'context_bare':
            for w in sorted(context_df['word'].unique()):
                units.append(Unit(f"context_bare|{w}", frame + w, [(w, len(frame), len(frame) + len(w))],
                                  [{'word': w, 'context_char_start': len(frame)}]))
        else:
            for _, r in context_df.iterrows():
                pre, post = r['text'].split(TARGET_SLOT)
                s = len(frame) + len(pre)
                units.append(Unit(f"context_sentence|{r['word']}|{r['context_id']}", frame + pre + r['word'] + post,
                                  [(r['word'], s, s + len(r['word']))],
                                  [{'word': r['word'], 'context_id': r['context_id'], 'meaning_id': r['meaning_id'],
                                    'sense_id': r['sense_id'], 'dominance': r['dominance'],
                                    'frame_id': r['frame_id'], 'n_pre_target_words': int(r['n_pre_target_words']),
                                    'context_char_start': len(frame)}]))
    return units


def align_target(offsets: list, ids: list, text: str, span: tuple[int, int], word: str, decode) -> dict:
    """Map a character span to the token span that realises it in THIS input. A
    target token may absorb leading whitespace (' word', '\\nword'); any other
    merge across the word boundary is an alignment failure, never silently accepted."""
    s, e = span
    out = {'word': word, 'ok': False, 'reason': '', 't0': -1, 't1': -1, 'n_sub': 0,
           'leading_ws_absorbed': False, 'decoded': None, 'target_ids': []}
    if text[s:e] != word:
        out['reason'] = 'char_span_mismatch'
        return out
    hit = [t for t, (a, b) in enumerate(offsets) if b > a and a < e and b > s]
    if not hit:
        out['reason'] = 'no_token_overlaps_span'
        return out
    t0, t1 = hit[0], hit[-1]
    if hit != list(range(t0, t1 + 1)):
        out['reason'] = 'non_contiguous_tokens'
        return out
    a0 = offsets[t0][0]
    if a0 < s:
        if text[a0:s].strip() != '':
            out['reason'] = 'prefix_merged_into_target_token'
            return out
        out['leading_ws_absorbed'] = True
    if offsets[t1][1] > e:
        out['reason'] = 'suffix_merged_into_target_token'
        return out
    covered = np.zeros(e - s, bool)
    for t in range(t0, t1 + 1):
        a, b = offsets[t]
        covered[max(a, s) - s:min(b, e) - s] = True
    if not covered.all():
        out['reason'] = 'span_not_fully_covered'
        return out
    if t0 < 1:
        out['reason'] = 'target_has_no_preceding_position'
        return out
    dec = decode(ids[t0:t1 + 1])
    out['decoded'] = dec
    if dec.strip() != word:
        out['reason'] = f'decoded_target_mismatch({dec!r})'
        return out
    out.update({'ok': True, 't0': t0, 't1': t1, 'n_sub': t1 - t0 + 1, 'target_ids': list(ids[t0:t1 + 1])})
    return out


def offsets_consistent(offsets: list, ids: list, text: str, decode1) -> bool:
    """Sanity check of a fast tokenizer's offset_mapping: non-empty spans must be
    monotone, reproduce the decoded token text and cover every non-space character."""
    covered = np.zeros(len(text), bool)
    last_end, n_tok, n_match = 0, 0, 0
    for (a, b), i in zip(offsets, ids):
        if b <= a:
            continue
        if a < 0 or b > len(text) or a < last_end - 1:
            return False
        n_tok += 1
        piece, dec = text[a:b], decode1(i)
        n_match += int(piece == dec or piece.strip() == dec.strip())
        covered[a:b] = True
        last_end = b
    nonspace = np.array([not c.isspace() for c in text], bool)
    return n_tok > 0 and n_match >= 0.98 * n_tok and bool(covered[nonspace].all())


def offsets_by_decoding(ids: list, text: str, decode) -> list | None:
    """Character offsets rebuilt from incremental decoding (fallback when the
    tokenizer's offset_mapping fails offsets_consistent). Tokens that are not part
    of `text` (e.g. a BOS added by the tokenizer) get (0, 0)."""
    full = decode(ids)
    shift = full.find(text)
    if shift < 0:
        return None
    offs, prev = [], 0
    for i in range(len(ids)):
        cur = len(decode(ids[:i + 1]))
        a, b = prev - shift, cur - shift
        offs.append((a, b) if (a >= 0 and b > a) else (0, 0))
        prev = cur
    return offs if prev == len(full) else None


def _as_id_list(x) -> list:
    if hasattr(x, 'keys'):
        x = x['input_ids']
    if hasattr(x, 'tolist'):
        x = x.tolist()
    if x and isinstance(x[0], list):
        if len(x) != 1:
            raise ChatTemplateError("unexpected batched output from apply_chat_template")
        x = x[0]
    return [int(i) for i in x]


BOUNDARY_DEFINITION = ('token ∈ B if it is a special/control token, decodes to an empty string, or its first '
                       'character is whitespace or punctuation other than an apostrophe or hyphen (i.e. it cannot '
                       'continue an alphabetic word); partial UTF-8 byte tokens are not in B')


def boundary_token_ids(tok) -> np.ndarray:
    decoded = tok.batch_decode([[i] for i in range(len(tok))], skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)
    special = set(tok.all_special_ids)
    keep = [i for i, s in enumerate(decoded)
            if i in special or s == '' or (s[0] != '�' and not s[0].isalnum() and s[0] not in "'-")]
    return np.array(keep, dtype=np.int64)


# ════════════════════════════════════════════════════════════════════════════
# OFFLINE TINY TEST MODELS (random weights; tokenizers reproduce family rules)
# ════════════════════════════════════════════════════════════════════════════
LLAMA3_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|"
                r"\s*[\r\n]+|\s+(?!\S)|\s+")
QWEN2_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|"
               r"\s*[\r\n]+|\s+(?!\S)|\s+")
LLAMA3_STYLE_TEMPLATE = (
    "{{ bos_token }}{% if messages[0]['role'] != 'system' %}<|start_header_id|>system<|end_header_id|>\n\n"
    "Today Date: {{ date_string | default('01 Jan 2000') }}<|eot_id|>{% endif %}"
    "{% for m in messages %}<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n{{ m['content'] | trim }}"
    "<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}")
CHATML_STYLE_TEMPLATE = (
    "{% if messages[0]['role'] != 'system' %}<|im_start|>system\nYou are a test assistant.<|im_end|>\n{% endif %}"
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
TINY_CORPUS_WORDS = ('bank paper river money house table water light school music party watch '
                     'letter string word real english item time each one decide quickly accurately '
                     'whether respond exact format line ease hard easy explain revise answers think '
                     'different meanings related judge completely closely participant experiment '
                     'lexical decision meaning relatedness read following text see will you are a the of for '
                     'and to is').split()
TINY_KINDS = {'llama': ('llama3', True, None), 'qwen2': ('qwen2', False, None),
              'llama_chat': ('llama3', True, LLAMA3_STYLE_TEMPLATE), 'qwen2_chat': ('qwen2', False, CHATML_STYLE_TEMPLATE)}


def build_tiny_tokenizer(style: str, bos: bool, chat_template: str | None):
    """Byte-level BPE trained on a fixed corpus, with the pre-tokenisation of the
    Llama-3 ('llama3'), Qwen2 ('qwen2') or GPT-2/SmolLM2 ('gpt2') tokenizers."""
    from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast
    corpus = [SOURCE_LD_TEMPLATE.replace(ITEM_SLOT, ''), AUTHORED_RELATEDNESS_TEMPLATE.replace(ITEM_SLOT, ''),
              CONTEXT_FRAME_TEMPLATE.replace(TEXT_SLOT, ''), ' '.join(TINY_CORPUS_WORDS),
              '\n'.join(TINY_CORPUS_WORDS)] * 4
    specials = ['<|bos|>', '<|eos|>', '<|pad|>', '<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>',
                '<|im_start|>', '<|im_end|>']
    tok = Tokenizer(models.BPE())
    if style == 'gpt2':
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    else:
        if style == 'qwen2':
            tok.normalizer = normalizers.NFC()
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(LLAMA3_SPLIT if style == 'llama3' else QWEN2_SPLIT), behavior='isolated'),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=560, special_tokens=specials, show_progress=False,
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    if bos:
        tok.post_processor = processors.TemplateProcessing(
            single='<|bos|> $A', special_tokens=[('<|bos|>', tok.token_to_id('<|bos|>'))])
    ft = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token='<|bos|>' if bos else None, eos_token='<|eos|>',
                                 pad_token='<|pad|>',
                                 additional_special_tokens=['<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>',
                                                            '<|im_start|>', '<|im_end|>'])
    if chat_template:
        ft.chat_template = chat_template
    return ft


def build_tiny_test_model(kind: str, seed: int):
    torch = _torch()
    tr = _transformers()
    style, bos, template = TINY_KINDS[kind]
    tok = build_tiny_tokenizer(style, bos, template)
    common = dict(vocab_size=len(tok), hidden_size=48, intermediate_size=96, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=4096, tie_word_embeddings=False,
                  eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    torch.manual_seed(seed + sorted(TINY_KINDS).index(kind) + 1)
    if style == 'llama3':
        model = tr.LlamaForCausalLM(tr.LlamaConfig(num_hidden_layers=4, bos_token_id=tok.bos_token_id, **common))
    else:
        model = tr.Qwen2ForCausalLM(tr.Qwen2Config(num_hidden_layers=3, **common))
    model.eval()
    return tok, model


# ════════════════════════════════════════════════════════════════════════════
# FROZEN LANGUAGE MODEL WITH A VERIFIED LOGIT LENS
# ════════════════════════════════════════════════════════════════════════════
DTYPE_NAMES = {'bf16': 'bfloat16', 'fp16': 'float16', 'fp32': 'float32'}
LENS_METHOD = ('logit lens (nostalgebraist, 2020): log_softmax(W_U · norm_f(h_l) [+ b]) with the model\'s own final '
               'norm and output embedding; the last block\'s HF hidden state is post-norm (verified) and is read out '
               'without a second norm, so the final layer equals the model output; intermediate layers are '
               'logit-lens ESTIMATES; model-level logit soft-capping / scaling applied if configured')


class ScoringModel:
    """Frozen decoder-only LM. Column convention of every layerwise array:
    column 0 = embedding output (hidden_states[0]); column k = output of block
    k-1 (block_index = k-1), k = 1..L; normalised depth = k / L."""

    def __init__(self, spec: ModelSpec, cfg, load_weights: bool = True):
        self.spec, self.cfg = spec, cfg
        self.model = None
        self.lens_ok = None
        self.lens_check, self.padding_check = {}, {}
        self.batch_size = cfg.batch_size
        self.force_decode_offsets = False
        tr = _transformers()
        if _version_tuple(tr.__version__) < MIN_TRANSFORMERS:
            logger.warning(f"transformers {tr.__version__} < {'.'.join(map(str, MIN_TRANSFORMERS))}: Qwen3 and "
                           f"SmolLM3 cannot be loaded")
        if spec.test_model:
            self.tokenizer, self._test_model = build_tiny_test_model(spec.test_model, cfg.seed)
            self.config = self._test_model.config
            self.resolved_revision = 'local-test-random-weights'
        else:
            self.tokenizer = tr.AutoTokenizer.from_pretrained(spec.repo_id, revision=spec.revision,
                                                             trust_remote_code=cfg.trust_remote_code)
            self.config = tr.AutoConfig.from_pretrained(spec.repo_id, revision=spec.revision,
                                                       trust_remote_code=cfg.trust_remote_code)
            self.resolved_revision = getattr(self.config, '_commit_hash', None) or 'unresolved'
        tok = self.tokenizer
        if not getattr(tok, 'is_fast', False):
            raise RuntimeError(f"{spec.alias}: a fast tokenizer (offset mapping) is required for exact target alignment")
        tcfg = getattr(self.config, 'text_config', None) or self.config
        self.num_layers = int(tcfg.num_hidden_layers)
        self.hidden_size = int(tcfg.hidden_size)
        self.model_type = str(getattr(tcfg, 'model_type', 'unknown'))
        self.chat = spec.chat
        self.template_kwargs = {}
        if self.chat:
            if not getattr(tok, 'chat_template', None):
                raise ChatTemplateError(f"{spec.alias}: INSTRUCT model without a chat template; refusing to "
                                        f"hand-write one")
            if 'date_string' in tok.chat_template:
                # Llama-3.x templates otherwise insert strftime_now(), which makes
                # every input (and score) depend on the calendar date of the run.
                self.template_kwargs['date_string'] = cfg.chat_date_string
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else (
            tok.eos_token_id if tok.eos_token_id is not None else 0)
        self.boundary_ids = boundary_token_ids(tok)
        self._boundary_set = set(self.boundary_ids.tolist())
        if load_weights:
            self.load_weights()

    def tokenizer_fingerprint(self) -> dict:
        tok = self.tokenizer
        return {'class': type(tok).__name__, 'len': len(tok),
                'vocab_sha256': sha256_json(sorted(tok.get_vocab().items())),
                'special_tokens': {k: str(v) for k, v in tok.special_tokens_map.items()},
                'chat_template_sha256': sha256_text(tok.chat_template) if self.chat else None,
                'template_kwargs': self.template_kwargs,
                'boundary_set_sha256': sha256_json(self.boundary_ids.tolist())}

    def load_weights(self):
        torch = _torch()
        tr = _transformers()
        cfg = self.cfg
        dtype = getattr(torch, DTYPE_NAMES[cfg.dtype])
        if self.spec.test_model:
            self.model = self._test_model.to(dtype).to(cfg.resolved_device)
            self.device_map = cfg.resolved_device
        else:
            dkey = 'dtype' if _version_tuple(tr.__version__) >= (4, 56, 0) else 'torch_dtype'
            kw = {dkey: dtype, 'revision': self.spec.revision, 'trust_remote_code': cfg.trust_remote_code}
            if cfg.attn_implementation:
                kw['attn_implementation'] = cfg.attn_implementation
            if cfg.quantization != 'none':
                kw['quantization_config'] = tr.BitsAndBytesConfig(
                    load_in_8bit=cfg.quantization == '8bit', load_in_4bit=cfg.quantization == '4bit',
                    bnb_4bit_compute_dtype=dtype)
            if (cfg.device_map == 'auto' and cfg.resolved_device.startswith('cuda')) or cfg.quantization != 'none':
                kw['device_map'] = 'auto'
            self.model = tr.AutoModelForCausalLM.from_pretrained(self.spec.repo_id, **kw)
            if 'device_map' not in kw:
                self.model = self.model.to(cfg.resolved_device)
            self.device_map = getattr(self.model, 'hf_device_map', None) or cfg.resolved_device
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dtype = str(next(self.model.parameters()).dtype)
        prefix = getattr(self.model, 'base_model_prefix', None)
        bb = getattr(self.model, prefix, None) if prefix else None
        self.backbone = bb if isinstance(bb, torch.nn.Module) else self.model
        self.head = self.model.get_output_embeddings()
        self.final_norm, self.final_norm_name = None, None
        for a in ('norm', 'final_layernorm', 'ln_f', 'final_layer_norm', 'norm_f'):
            m = getattr(self.backbone, a, None)
            if isinstance(m, torch.nn.Module):
                self.final_norm, self.final_norm_name = m, a
                break
        self.input_device = self.model.get_input_embeddings().weight.device
        self.head_device, self.head_dtype = self.head.weight.device, self.head.weight.dtype
        params = list(self.final_norm.parameters()) if self.final_norm is not None else []
        self.norm_device = params[0].device if params else self.head_device
        self.norm_dtype = params[0].dtype if params else self.head_dtype
        tcfg = getattr(self.model.config, 'text_config', None) or self.model.config
        self.softcap = getattr(tcfg, 'final_logit_softcapping', None)
        self.logit_scale = getattr(tcfg, 'logit_scale', None)
        self.last_is_post_norm = True
        n_logits = int(self.head.weight.shape[0])
        mask = torch.zeros(n_logits, dtype=torch.bool)
        mask[torch.as_tensor(self.boundary_ids[self.boundary_ids < n_logits])] = True
        self.boundary_mask = mask.to(self.head_device)
        self.n_parameters = int(sum(p.numel() for p in self.model.parameters()))
        self.attn_implementation = str(getattr(self.model.config, '_attn_implementation', None))

    def free(self):
        self.model = self.backbone = self.head = self.final_norm = self._test_model = None
        gc.collect()
        torch = _torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── tokenisation ─────────────────────────────────────────────────────────
    def encode_unit(self, unit: Unit) -> EncodedUnit:
        tok = self.tokenizer
        shift = 0
        if self.chat:
            msgs = [{'role': 'user', 'content': unit.text}]
            rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                               **self.template_kwargs)
            try:
                ids_tpl = _as_id_list(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                              return_dict=False, **self.template_kwargs))
            except TypeError:
                ids_tpl = _as_id_list(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                              **self.template_kwargs))
            shift = None
            for cand, lead in ((unit.text, 0), (unit.text.rstrip(), 0),
                               (unit.text.strip(), len(unit.text) - len(unit.text.lstrip()))):
                if cand and rendered.count(cand) == 1:
                    shift = rendered.find(cand) - lead
                    break
            if shift is None:
                raise ChatTemplateError(f"{self.spec.alias}: the chat template altered the user text so that its "
                                        f"target positions cannot be located")
            enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
            ids = [int(i) for i in enc['input_ids']]
            if ids != ids_tpl:
                raise ChatTemplateError(f"{self.spec.alias}: tokenising the rendered chat string differs from "
                                        f"apply_chat_template(tokenize=True)")
            full_text = rendered
        else:
            enc = tok(unit.text, add_special_tokens=True, return_offsets_mapping=True)
            ids = [int(i) for i in enc['input_ids']]
            full_text = unit.text
        offsets = [tuple(int(v) for v in o) for o in enc['offset_mapping']]
        dec1 = lambda i: tok.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        decn = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        method = 'offset_mapping'
        if self.force_decode_offsets or not offsets_consistent(offsets, ids, full_text, dec1):
            rebuilt = offsets_by_decoding(ids, full_text, decn)
            if rebuilt is not None and offsets_consistent(rebuilt, ids, full_text, dec1):
                offsets, method = rebuilt, 'decode_fallback'
            else:
                offsets, method = [(0, 0)] * len(ids), 'unavailable'
        bos = tok.bos_token_id
        if bos is not None and len(ids) > 1 and ids[0] == bos and ids[1] == bos:
            raise RuntimeError(f"{self.spec.alias}: duplicated BOS token at the start of the input")
        targets = []
        for (w, s, e), meta in zip(unit.targets, unit.meta):
            al = align_target(offsets, ids, full_text, (s + shift, e + shift), w, decn)
            if method == 'unavailable':
                al.update({'ok': False, 'reason': 'token_offsets_unavailable'})
            al['offset_method'] = method
            al['first_token_is_boundary'] = bool(al['ok'] and ids[al['t0']] in self._boundary_set)
            al['target_tokens'] = '|'.join(dec1(i) for i in al['target_ids'])
            pre = []
            if al['ok'] and 'context_char_start' in meta:
                c0 = meta['context_char_start'] + shift
                pre = [i for i, (a, b) in enumerate(offsets) if b > a and a >= c0 and i < al['t0']]
            al['pre_target_token_indices'] = pre
            al.update({k: v for k, v in meta.items() if k not in ('word', 'context_char_start')})
            targets.append(al)
        return EncodedUnit(unit.unit_id, ids, full_text, targets)

    # ── forward ──────────────────────────────────────────────────────────────
    def _pad(self, batch: list[EncodedUnit]):
        torch = _torch()
        T = max(len(u.input_ids) for u in batch)
        ids = torch.full((len(batch), T), int(self.pad_id), dtype=torch.long)
        am = torch.zeros((len(batch), T), dtype=torch.long)
        for b, u in enumerate(batch):
            ids[b, :len(u.input_ids)] = torch.tensor(u.input_ids, dtype=torch.long)
            am[b, :len(u.input_ids)] = 1
        # RIGHT padding: real tokens start at position 0, so absolute positions (and
        # every target / prediction index) are unchanged, and under causal masking
        # no real token attends to a pad.
        return ids.to(self.input_device), am.to(self.input_device)

    def _hidden_states(self, ids, am):
        hs = self.backbone(input_ids=ids, attention_mask=am, output_hidden_states=True, use_cache=False).hidden_states
        if hs is None or len(hs) != self.num_layers + 1:
            raise RuntimeError(f"{self.spec.alias}: expected {self.num_layers + 1} hidden-state tensors, got "
                               f"{None if hs is None else len(hs)}")
        for h in (hs[0], hs[-1]):
            if tuple(h.shape) != (ids.shape[0], ids.shape[1], self.hidden_size):
                raise RuntimeError(f"{self.spec.alias}: hidden-state shape {tuple(h.shape)} does not match the "
                                   f"configuration")
        return hs

    def _postprocess_logits(self, logits):
        torch = _torch()
        if self.logit_scale not in (None, 1, 1.0):
            logits = logits * float(self.logit_scale)
        if self.softcap:
            logits = float(self.softcap) * torch.tanh(logits / float(self.softcap))
        return logits

    def lens_logprobs(self, h, k: int):
        """Log-probabilities of the logit lens at layer column k for states h (P, d)."""
        torch = _torch()
        x = h
        if not (k == self.num_layers and self.last_is_post_norm):
            x = self.final_norm(x.to(self.norm_device, self.norm_dtype))
        logits = self.head(x.to(self.head_device, self.head_dtype)).float()
        return torch.log_softmax(self._postprocess_logits(logits), dim=-1)

    def score_encoded(self, encoded: list[EncodedUnit], want_reps: bool, batch_size: int | None = None,
                      desc: str = '') -> dict:
        """Per target: subtoken log-probabilities (each from position t−1), the
        boundary-set mass after the word (eow, position t1) and before it (bow,
        position t0−1) at every layer; for context targets the final-layer
        surprisal of the pre-target context tokens; optional representations
        AT the target positions (last subtoken / mean of subtokens)."""
        torch = _torch()
        L1 = self.num_layers + 1
        n_t = sum(len(u.targets) for u in encoded)
        t_start = np.cumsum([0] + [len(u.targets) for u in encoded])
        flat = [t for u in encoded for t in u.targets]
        n_sub = np.array([t['n_sub'] if t['ok'] else 0 for t in flat], int)
        tok_start = np.concatenate([[0], np.cumsum(n_sub)[:-1]]).astype(int) if n_t else np.zeros(0, int)
        tok_logp = np.full((int(n_sub.sum()), L1), np.nan, np.float32)
        eow = np.full((n_t, L1), np.nan, np.float32)
        bow = np.full((n_t, L1), np.nan, np.float32)
        pre_sum = np.zeros(n_t)
        pre_n = np.zeros(n_t, int)
        reps_last = np.zeros((n_t, L1, self.hidden_size), np.float16) if want_reps else None
        reps_mean = np.zeros((n_t, L1, self.hidden_size), np.float16) if want_reps else None
        bs = batch_size or self.batch_size
        order = sorted(range(len(encoded)), key=lambda i: -len(encoded[i].input_ids))
        n_batches = max(1, math.ceil(len(order) / bs))
        t_clock = time.time()
        for bi in range(0, len(order), bs):
            idxs = order[bi:bi + bs]
            ids, am = self._pad([encoded[i] for i in idxs])
            rows, pos, tgt, kind, own = [], [], [], [], []      # kind: 0 subtoken, 1 eow, 2 bow, 3 pre-target
            rep_rows, rep_t0, rep_t1, rep_own = [], [], [], []
            for b, ui in enumerate(idxs):
                u = encoded[ui]
                for j, t in enumerate(u.targets):
                    gi = int(t_start[ui] + j)
                    if not t['ok']:
                        continue
                    for s, tt in enumerate(range(t['t0'], t['t1'] + 1)):
                        rows.append(b), pos.append(tt - 1), tgt.append(u.input_ids[tt]), kind.append(0)
                        own.append(int(tok_start[gi] + s))
                    rows.append(b), pos.append(t['t1']), tgt.append(0), kind.append(1), own.append(gi)
                    rows.append(b), pos.append(t['t0'] - 1), tgt.append(0), kind.append(2), own.append(gi)
                    for pi in t['pre_target_token_indices']:
                        if pi >= 1:
                            rows.append(b), pos.append(pi - 1), tgt.append(u.input_ids[pi]), kind.append(3)
                            own.append(gi)
                    if want_reps:
                        rep_rows.append(b), rep_t0.append(t['t0']), rep_t1.append(t['t1']), rep_own.append(gi)
            kind_a, own_a = np.array(kind, int), np.array(own, int)
            with torch.inference_mode():
                hs = self._hidden_states(ids, am)
                for k in range(L1):
                    h = hs[k]
                    if self.lens_ok and rows:
                        use = np.arange(len(rows)) if k == self.num_layers else np.where(kind_a != 3)[0]
                        r_t = torch.tensor([rows[i] for i in use], device=h.device)
                        p_t = torch.tensor([pos[i] for i in use], device=h.device)
                        sel, bmass = [], []
                        for c0 in range(0, len(use), self.cfg.lens_chunk):
                            lp = self.lens_logprobs(h[r_t[c0:c0 + self.cfg.lens_chunk],
                                                      p_t[c0:c0 + self.cfg.lens_chunk]], k)
                            tg = torch.tensor([tgt[i] for i in use[c0:c0 + self.cfg.lens_chunk]], device=lp.device)
                            sel.append(lp.gather(1, tg[:, None])[:, 0].cpu())
                            bmass.append(torch.logsumexp(lp.masked_fill(~self.boundary_mask, float('-inf')),
                                                         -1).cpu())
                        sel = torch.cat(sel).numpy()
                        bmass = torch.cat(bmass).numpy()
                        ku, ou = kind_a[use], own_a[use]
                        tok_logp[ou[ku == 0], k] = sel[ku == 0]
                        eow[ou[ku == 1], k] = bmass[ku == 1]
                        bow[ou[ku == 2], k] = bmass[ku == 2]
                        if k == self.num_layers and (ku == 3).any():
                            np.add.at(pre_sum, ou[ku == 3], -sel[ku == 3] / LN2)
                            np.add.at(pre_n, ou[ku == 3], 1)
                    if want_reps and rep_rows:
                        hr = h.float()
                        ar = torch.arange(hr.shape[1], device=hr.device)
                        t0s = torch.tensor(rep_t0, device=hr.device)
                        t1s = torch.tensor(rep_t1, device=hr.device)
                        rr = torch.tensor(rep_rows, device=hr.device)
                        mask = ((ar[None, :] >= t0s[:, None]) & (ar[None, :] <= t1s[:, None])).float()
                        reps_last[rep_own, k] = hr[rr, t1s].cpu().numpy().astype(np.float16)
                        reps_mean[rep_own, k] = ((hr[rr] * mask[..., None]).sum(1) /
                                                 mask.sum(1, keepdim=True)).cpu().numpy().astype(np.float16)
                del hs
            if desc and n_batches >= 20 and (bi // bs) % (n_batches // 10) == 0:
                logger.info(f"  {desc}: {min(bi + bs, len(order))}/{len(order)} units ({time.time() - t_clock:.0f}s)")
        with np.errstate(invalid='ignore', divide='ignore'):
            pre_mean = np.where(pre_n > 0, pre_sum / np.maximum(pre_n, 1), np.nan)
        return {'tok_logp': tok_logp, 'tok_start': tok_start, 'n_sub': n_sub, 'eow_logmass': eow,
                'bow_logmass': bow, 'pre_target_mean_surprisal_bits': pre_mean, 'n_pre_target_tokens_scored': pre_n,
                'reps_last': reps_last, 'reps_mean': reps_mean}

    # ── preflight checks ─────────────────────────────────────────────────────
    def verify_readout(self, sample: list[EncodedUnit]) -> dict:
        """Establish empirically (i) the hidden-state count / width, (ii) whether
        hidden_states[-1] is post-norm, (iii) that the logit-lens readout of the
        last block reproduces the model's own next-token log-probabilities at the
        exact target-prediction positions. Failure marks the model's layerwise
        surprisal UNAVAILABLE instead of producing misleading values."""
        torch = _torch()
        res = {'method': LENS_METHOD, 'status': 'UNAVAILABLE', 'reason': ''}
        self.lens_ok = False
        batch = [u for u in sample if any(t['ok'] for t in u.targets)][:3]
        if self.final_norm is None or not batch:
            res['reason'] = 'no final normalisation module' if self.final_norm is None else 'no aligned target'
            self.lens_check = res
            return res
        ids, am = self._pad(batch)
        with torch.inference_mode():
            out = self.model(input_ids=ids, attention_mask=am, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
            res.update({'n_hidden_state_tensors': len(hs), 'expected_hidden_state_tensors': self.num_layers + 1,
                        'hidden_size_observed': int(hs[-1].shape[-1])})
            if len(hs) != self.num_layers + 1 or int(hs[-1].shape[-1]) != self.hidden_size:
                res['reason'] = 'hidden-state count or width does not match the configuration'
                self.lens_check = res
                return res
            rows, pos, tg = [], [], []
            for b, u in enumerate(batch):
                for t in u.targets:
                    if t['ok']:
                        for tt in range(t['t0'], t['t1'] + 1):
                            rows.append(b), pos.append(tt - 1), tg.append(u.input_ids[tt])
            r_t = torch.tensor(rows, device=out.logits.device)
            p_t = torch.tensor(pos, device=out.logits.device)
            ref = out.logits[r_t, p_t].float()
            h_last = hs[-1][r_t.to(hs[-1].device), p_t.to(hs[-1].device)]
            direct = self._postprocess_logits(self.head(h_last.to(self.head_device, self.head_dtype)).float())
            normed = self._postprocess_logits(self.head(self.final_norm(h_last.to(self.norm_device, self.norm_dtype))
                                                        .to(self.head_device, self.head_dtype)).float())
            rel = lambda a: float((a.to(ref.device) - ref).norm() / ref.norm().clamp_min(1e-6))
            e_direct, e_normed = rel(direct), rel(normed)
            self.last_is_post_norm = bool(e_direct <= e_normed)
            res.update({'relative_error_head_on_last_state': e_direct,
                        'relative_error_head_on_norm_of_last_state': e_normed,
                        'hidden_states_last_is_post_norm': self.last_is_post_norm})
            if min(e_direct, e_normed) > 0.02:
                res['reason'] = f'the output head does not reproduce the model logits ({e_direct:.3g} / {e_normed:.3g})'
                self.lens_check = res
                return res
            lp_ref = torch.log_softmax(ref, -1)
            tgt = torch.tensor(tg, device=lp_ref.device)
            d_tgt = (self.lens_logprobs(h_last, self.num_layers).to(lp_ref.device).gather(1, tgt[:, None])
                     - lp_ref.gather(1, tgt[:, None])).abs()
            res['max_abs_diff_target_logprob_nats_last_layer'] = float(d_tgt.max())
            mid = max(1, self.num_layers // 2)
            h_mid = hs[mid][r_t.to(hs[mid].device), p_t.to(hs[mid].device)]
            raw = torch.log_softmax(self._postprocess_logits(
                self.head(h_mid.to(self.head_device, self.head_dtype)).float()), -1)
            res['mid_layer_lens_vs_raw_dot_product_max_abs_diff_nats'] = float(
                (self.lens_logprobs(h_mid, mid) - raw).abs().max())
        tol = 1e-3 if self.cfg.dtype == 'fp32' else 0.05
        if res['max_abs_diff_target_logprob_nats_last_layer'] > tol:
            res['reason'] = (f"last-layer lens differs from the model's own log-probabilities by "
                             f"{res['max_abs_diff_target_logprob_nats_last_layer']:.3g} nats (> {tol})")
        else:
            res['status'] = 'OK'
            self.lens_ok = True
        self.lens_check = res
        return res

    def check_padding(self, encoded: list[EncodedUnit]) -> dict:
        """Scores of the same targets in one padded batch vs one unit at a time (no
        padding). A violation beyond tolerance switches this model to batch size 1."""
        if not self.lens_ok:
            self.padding_check = {'status': 'SKIPPED', 'reason': 'logit lens unavailable'}
            return self.padding_check
        sample = [u for u in encoded if any(t['ok'] for t in u.targets)]
        lens = sorted({len(u.input_ids) for u in sample})
        pick = []
        for ln in (lens[0], lens[len(lens) // 2], lens[-1]) if lens else ():
            pick += [u for u in sample if len(u.input_ids) == ln][:2]
        pick = list({u.unit_id: u for u in pick}.values())[:6]
        if len(pick) < 2:
            self.padding_check = {'status': 'SKIPPED', 'reason': 'fewer than two units'}
            return self.padding_check
        together = self.score_encoded(pick, want_reps=False, batch_size=len(pick))['tok_logp']
        single = np.concatenate([self.score_encoded([u], want_reps=False, batch_size=1)['tok_logp'] for u in pick])
        d = float(np.nanmax(np.abs(together - single)) / LN2)
        tol = 1e-3 if self.cfg.dtype == 'fp32' else self.cfg.padding_tolerance_bits
        st = {'max_abs_diff_bits': d, 'tolerance_bits': tol, 'n_units': len(pick),
              'lengths': [len(u.input_ids) for u in pick], 'status': 'PASS'}
        if d > tol:
            st['status'] = 'FAIL→batch_size=1'
            self.batch_size = 1
            logger.warning(f"{self.spec.alias}: batched vs single scores differ by {d:.3g} bits (> {tol}); "
                           f"falling back to batch_size=1 (no padding)")
        self.padding_check = st
        return st

    def describe(self) -> dict:
        tok = self.tokenizer
        probe = tok('a', add_special_tokens=True)['input_ids']
        return {
            'alias': self.spec.alias, 'repo_id': self.spec.repo_id, 'requested_revision': self.spec.revision,
            'resolved_revision': self.resolved_revision, 'variant': self.spec.variant, 'family': self.spec.family,
            'base_of': self.spec.base_of, 'test_model_random_weights': bool(self.spec.test_model),
            'model_type': self.model_type, 'num_hidden_layers': self.num_layers, 'hidden_size': self.hidden_size,
            'tokenizer_len': len(tok), 'tokenizer_class': type(tok).__name__,
            'special_tokens': {k: str(v) for k, v in tok.special_tokens_map.items()},
            'special_token_ids': {'bos': tok.bos_token_id, 'eos': tok.eos_token_id, 'pad': tok.pad_token_id,
                                  'pad_id_used_for_padding': self.pad_id},
            'tokenizer_adds_bos': bool(tok.bos_token_id is not None and probe and probe[0] == tok.bos_token_id),
            'padding_used': 'manual right padding with attention mask (positions of real tokens unchanged)',
            'chat_template_applied': self.chat,
            'chat_template_sha256': sha256_text(tok.chat_template) if getattr(tok, 'chat_template', None) else None,
            'chat_template_kwargs': self.template_kwargs, 'boundary_set_size': int(len(self.boundary_ids)),
            'boundary_definition': BOUNDARY_DEFINITION, 'dtype': self.dtype, 'device_map': str(self.device_map),
            'attn_implementation': self.attn_implementation, 'quantization': self.cfg.quantization,
            'n_parameters': self.n_parameters, 'final_norm_module': self.final_norm_name,
            'logit_softcapping': self.softcap, 'logit_scale': self.logit_scale,
            'lens_verification': self.lens_check, 'padding_check': self.padding_check,
            'batch_size_used': self.batch_size}


# ════════════════════════════════════════════════════════════════════════════
# CACHE (per model × condition; manifest-keyed; stale caches are refused)
# ════════════════════════════════════════════════════════════════════════════
AGGREGATES = ('sum', 'mean', 'first', 'whole_word')
AGGREGATE_DEFINITIONS = {
    'sum': 'PRIMARY. Σ_j −log2 p(t_j | context, t_<j): sequence surprisal of all target subtokens (bits)',
    'mean': 'sensitivity: sum / number of subtokens (length-normalised)',
    'first': ('sensitivity: surprisal of the first subtoken only. In the isolated protocol it is read from a '
              'hidden state that is identical for every item (same prefix)'),
    'whole_word': ('sensitivity: sum + eow − bow·[first target token ∈ B] (Pimentel & Meister, 2024), with '
                   'eow = −log2 Σ_{t∈B} p(t | context, word) and bow = −log2 Σ_{t∈B} p(t | context); B = boundary '
                   'set (see BOUNDARY_DEFINITION). Assumes canonical tokenisation; when the first target token is not '
                   'in B (item after a newline) the residual prefix-boundary term is constant across items of an '
                   'isolated condition and does not affect item effects. It does NOT remove the dependence of summed '
                   'surprisal on subtoken count (see tokenization_bias_audit)')}


def units_fingerprint(encoded: list[EncodedUnit]) -> str:
    return sha256_json([[u.unit_id, u.input_ids, [[t['t0'], t['t1'], t['ok']] for t in u.targets]] for u in encoded])


def analysis_key(cfg, cond: PromptCondition, units: list[Unit]) -> dict:
    """Cache keys checkable WITHOUT a tokenizer (used by --analysis_only)."""
    return {'script_version': SCRIPT_VERSION, 'condition': cond.name, 'protocol': cond.protocol,
            'template_sha256': cond.template_sha256,
            'context_sha256': cfg.context_sha256 if cond.protocol.startswith('context_') else None,
            'units_sha256': sha256_json([[u.unit_id, u.text, u.targets] for u in units]),
            'dtype': cfg.dtype, 'quantization': cfg.quantization, 'chat_date_string': cfg.chat_date_string,
            'list_params': [cfg.list_size, cfg.list_repeats, cfg.seed] if cond.protocol == 'list' else None,
            'aggregate_definitions_sha256': sha256_json(AGGREGATE_DEFINITIONS)}


class CacheEntry:
    def __init__(self, out_dir: str, alias: str, condition: str):
        self.dir = os.path.join(out_dir, 'cache', safe_name(alias), safe_name(condition))
        self.manifest_path = os.path.join(self.dir, 'manifest.json')

    def exists(self) -> bool:
        return os.path.exists(self.manifest_path) and os.path.exists(os.path.join(self.dir, 'scores.npz'))

    def manifest(self) -> dict:
        return read_json(self.manifest_path)

    def mismatches(self, expected_key: dict, field_name: str = 'key') -> list[str]:
        if not self.exists():
            return ['<missing>']
        have = self.manifest().get(field_name, {})
        return sorted(k for k in set(have) | set(expected_key)
                      if _clean_json(have.get(k)) != _clean_json(expected_key.get(k)))

    def save(self, key: dict, akey: dict, info: dict, arrays: dict, reps: dict):
        tmp = self.dir + '.partial'
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)
        with open(os.path.join(tmp, 'scores.npz'), 'wb') as f:
            np.savez_compressed(f, **arrays)
        saved = {}
        for name, arr in reps.items():
            np.save(os.path.join(tmp, f'reps_{name}.npy'), arr)
            saved[name] = list(arr.shape)
        atomic_write_json(os.path.join(tmp, 'manifest.json'),
                          {'key': key, 'analysis_key': akey,
                           'info': dict(info, representations_saved=saved,
                                        created_utc=datetime.now(timezone.utc).isoformat())})
        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)
        os.replace(tmp, self.dir)

    def load(self) -> tuple[dict, dict, dict]:
        man = self.manifest()
        with np.load(os.path.join(self.dir, 'scores.npz'), allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        reps = {name: np.load(os.path.join(self.dir, f'reps_{name}.npy'), mmap_mode='r')
                for name in ('last', 'mean') if os.path.exists(os.path.join(self.dir, f'reps_{name}.npy'))}
        return man, arrays, reps


TARGET_META = ('list_position', 'repeat', 'pool', 'previous_word', 'context_id', 'meaning_id', 'sense_id',
               'dominance', 'frame_id', 'n_pre_target_words')


def pack_scores(encoded: list[EncodedUnit], sc: dict) -> dict:
    flat = [(u.unit_id, t) for u in encoded for t in u.targets]
    arrays = {'unit_id': np.array([f[0] for f in flat], dtype=str),
              'word': np.array([f[1]['word'] for f in flat], dtype=str),
              'align_ok': np.array([f[1]['ok'] for f in flat], bool),
              'align_reason': np.array([f[1]['reason'] for f in flat], dtype=str),
              'offset_method': np.array([f[1]['offset_method'] for f in flat], dtype=str),
              'target_tokens': np.array([f[1]['target_tokens'] for f in flat], dtype=str),
              'leading_ws_absorbed': np.array([f[1]['leading_ws_absorbed'] for f in flat], bool),
              'first_token_is_boundary': np.array([f[1]['first_token_is_boundary'] for f in flat], bool),
              'n_sub': sc['n_sub'].astype(int), 'tok_start': sc['tok_start'].astype(int), 'tok_logp': sc['tok_logp'],
              'eow_logmass': sc['eow_logmass'], 'bow_logmass': sc['bow_logmass'],
              'n_pre_target_tokens': sc['n_pre_target_tokens_scored'].astype(int),
              'pre_target_mean_surprisal_bits': sc['pre_target_mean_surprisal_bits']}
    for c in TARGET_META:
        arrays[f'meta_{c}'] = np.array(['' if f[1].get(c) is None else str(f[1].get(c, '')) for f in flat], dtype=str)
    return arrays


def aggregate_surprisal(arrays: dict) -> dict:
    """Per-target layerwise surprisal (bits) for every aggregate, shape
    (n_targets, L+1); NaN for targets that failed alignment."""
    tok = arrays['tok_logp'].astype(np.float64)
    n_sub, start = arrays['n_sub'], arrays['tok_start']
    n_t, L1 = arrays['eow_logmass'].shape
    s = np.full((n_t, L1), np.nan)
    first = np.full((n_t, L1), np.nan)
    ok = n_sub > 0
    if ok.any():
        idx = np.where(ok)[0]
        s[idx] = -np.add.reduceat(tok, start[idx], axis=0) / LN2
        first[idx] = -tok[start[idx]] / LN2
    eow_bits = -arrays['eow_logmass'].astype(np.float64) / LN2
    bow_bits = -arrays['bow_logmass'].astype(np.float64) / LN2
    return {'sum': s, 'mean': s / np.where(ok, n_sub, np.nan)[:, None], 'first': first,
            'whole_word': s + eow_bits - bow_bits * arrays['first_token_is_boundary'][:, None]}


# ════════════════════════════════════════════════════════════════════════════
# PRESPECIFIED ANALYSIS PLAN (fixed in code before any model was run)
# ════════════════════════════════════════════════════════════════════════════
REGIONS = {'shallow': (0.0, 1.0 / 3.0), 'middle': (1.0 / 3.0, 2.0 / 3.0), 'deep': (2.0 / 3.0, 1.0)}
SUMMARIES = ('shallow', 'middle', 'deep', 'output')
# Orientation: +1 if the hypothesised human-like effect is a positive coefficient on
# surprisal (cost), −1 if negative (facilitation). Oriented effect = sign × β_std.
ORIENT = {'log_subtlex': -1, 'gcide_etymologies': 1, 'wn_n_all': -1, 'ambiguity': 1, 'senses': -1}
# (id, analysis, kind, terms, expected sign, family, source statement)
HYPOTHESES = [
    ('S1A_meanings', 'stage1_A', 'effect', ('ambiguity',), '+', 'stage1_calibration',
     'Exp. 1 regression (visual): ambiguity β = +.18, t = 2.7 — slower'),
    ('S1A_senses', 'stage1_A', 'effect', ('wn_n_all',), '-', 'stage1_calibration',
     'Exp. 1 regression (visual): word senses β = −.17, t = −2.4 — faster'),
    ('S1B_senses', 'stage1_B', 'effect', ('senses',), '-', 'stage1_calibration',
     'Exp. 2 (visual): many senses 14 ms faster, significant'),
    ('S1B_meanings_null', 'stage1_B', 'equivalence', ('ambiguity',), '0', 'stage1_calibration',
     'Exp. 2 (visual): ambiguity 6 ms slower, NOT significant'),
    ('S1B_separation', 'stage1_B', 'contrast', ('ambiguity', 'senses'), '+', 'stage1_calibration',
     'meanings and senses come apart within one model (β_ambiguity − β_senses > 0)'),
    ('S1C_senses', 'stage1_C', 'effect', ('senses',), '-', 'stage1_calibration',
     'Exp. 3 (auditory; text here): many senses 33 ms faster, significant'),
    ('S1C_meanings', 'stage1_C', 'effect', ('ambiguity',), '+', 'stage1_calibration',
     'Exp. 3 (auditory; text here): ambiguity 29 ms slower, significant'),
    ('S1C_separation', 'stage1_C', 'contrast', ('ambiguity', 'senses'), '+', 'stage1_calibration',
     'meanings and senses come apart within one model (β_ambiguity − β_senses > 0)'),
    ('S2_meanings', 'stage2', 'effect', ('gcide_etymologies',), '+', 'stage2_generalisation',
     'meanings proxy: more etymologies → higher surprisal'),
    ('S2_senses', 'stage2', 'effect', ('wn_n_all',), '-', 'stage2_generalisation',
     'senses proxy: more WordNet senses → lower surprisal'),
    ('S2_frequency', 'stage2', 'effect', ('log_subtlex',), '-', 'stage2_generalisation',
     'higher frequency → lower surprisal'),
    ('S2_separation', 'stage2', 'contrast', ('gcide_etymologies', 'wn_n_all'), '+', 'stage2_generalisation',
     'meanings and senses proxies come apart (β_gcide − β_wn_n_all > 0)'),
    ('D_frequency_shallow', 'stage2', 'slope', ('log_subtlex',), '-', 'depth_hypothesis',
     'oriented frequency effect decreases with depth (large in shallow layers, flattening)'),
    ('D_meanings_deep', 'stage2', 'slope', ('gcide_etymologies',), '+', 'depth_hypothesis',
     'oriented meanings effect increases with depth'),
    ('D_senses_deep', 'stage2', 'slope', ('wn_n_all',), '+', 'depth_hypothesis',
     'oriented senses effect increases with depth'),
    ('D_meanings_vs_frequency', 'stage2', 'slope_contrast', ('gcide_etymologies', 'log_subtlex'), '+',
     'depth_hypothesis', 'depth slopes of meanings and frequency differ in the predicted direction'),
    ('D_senses_vs_frequency', 'stage2', 'slope_contrast', ('wn_n_all', 'log_subtlex'), '+', 'depth_hypothesis',
     'depth slopes of senses and frequency differ in the predicted direction'),
]
STAGE2_SPECS = {
    'primary': (['gcide_etymologies', 'wn_n_all', 'log_subtlex', 'length', 'concreteness'], 'confirmatory'),
    'robust_orthoN': (['gcide_etymologies', 'wn_n_all', 'log_subtlex', 'length', 'concreteness', 'ortho_n'],
                      'robustness'),
    'robust_orthoN_nsyll': (['gcide_etymologies', 'wn_n_all', 'log_subtlex', 'length', 'concreteness', 'ortho_n',
                             'n_syll'], 'robustness'),
    # Summed surprisal grows mechanically with the number of subtokens and rare words
    # split into more subtokens: a prespecified, model-specific robustness covariate.
    'robust_n_subtokens': (['gcide_etymologies', 'wn_n_all', 'log_subtlex', 'length', 'concreteness',
                            'n_subtokens'], 'robustness'),
    'explor_sim_top2': (['gcide_etymologies', 'wn_n_all', 'log_subtlex', 'length', 'concreteness', 'sim_top2'],
                        'exploratory'),
    'explor_min_sim': (['gcide_etymologies', 'wn_n_all', 'log_subtlex', 'length', 'concreteness', 'min_sim'],
                       'exploratory'),
    'explor_mean_sim': (['gcide_etymologies', 'wn_n_all', 'log_subtlex', 'length', 'concreteness', 'mean_sim'],
                        'exploratory'),
}
MATCH_CONTRASTS = {
    'meanings': {'covariates': ['wn_n_all', 'log_subtlex', 'length', 'concreteness', 'ortho_n'],
                 'definition': 'gcide_etymologies >= 2 (treated) vs == 1 (control)'},
    'senses': {'covariates': ['gcide_etymologies', 'log_subtlex', 'length', 'concreteness', 'ortho_n'],
               'definition': 'wn_n_all top tertile (treated) vs bottom tertile (control)'},
    'frequency': {'covariates': ['gcide_etymologies', 'wn_n_all', 'length', 'concreteness', 'ortho_n'],
                  'definition': 'log_subtlex top tertile (treated) vs bottom tertile (control)'},
}


def analysis_plan(cfg) -> dict:
    return {
        'status': (f'Fixed in this script (v{SCRIPT_VERSION}) before it was run on any model. It is NOT a formal '
                   'preregistration; register it (e.g. on OSF) before running the real models if the confirmatory '
                   'labels are to carry weight.'),
        'primary_measure': cfg.primary_measure, 'primary_aggregate': cfg.primary_aggregate,
        'aggregate_definitions': AGGREGATE_DEFINITIONS, 'boundary_set': BOUNDARY_DEFINITION,
        'primary_condition': cfg.primary_condition,
        'stand_in_mapping': ('THEORETICAL SIGN COMPARISON ONLY: human processing cost (slower lexical decision) is '
                             'compared with HIGHER surprisal. It is not a validated equivalence and surprisal is not a '
                             'reaction time.'),
        'depth': ('block outputs only; normalised depth = (block_index + 1) / num_hidden_layers, identical for every '
                  'model family; embedding output stored but not analysed'),
        'regions': {k: f'({a:.4f}, {b:.4f}]' for k, (a, b) in REGIONS.items()},
        'summaries': ('each prespecified effect is evaluated on the mean standardised β over the shallow, middle and '
                      'deep regions and at the output (final) layer — never on the best of many layers'),
        'hypotheses': [{'id': h[0], 'analysis': h[1], 'kind': h[2], 'terms': list(h[3]), 'expected': h[4],
                        'family': h[5], 'source': h[6]} for h in HYPOTHESES],
        'standardisation': ('surprisal z-scored within layer; continuous predictors z-scored; Stage 1 factors '
                            'effect-coded ±0.5 (B, C: β = level difference in SD units) or dummy 0/1 (A ambiguity)'),
        'uncertainty': ('nonparametric bootstrap over words (B = %d), stratified within the published design cells '
                        '(Appendix A: ambiguity groups; B, C: the four cells), the same resample for every layer; '
                        'percentile 95%% CIs; two-sided p from the bootstrap SE (normal approximation); equivalence '
                        'by two one-sided tests against ±SESOI = %.3g SD' % (cfg.n_bootstrap, cfg.sesoi_std)),
        'multiplicity': ('Holm within each prespecified family (stage1_calibration, stage2_generalisation, '
                         'depth_hypothesis) of one model × condition × aggregate; layerwise profiles: '
                         'Benjamini–Hochberg across layers (exploratory)'),
        'classification': {
            'directional': 'supported if Holm p < alpha in the expected direction; contradicted if Holm p < alpha in '
                           'the opposite direction; otherwise inconclusive',
            'equivalence': 'supported if Holm-adjusted TOST p < alpha; contradicted if Holm-adjusted two-sided p < '
                           'alpha; otherwise inconclusive (non-significance is never read as a null)'},
        'meanings_vs_senses_pattern': ('per design and summary: "distinct effects in the predicted directions", '
                                       '"meanings and senses differ, not in the predicted pattern", "one shared '
                                       'effect (not distinguished)", or "no detectable difference"'),
        'depth_test': ('direct test of the depth hypothesis: least-squares slope of the ORIENTED standardised effect '
                       'on normalised depth across all blocks, per term, and slope differences between ambiguity '
                       'terms and frequency; region contrasts (deep − shallow), |β| contrasts and peak depth are '
                       'secondary and exploratory'),
        'prompt_and_context': ('task-instruction effects: other isolated frames vs the primary frame, paired by word; '
                               'context effects: contexts vs the bare word in ONE shared frame, so the two '
                               'manipulations are never confounded; both exploratory'),
        'homonymy_polysemy': ('primary analysis only with human-validated meaning_id / sense_id annotations: cosine '
                              'distance between the target\'s representations in two contexts, different-meaning '
                              'pairs vs same-meaning different-sense pairs, cluster bootstrap over words'),
        'probes': ('secondary; linear ridge probe (dual form), λ = %.3g · mean diag(K_train), %d word-grouped folds, '
                   '%d label permutations, no-hidden-state baseline; per appendix, never pooled'
                   % (cfg.probe_lambda_scale, cfg.n_folds, cfg.n_permutations)),
        'matched': {'contrasts': MATCH_CONTRASTS, 'caliper_sd': cfg.match_caliper_sd},
        'stage2_specs': {k: {'predictors': v[0], 'status': v[1]} for k, v in STAGE2_SPECS.items()},
        'gcide_zero_policy_primary': cfg.gcide_zero_policy,
    }


# ════════════════════════════════════════════════════════════════════════════
# SCORE SETS (cached model outputs, ready for analysis)
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class ScoreSet:
    alias: str
    condition: str
    protocol: str
    cond_status: str
    n_layers: int
    model_info: dict
    targets: pd.DataFrame
    surprisal: dict
    reps: dict

    @property
    def depth(self) -> np.ndarray:
        return np.arange(1, self.n_layers + 1) / self.n_layers

    def ctx(self, aggregate: str | None = None) -> dict:
        mi = self.model_info
        d = {'model': self.alias, 'repo_id': mi.get('repo_id'), 'variant': mi.get('variant'),
             'family': mi.get('family'), 'test_model': bool(mi.get('test_model_random_weights')),
             'condition': self.condition, 'protocol': self.protocol, 'condition_status': self.cond_status}
        if aggregate is not None:
            d['aggregate'] = aggregate
        return d

    def word_matrix(self, aggregate: str, mask=None) -> pd.DataFrame:
        """Words × block columns (k = 1..L); repeated targets of a word (list repeats,
        several contexts) are averaged; failed alignments excluded."""
        ok = self.targets['align_ok'].values & (True if mask is None else np.asarray(mask, bool))
        df = pd.DataFrame(self.surprisal[aggregate][ok][:, 1:], columns=range(1, self.n_layers + 1))
        df['word'] = self.targets.loc[ok, 'word'].values
        return df.groupby('word', sort=True).mean()

    def n_subtokens(self) -> pd.Series:
        return self.targets[self.targets['align_ok']].groupby('word')['n_sub'].mean()


def load_scoreset(cache: CacheEntry, cond: PromptCondition, model_info: dict) -> ScoreSet:
    man, arrays, reps = cache.load()
    tg = pd.DataFrame({k: arrays[k] for k in ('unit_id', 'word', 'align_ok', 'align_reason', 'offset_method',
                                              'target_tokens', 'leading_ws_absorbed', 'first_token_is_boundary',
                                              'n_pre_target_tokens', 'pre_target_mean_surprisal_bits')})
    tg['n_sub'] = arrays['n_sub']
    for c in TARGET_META:
        tg[c] = arrays[f'meta_{c}']
    return ScoreSet(model_info.get('alias', man['key']['model_alias']), cond.name, cond.protocol, cond.status,
                    int(man['key']['num_hidden_layers']), model_info, tg, aggregate_surprisal(arrays), reps)


def readout_type(k: np.ndarray, L: int) -> np.ndarray:
    return np.where(k == 0, 'embedding output (lens estimate; not analysed)',
                    np.where(k == L, 'model output (final layer)', 'logit-lens estimate (intermediate block)'))


def item_layer_frame(ss: ScoreSet) -> pd.DataFrame:
    """Per-target, per-layer long table with token counts and alignment diagnostics."""
    L1 = ss.n_layers + 1
    k = np.tile(np.arange(L1), len(ss.targets))
    out = ss.targets.loc[ss.targets.index.repeat(L1)].reset_index(drop=True)
    out.insert(0, 'protocol', ss.protocol)
    out.insert(0, 'condition', ss.condition)
    out.insert(0, 'model', ss.alias)
    out['layer_column'] = k
    out['block_index'] = k - 1
    out['depth'] = k / ss.n_layers
    out['readout_type'] = readout_type(k, ss.n_layers)
    for agg in AGGREGATES:
        out[f'surprisal_{agg}_bits'] = ss.surprisal[agg].reshape(-1)
    return out


# ════════════════════════════════════════════════════════════════════════════
# DESIGNS
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Design:
    analysis: str
    spec: str
    status: str
    words: np.ndarray
    terms: list
    raw: pd.DataFrame
    kinds: dict                    # term -> 'z' (standardised) | 'fixed' (dummy / effect code)
    strata: np.ndarray | None
    expected: dict
    excluded: pd.DataFrame         # rows dropped for missing covariates (row_id, word, missing)
    zero_policy: str | None = None

    @property
    def n(self) -> int:
        return len(self.words)

    def X_raw(self, idx=None, center=False) -> np.ndarray:
        R = self.raw.values.astype(float) if idx is None else self.raw.values.astype(float)[idx]
        if center:
            R = R - R.mean(0)
        return np.column_stack([np.ones(len(R)), R])

    def X_std(self, idx=None) -> np.ndarray:
        R = self.raw.values.astype(float) if idx is None else self.raw.values.astype(float)[idx]
        cols = []
        for j, t in enumerate(self.terms):
            x = R[:, j]
            if self.kinds[t] == 'z':
                sd = x.std(ddof=1)
                x = (x - x.mean()) / sd if sd > 0 else np.full_like(x, np.nan)
            cols.append(x)
        return np.column_stack([np.ones(len(R))] + cols)

    def restrict(self, words_available) -> 'Design':
        keep = np.isin(self.words, np.asarray(list(words_available)))
        return Design(self.analysis, self.spec, self.status, self.words[keep], self.terms,
                      self.raw[keep].reset_index(drop=True), self.kinds,
                      None if self.strata is None else self.strata[keep], self.expected, self.excluded,
                      self.zero_policy)


def _complete_cases(r: pd.DataFrame, cols: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    miss = r[cols].isna()
    gone = r[miss.any(axis=1)]
    excluded = pd.DataFrame({'row_id': gone['row_id'].values, 'word': gone['word'].values,
                             'missing': [', '.join(c for c in cols if pd.isna(v[c])) for _, v in gone.iterrows()]})
    return r[~miss.any(axis=1)], excluded


def design_stage1_A(stage1: pd.DataFrame, spec: str = 'regression') -> Design:
    cov = ['wn_n_all', 'log_subtlex', 'length', 'concreteness'] + {
        'regression': [], 'explor_sim_top2': ['sim_top2'], 'regression+n_subtokens': ['n_subtokens']}[spec]
    r, excl = _complete_cases(stage1[stage1['appendix'] == 'A'], cov + ['ambiguity'])
    raw = pd.DataFrame({'ambiguity': (r['ambiguity'] == 'ambiguous').astype(float).values,
                        **{c: r[c].values.astype(float) for c in cov}})
    status = {'regression': 'confirmatory', 'explor_sim_top2': 'exploratory', 'regression+n_subtokens': 'robustness'}
    return Design('stage1_A', spec, status[spec], r['word'].values.astype(str), list(raw.columns), raw,
                  {'ambiguity': 'fixed', **{c: 'z' for c in cov}}, r['ambiguity'].values.astype(str),
                  {'ambiguity': '+', 'wn_n_all': '-', 'log_subtlex': '-'}, excl)


def design_stage1_BC(stage1: pd.DataFrame, appendix: str, spec: str = 'factorial') -> Design:
    cov = {'factorial': [], 'factorial+covariates': ['log_subtlex', 'length', 'concreteness'],
           'factorial+n_subtokens': ['n_subtokens']}[spec]
    r, excl = _complete_cases(stage1[stage1['appendix'] == appendix], ['ambiguity', 'senses'] + cov)
    a = np.where(r['ambiguity'] == 'ambiguous', 0.5, -0.5)
    s = np.where(r['senses'] == 'many', 0.5, -0.5)
    raw = pd.DataFrame({'ambiguity': a, 'senses': s, 'ambiguity:senses': a * s,
                        **{c: r[c].values.astype(float) for c in cov}})
    kinds = {'ambiguity': 'fixed', 'senses': 'fixed', 'ambiguity:senses': 'fixed', **{c: 'z' for c in cov}}
    exp = {'B': {'senses': '-', 'ambiguity': '0'}, 'C': {'senses': '-', 'ambiguity': '+'}}[appendix]
    status = 'confirmatory' if spec == 'factorial' else 'robustness'
    return Design(f'stage1_{appendix}', spec, status, r['word'].values.astype(str), list(raw.columns), raw, kinds,
                  (r['ambiguity'].astype(str) + '/' + r['senses'].astype(str)).values, exp, excl)


def design_stage2(rows: pd.DataFrame, spec: str, policy: str) -> Design:
    cols, status = STAGE2_SPECS[spec]
    r, excl = _complete_cases(rows, cols)
    raw = pd.DataFrame({c: r[c].values.astype(float) for c in cols})
    return Design('stage2', spec, status, r['word'].values.astype(str), cols, raw, {c: 'z' for c in cols}, None,
                  {'gcide_etymologies': '+', 'wn_n_all': '-', 'log_subtlex': '-'}, excl, policy)


def design_exclusions(samples: Samples, cfg) -> pd.DataFrame:
    """Rows dropped by complete-case analysis in every model-independent design."""
    designs = [design_stage1_A(samples.stage1), design_stage1_A(samples.stage1, 'explor_sim_top2')]
    designs += [design_stage1_BC(samples.stage1, a, s) for a in 'BC' for s in ('factorial', 'factorial+covariates')]
    designs += [design_stage2(samples.stage2[cfg.gcide_zero_policy], s, cfg.gcide_zero_policy)
                for s in STAGE2_SPECS if s != 'robust_n_subtokens']
    out = [d.excluded.assign(scope='design', stage=d.analysis, step=f'{d.spec}: complete cases')
           for d in designs if len(d.excluded)]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# LAYERWISE FITS AND DESIGN-PRESERVING BOOTSTRAP
# ════════════════════════════════════════════════════════════════════════════
def _resample(design: Design, rng) -> np.ndarray:
    if design.strata is None:
        return rng.integers(0, design.n, design.n)
    idx = []
    for s in np.unique(design.strata):
        w = np.where(design.strata == s)[0]
        idx.append(w[rng.integers(0, len(w), len(w))])
    return np.concatenate(idx)


def boot_std_betas(design: Design, Ys: list, B: int, seed: int) -> list:
    """Bootstrap of standardised coefficients (terms × layers) for one or more
    outcome matrices with the SAME word resample (joint across layers; paired
    across conditions); stratified within the design cells. Returns (B, k, L) arrays."""
    rng = np.random.default_rng(seed)
    out = [np.full((B, len(design.terms), Y.shape[1]), np.nan) for Y in Ys]
    for b in range(B):
        idx = _resample(design, rng)
        X = design.X_std(idx)
        if not np.isfinite(X).all() or np.linalg.matrix_rank(X) < X.shape[1]:
            continue
        for i, Y in enumerate(Ys):
            Z = zscore_cols(Y[idx])
            good = np.isfinite(Z).all(0)
            if good.any():
                out[i][b][:, good] = ols_beta_only(X, Z[:, good])[1:]
    return out


def std_betas(design: Design, Y: np.ndarray) -> np.ndarray:
    return ols_multi(design.X_std(), zscore_cols(Y))['beta'][1:]


def design_ctx(design: Design) -> dict:
    d = {'analysis': design.analysis, 'spec': design.spec, 'spec_status': design.status,
         'zero_policy': design.zero_policy}
    if design.analysis.startswith('stage1_'):
        e, m = APPENDIX_DESIGN[design.analysis[-1]]
        d.update({'appendix': design.analysis[-1], 'experiment': e, 'modality': m,
                  'modality_note': 'spoken items presented as TEXT' if m == 'auditory' else ''})
    return d


def fit_layerwise(design: Design, Y: np.ndarray, depth: np.ndarray, ctx: dict, alpha: float,
                  boot: np.ndarray | None = None, include_intercept: bool = False,
                  center_raw: bool = False) -> pd.DataFrame:
    """Per-layer OLS (raw units: bits; standardised: SD units) with HC3 SEs,
    BH-FDR across layers per term, VIF, construct labels and readout type."""
    if not np.isfinite(design.X_std()).all():
        return pd.DataFrame()          # a predictor is constant in this sample: not identifiable
    fr = ols_multi(design.X_raw(center=center_raw), Y)
    fs = ols_multi(design.X_std(), zscore_cols(Y))
    tcrit = stats.t.ppf(1 - alpha / 2, fr['df']) if fr['df'] > 0 else np.nan
    names = (['(intercept)'] if include_intercept else []) + list(design.terms)
    offs = 0 if include_intercept else 1
    vif = vif_table(design.raw.values.astype(float), design.terms) if len(design.terms) > 1 else {}
    L = Y.shape[1]
    ro = readout_type(np.arange(1, L + 1), L)
    rows = []
    for jj, term in enumerate(names):
        j = jj + offs
        b, se, sh = fr['beta'][j], fr['se'][j], fr['se_hc3'][j]
        with np.errstate(invalid='ignore', divide='ignore'):
            t_h, t_c = b / sh, b / se
        p_h, p_c = p_two_sided_t(t_h, fr['df']), p_two_sided_t(t_c, fr['df'])
        bs, shs = fs['beta'][j], fs['se_hc3'][j]
        bl = bh = np.full(L, np.nan)
        if boot is not None and term in design.terms:
            bl, bh = percentile_ci(boot[:, design.terms.index(term), :], alpha)
        construct, cstatus = TERM_CONSTRUCT.get(term, ('', ''))
        es = design.expected.get(term, '')
        for l in range(L):
            rows.append(dict(ctx, **design_ctx(design), layer_column=l + 1, block_index=l, depth=float(depth[l]),
                             readout_type=ro[l], term=term, construct=construct, construct_status=cstatus,
                             n=design.n, n_excluded_missing=len(design.excluded),
                             beta=b[l], se=se[l], se_hc3=sh[l], t_hc3=t_h[l], p_hc3=p_h[l], p_classical=p_c[l],
                             ci_low=b[l] - tcrit * sh[l], ci_high=b[l] + tcrit * sh[l],
                             beta_std=bs[l], se_std_hc3=shs[l], ci_std_low=bs[l] - tcrit * shs[l],
                             ci_std_high=bs[l] + tcrit * shs[l], boot_ci_std_low=bl[l], boot_ci_std_high=bh[l],
                             r2=fr['r2'][l], df_resid=fr['df'], vif=vif.get(term, np.nan), expected_sign=es,
                             units='beta: bits per predictor unit; beta_std: SD(y) per SD(x) or per factor contrast'))
    df = pd.DataFrame(rows)
    df['p_fdr_layers'] = df.groupby('term', sort=False)['p_hc3'].transform(lambda s: bh_fdr(s.values))
    return df


def add_global_fdr(df: pd.DataFrame, group_cols) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [c for c in group_cols if c in df]
    df['p_fdr_global_term'] = df.groupby(cols, sort=False, dropna=False)['p_hc3'].transform(
        lambda s: bh_fdr(s.values))
    return df


def region_mask(depth: np.ndarray, region: str) -> np.ndarray:
    a, b = REGIONS[region]
    return (depth > a + 1e-12) & (depth <= b + 1e-12)


def summary_values(v: np.ndarray, depth: np.ndarray, summary: str) -> np.ndarray:
    """v: (..., L) → depth-region mean or output-layer value."""
    if summary == 'output':
        return v[..., -1]
    m = region_mask(depth, summary)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(v[..., m], -1) if m.any() else np.full(v.shape[:-1], np.nan)


def depth_slope(v: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Least-squares slope of v (..., L) on normalised depth."""
    d = depth - depth.mean()
    return ((v - np.nanmean(v, -1, keepdims=True)) * d).sum(-1) / (d * d).sum()


def hypothesis_values(kind: str, terms: tuple, design: Design, beta: np.ndarray, depth: np.ndarray):
    """beta: (..., k, L). Yields (summary, value array over leading dims)."""
    j = [design.terms.index(t) for t in terms]
    if kind in ('slope', 'slope_contrast'):
        o = [ORIENT[t] * beta[..., jj, :] for t, jj in zip(terms, j)]
        yield 'all blocks (slope per unit depth)', \
            depth_slope(o[0], depth) - (depth_slope(o[1], depth) if kind == 'slope_contrast' else 0.0)
        return
    v = beta[..., j[0], :] - (beta[..., j[1], :] if kind == 'contrast' else 0.0)
    for s in SUMMARIES:
        yield s, summary_values(v, depth, s)


def evaluate_hypotheses(designs: dict, points: dict, boots: dict, depth: np.ndarray, ctx: dict, cfg) -> pd.DataFrame:
    """Prespecified confirmatory hypotheses for one model × condition × aggregate."""
    rows = []
    for hid, analysis, kind, terms, expected, family, source in HYPOTHESES:
        if analysis not in designs:
            continue
        d = designs[analysis]
        pt = dict(hypothesis_values(kind, terms, d, points[analysis], depth))
        for summary, bv in hypothesis_values(kind, terms, d, boots[analysis], depth):
            est = float(pt[summary])
            bv = bv[np.isfinite(bv)]
            se = float(np.std(bv, ddof=1)) if len(bv) > 2 else np.nan
            lo, hi = percentile_ci(bv, cfg.alpha) if len(bv) > 2 else (np.nan, np.nan)
            lo90, hi90 = percentile_ci(bv, 2 * cfg.alpha) if len(bv) > 2 else (np.nan, np.nan)
            rows.append(dict(ctx, hypothesis=hid, family=family, analysis=analysis, kind=kind,
                             terms=' − '.join(terms), expected=expected, source_statement=source, summary=summary,
                             estimate=est, boot_se=se, ci95_low=float(lo), ci95_high=float(hi),
                             ci90_low=float(lo90), ci90_high=float(hi90), p_two_sided=p_normal(est, se),
                             p_tost=p_tost_normal(est, se, cfg.sesoi_std) if kind == 'equivalence' else np.nan,
                             sesoi_std=cfg.sesoi_std if kind == 'equivalence' else np.nan, n=d.n,
                             n_boot_valid=len(bv),
                             # sensitivity (Lakens, 2022): smallest effect the unadjusted two-sided test detects
                             # with 80% power at this SE; TOST can succeed only if z(1−α)·SE < SESOI (estimate 0)
                             mde80=float((stats.norm.ppf(1 - cfg.alpha / 2) + stats.norm.ppf(0.8)) * se),
                             equivalence_attainable=(bool(stats.norm.ppf(1 - cfg.alpha) * se < cfg.sesoi_std)
                                                     if kind == 'equivalence' else None)))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['p_holm'], df['p_tost_holm'] = np.nan, np.nan
    for fam, g in df.groupby('family'):
        p = np.concatenate([g['p_two_sided'].values, g['p_tost'].values])
        adj = holm(p)
        df.loc[g.index, 'p_holm'] = adj[:len(g)]
        df.loc[g.index, 'p_tost_holm'] = adj[len(g):]
    df['evidence'] = [classify(r, cfg.alpha) for r in df.itertuples()]
    return df


def classify(r, alpha: float) -> str:
    sig = np.isfinite(r.p_holm) and r.p_holm < alpha
    if r.kind == 'equivalence':
        if np.isfinite(r.p_tost_holm) and r.p_tost_holm < alpha:
            return 'supported (equivalent to zero within ±SESOI)'
        return 'contradicted (non-zero effect)' if sig else 'inconclusive'
    if not sig:
        return 'inconclusive'
    return 'supported' if np.sign(r.estimate) == (1 if r.expected == '+' else -1) else 'contradicted'


PATTERN_ROWS = {'stage1_B': ('S1B_meanings_null', 'S1B_senses', 'S1B_separation'),
                'stage1_C': ('S1C_meanings', 'S1C_senses', 'S1C_separation'),
                'stage2': ('S2_meanings', 'S2_senses', 'S2_separation')}


def meanings_senses_pattern(h: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Do meanings and senses produce distinct effects, or one shared
    (predictability-like) effect? Classified per design and summary."""
    rows = []
    keys = ['model', 'condition', 'aggregate', 'role']
    for kv, g in h.groupby(keys, sort=False):
        gi = g.set_index(['hypothesis', 'summary'])
        for analysis, (hm, hs, hsep) in PATTERN_ROWS.items():
            for summary in SUMMARIES:
                if (hm, summary) not in gi.index or (hs, summary) not in gi.index or (hsep, summary) not in gi.index:
                    continue
                m, s, sep = gi.loc[(hm, summary)], gi.loc[(hs, summary)], gi.loc[(hsep, summary)]
                sig = lambda r: np.isfinite(r['p_holm']) and r['p_holm'] < alpha
                m_ok = m['evidence'].startswith('supported') or (m['expected'] == '0' and not sig(m))
                if sep['evidence'] == 'supported' and s['evidence'] == 'supported' and m_ok:
                    pat = 'distinct effects in the predicted directions'
                elif sig(sep):
                    pat = 'meanings and senses differ, not in the predicted pattern'
                elif sig(m) and sig(s) and np.sign(m['estimate']) == np.sign(s['estimate']):
                    pat = 'one shared effect (meanings and senses not distinguished)'
                else:
                    pat = 'no detectable difference between meanings and senses'
                rows.append(dict(zip(keys, kv), analysis=analysis, summary=summary,
                                 meanings_estimate=m['estimate'], meanings_evidence=m['evidence'],
                                 senses_estimate=s['estimate'], senses_evidence=s['evidence'],
                                 separation_estimate=sep['estimate'], separation_ci95_low=sep['ci95_low'],
                                 separation_ci95_high=sep['ci95_high'], separation_evidence=sep['evidence'],
                                 pattern=pat))
    return pd.DataFrame(rows)


def secondary_depth_tests(design: Design, point: np.ndarray, boot: np.ndarray, depth: np.ndarray, ctx: dict,
                          alpha: float) -> list:
    """Exploratory depth descriptors: oriented-effect slope and deep − shallow
    contrast, |β| deep − shallow, and peak depth of |β|, each with bootstrap CIs."""
    rows = []
    sh, dp = region_mask(depth, 'shallow'), region_mask(depth, 'deep')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for j, t in enumerate(design.terms):
            if t not in ORIENT:
                continue
            o_p, o_b = ORIENT[t] * point[j], ORIENT[t] * boot[:, j, :]
            a_p, a_b = np.abs(point[j]), np.abs(boot[:, j, :])
            stats_ = {
                'oriented slope on depth': (depth_slope(o_p, depth), depth_slope(o_b, depth)),
                'oriented deep − shallow': (np.nanmean(o_p[dp]) - np.nanmean(o_p[sh]),
                                            np.nanmean(o_b[:, dp], 1) - np.nanmean(o_b[:, sh], 1)),
                '|β| deep − shallow': (np.nanmean(a_p[dp]) - np.nanmean(a_p[sh]),
                                       np.nanmean(a_b[:, dp], 1) - np.nanmean(a_b[:, sh], 1)),
                'peak depth of |β|': (depth[np.nanargmax(a_p)],
                                      np.array([depth[np.nanargmax(r)] if np.isfinite(r).any() else np.nan
                                                for r in a_b]))}
            for name, (est, bv) in stats_.items():
                lo, hi = percentile_ci(bv, alpha)
                se = np.nanstd(bv, ddof=1)
                rows.append(dict(ctx, **design_ctx(design), term=t, statistic=name, estimate=float(est),
                                 boot_ci_low=float(lo), boot_ci_high=float(hi),
                                 p_normal=p_normal(float(est), se) if name != 'peak depth of |β|' else np.nan,
                                 n=design.n))
    return rows


# ════════════════════════════════════════════════════════════════════════════
# PROBES (secondary; per appendix, never pooled across designs)
# ════════════════════════════════════════════════════════════════════════════
def run_probe_target(X_layers, y, binary: bool, groups, cfg, ctx, target, pooling, depth, baseline_X, baseline_names):
    rng = np.random.default_rng([cfg.seed, int(sha256_text(target)[:8], 16)])
    folds = make_group_folds(groups, cfg.n_folds, cfg.seed, labels=y if binary else None)
    yv = np.asarray(y, float)
    yfit = np.where(yv == 1, 1.0, -1.0) if binary else (yv - yv.mean()) / (yv.std() or 1.0)
    Ycols = np.column_stack([yfit] + [yfit[rng.permutation(len(yfit))] for _ in range(cfg.n_permutations)])
    labs = (Ycols > 0).astype(int)
    rows = []
    for k in range(1, len(depth) + 1):
        pred = ridge_oof(X_layers(k), Ycols, folds, cfg.probe_lambda_scale)
        met = auroc_columns(pred, labs) if binary else r2_columns(pred, Ycols)
        obs, null = met[0], met[1:]
        rows.append(dict(ctx, target=target, pooling=pooling, layer_column=k, depth=float(depth[k - 1]),
                         metric='auroc_oof' if binary else 'r2_oof', value=float(obs),
                         perm_null_mean=float(np.mean(null)), perm_null_q95=float(np.quantile(null, 0.95)),
                         p_perm=(1 + np.sum(null >= obs)) / (1 + len(null)), n=len(yv),
                         n_positive=int((yv == 1).sum()) if binary else np.nan, n_folds=cfg.n_folds,
                         lambda_scale=cfg.probe_lambda_scale, n_permutations=cfg.n_permutations))
    df = pd.DataFrame(rows)
    df['p_fdr_layers'] = bh_fdr(df['p_perm'].values)
    bpred = ridge_oof(baseline_X, yfit, folds, cfg.probe_lambda_scale)
    df['baseline_value'] = float(auroc_columns(bpred, (yfit > 0).astype(int))[0] if binary else
                                 r2_columns(bpred, yfit)[0])
    df['baseline_features'] = ', '.join(baseline_names)
    return df


def run_probes(ss: ScoreSet, samples: Samples, cfg) -> list:
    idx = {w: i for i, w in zip(ss.targets.index[ss.targets['align_ok']], ss.targets.loc[ss.targets['align_ok'], 'word'])}
    nsub = ss.n_subtokens()
    s2 = samples.stage2[cfg.gcide_zero_policy]
    targets = []
    for app in 'BC':
        r = samples.stage1[samples.stage1['appendix'] == app]
        targets += [(f'S1{app}_ambiguity', r, (r['ambiguity'] == 'ambiguous').astype(int), True, True),
                    (f'S1{app}_senses', r, (r['senses'] == 'many').astype(int), True, True)]
    targets += [('S2_log_subtlex', s2, s2['log_subtlex'], False, False),
                ('S2_wn_n_all', s2, s2['wn_n_all'], False, True),
                ('S2_gcide_etymologies', s2, s2['gcide_etymologies'], False, True)]
    tabs = []
    for name, frame, y, binary, base_freq in targets:
        m = frame['word'].isin(idx.keys()).values & np.isfinite(np.asarray(y, float))
        if base_freq:
            m &= frame['log_subtlex'].notna().values
        fr, yv = frame[m], np.asarray(y, float)[m]
        if len(fr) < 4 * cfg.n_folds or (binary and min((yv == 1).sum(), (yv == 0).sum()) < cfg.n_folds):
            continue
        ridx = np.array([idx[w] for w in fr['word']])
        order = np.argsort(ridx)
        ridx, fr, yv = ridx[order], fr.iloc[order], yv[order]
        names = ['length', 'n_subtokens'] + (['log_subtlex'] if base_freq else [])
        baseX = np.column_stack([fr['length'].values, nsub.loc[fr['word']].values] +
                                ([fr['log_subtlex'].values] if base_freq else [])).astype(float)
        for pooling, R in ss.reps.items():
            tabs.append(run_probe_target(lambda k, R=R, ridx=ridx: np.asarray(R[ridx, k, :], np.float64), yv, binary,
                                         fr['word'].values, cfg, dict(ss.ctx(None), status='secondary'), name,
                                         pooling, ss.depth, baseX, names))
    return tabs


# ════════════════════════════════════════════════════════════════════════════
# MATCHED ROBUSTNESS (model-independent matching, layerwise paired effects)
# ════════════════════════════════════════════════════════════════════════════
def build_matches(rows: pd.DataFrame, caliper: float, min_pairs: int) -> dict:
    out = {}
    for name, spec in MATCH_CONTRASTS.items():
        cov = spec['covariates']
        r = rows.dropna(subset=cov + ['gcide_etymologies', 'wn_n_all', 'log_subtlex']).reset_index(drop=True)
        if name == 'meanings':
            t_m, c_m = r['gcide_etymologies'] >= 2, r['gcide_etymologies'] == 1
        else:
            col = 'wn_n_all' if name == 'senses' else 'log_subtlex'
            lo, hi = r[col].quantile(1 / 3), r[col].quantile(2 / 3)
            t_m, c_m = (r[col] >= hi) & ~(r[col] <= lo), (r[col] <= lo) & ~(r[col] >= hi)
        T, C = r[t_m].reset_index(drop=True), r[c_m].reset_index(drop=True)
        pooled = np.sqrt((T[cov].var(ddof=1).values + C[cov].var(ddof=1).values) / 2)
        mt, mc, rej = greedy_caliper_match(T[cov].values, C[cov].values, caliper, pooled)
        info = {'definition': spec['definition'], 'covariates': cov, 'caliper_sd': caliper,
                'n_treated_pool': len(T), 'n_control_pool': len(C), 'n_pairs': len(mt), 'n_rejected': rej,
                'status': 'RUN' if len(mt) >= min_pairs else f'NOT_RUN (only {len(mt)} pairs < {min_pairs})'}
        bal = pd.DataFrame([{'contrast': name, 'covariate': c,
                             'smd_before': standardized_mean_difference(T[c], C[c], pooled[j]),
                             'smd_after': standardized_mean_difference(T[c].values[mt], C[c].values[mc], pooled[j])}
                            for j, c in enumerate(cov)])
        pairs = pd.DataFrame({'contrast': name, 'pair': np.arange(len(mt)),
                              'treated_row_id': T['row_id'].values[mt], 'treated_word': T['word'].values[mt],
                              'control_row_id': C['row_id'].values[mc], 'control_word': C['word'].values[mc]})
        out[name] = {'info': info, 'balance': bal, 'pairs': pairs}
    return out


def paired_layer_table(D: np.ndarray, depth: np.ndarray, ctx: dict, alpha: float, label: str) -> list:
    """Per layer: mean of paired differences D (n × L), t-based CI, p, BH-FDR."""
    n = D.shape[0]
    mu, sd = np.nanmean(D, 0), np.nanstd(D, 0, ddof=1)
    se = sd / np.sqrt(n)
    with np.errstate(invalid='ignore', divide='ignore'):
        t = mu / se
    p = p_two_sided_t(t, n - 1)
    fdr = bh_fdr(p)
    tc = stats.t.ppf(1 - alpha / 2, n - 1)
    return [dict(ctx, quantity=label, layer_column=l + 1, depth=float(depth[l]), n=n, mean_diff_bits=mu[l],
                 sd_diff=sd[l], ci_low=mu[l] - tc * se[l], ci_high=mu[l] + tc * se[l], t=t[l], p=p[l],
                 p_fdr_layers=fdr[l], dz=mu[l] / sd[l] if sd[l] > 0 else np.nan) for l in range(D.shape[1])]


# ════════════════════════════════════════════════════════════════════════════
# REPRESENTATION DISTANCE
# ════════════════════════════════════════════════════════════════════════════
def cosine_distance_by_layer(A, B) -> np.ndarray:
    """A, B: (n, L+1, d) → (n, L+1) cosine distance, computed layer by layer."""
    n, L1, _ = A.shape
    out = np.empty((n, L1))
    for k in range(L1):
        a = np.asarray(A[:, k, :], np.float32)
        b = np.asarray(B[:, k, :], np.float32)
        with np.errstate(invalid='ignore', divide='ignore'):
            out[:, k] = 1.0 - (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return out


def word_rep_index(ss: ScoreSet) -> dict:
    t = ss.targets
    return {w: i for i, w in zip(t.index[t['align_ok']], t.loc[t['align_ok'], 'word'])}


def pair_relation(a: pd.Series, b: pd.Series) -> str:
    """Relation of two annotated contexts of one word, from the SUPPLIED ids only."""
    if a['meaning_id'] != b['meaning_id']:
        return 'different_meaning'
    if a['sense_id'] and b['sense_id']:
        return 'same_sense' if a['sense_id'] == b['sense_id'] else 'different_sense_same_meaning'
    return 'same_meaning_sense_unannotated'


HP_CONTRASTS = [('different_meaning', 'different_sense_same_meaning'), ('different_meaning', 'same_sense'),
                ('different_sense_same_meaning', 'same_sense')]


def homonymy_polysemy_distance(css: ScoreSet, cfg, ctx: dict, context_df: pd.DataFrame) -> tuple[list, list, list]:
    """Cosine distance between the target's representations in two annotated
    contexts. Key contrast: different-meaning (homonymy-like) vs same-meaning
    different-sense (polysemy-like) pairs; cluster bootstrap over words. Pairs
    whose two context templates recur across words with only one relation type
    are confounded (the contrast would compare templates, not meanings): their
    share is reported and a majority makes the contrast not interpretable."""
    t = css.targets
    usable = t['align_ok'] & (t['meaning_id'] != '') & (t['n_pre_target_tokens'] > 0)
    tmpl = context_df.set_index(['word', 'context_id'])['text']
    pairs = []
    for w, g in t[usable].groupby('word'):
        ii = g.index.values
        for a in range(len(ii)):
            for b in range(a + 1, len(ii)):
                ta, tb = t.loc[ii[a]], t.loc[ii[b]]
                pairs.append((w, ii[a], ii[b], pair_relation(ta, tb),
                              tuple(sorted((tmpl[(w, ta['context_id'])], tmpl[(w, tb['context_id'])]))),
                              (ta['n_pre_target_tokens'] + tb['n_pre_target_tokens']) / 2,
                              abs(ta['n_pre_target_tokens'] - tb['n_pre_target_tokens']),
                              (ta['pre_target_mean_surprisal_bits'] + tb['pre_target_mean_surprisal_bits']) / 2))
    rows, prof, balance = [], [], []
    if not pairs:
        return rows, prof, balance
    P = pd.DataFrame(pairs, columns=['word', 'i', 'j', 'relation', 'templates', 'mean_pre_tokens',
                                     'abs_diff_pre_tokens', 'mean_pre_target_surprisal_bits'])
    by_tpl = P.groupby('templates').agg(n_words=('word', 'nunique'), n_relations=('relation', 'nunique'))
    P['template_confounded'] = P['templates'].map((by_tpl['n_words'] >= 2) & (by_tpl['n_relations'] == 1))
    confounded = float(P['template_confounded'].mean())
    for rel, g in P.groupby('relation'):
        for col in ('mean_pre_tokens', 'abs_diff_pre_tokens', 'mean_pre_target_surprisal_bits', 'template_confounded'):
            balance.append(dict(ctx, group=f'pairs: {rel}', covariate=col, mean=float(g[col].astype(float).mean()),
                                sd=float(g[col].astype(float).std(ddof=1)), n=len(g), smd=np.nan))
    for pooling, R in css.reps.items():
        Dist = cosine_distance_by_layer(R[P['i'].values], R[P['j'].values])[:, 1:]
        L = Dist.shape[1]
        wm = pd.DataFrame(Dist).assign(word=P['word'].values, relation=P['relation'].values) \
            .groupby(['relation', 'word']).mean()
        words = sorted(P['word'].unique())
        rng = np.random.default_rng(cfg.seed)
        boots = [rng.choice(len(words), len(words)) for _ in range(cfg.n_bootstrap)]
        for ra, rb in HP_CONTRASTS:
            if ra not in wm.index.get_level_values(0) or rb not in wm.index.get_level_values(0):
                continue
            A, Bm = wm.loc[ra], wm.loc[rb]

            def stat(sel_words):
                a = A.reindex([w for w in sel_words if w in A.index])
                b = Bm.reindex([w for w in sel_words if w in Bm.index])
                return a.mean(axis=0).values - b.mean(axis=0).values
            est = stat(words)
            bv = np.array([stat([words[i] for i in bi]) for bi in boots])
            lo, hi = percentile_ci(bv, cfg.alpha)
            se = np.nanstd(bv, 0, ddof=1)
            for l in range(L):
                prof.append(dict(ctx, pooling=pooling, contrast=f'{ra} − {rb}', layer_column=l + 1,
                                 depth=float(css.depth[l]), estimate=est[l], ci95_low=lo[l], ci95_high=hi[l],
                                 p_normal=p_normal(est[l], se[l]), n_words_a=len(A), n_words_b=len(Bm)))
            for s in SUMMARIES:
                ev, bs_ = summary_values(est, css.depth, s), summary_values(bv, css.depth, s)
                slo, shi = percentile_ci(bs_, cfg.alpha)
                rows.append(dict(ctx, pooling=pooling, contrast=f'{ra} − {rb}', summary=s, estimate=float(ev),
                                 ci95_low=float(slo), ci95_high=float(shi),
                                 p_two_sided=p_normal(float(ev), float(np.nanstd(bs_, ddof=1))), expected='+',
                                 n_words_a=len(A), n_words_b=len(Bm), template_confounded_pair_share=confounded))
    return rows, prof, balance


# ════════════════════════════════════════════════════════════════════════════
# LIST PROTOCOL (secondary): order, position and surrounding-item effects
# ════════════════════════════════════════════════════════════════════════════
def list_diagnostics(ss_list: ScoreSet, ss_iso: ScoreSet, samples: Samples, aggregate: str, alpha: float):
    t = ss_list.targets
    pos = pd.to_numeric(t['list_position'], errors='coerce').values
    rep = pd.to_numeric(t['repeat'], errors='coerce').values
    M = ss_list.surprisal[aggregate][:, 1:]
    iso = ss_iso.word_matrix(aggregate)
    rows = []
    for l in range(ss_list.n_layers):
        g = t['align_ok'].values & np.isfinite(M[:, l]) & np.isfinite(pos)
        rho, p = stats.spearmanr(pos[g], M[g, l]) if g.sum() > 3 else (np.nan, np.nan)
        wide = pd.DataFrame({'word': t['word'].values[g], 'rep': rep[g], 'y': M[g, l]}) \
            .pivot_table(index='word', columns='rep', values='y', aggfunc='mean')
        rel = np.nan
        if wide.shape[1] >= 2:
            cs = wide.corr().values
            rel = float(np.nanmean(cs[np.triu_indices_from(cs, 1)]))
        mw = wide.mean(axis=1)
        common = mw.index.intersection(iso.index)
        rows.append(dict(ss_list.ctx(aggregate), layer_column=l + 1, depth=float((l + 1) / ss_list.n_layers),
                         spearman_position=rho, p_position=p, between_repeat_reliability=rel,
                         r_with_isolated=_r(mw.loc[common], iso.loc[common, l + 1]) if len(common) > 3 else np.nan,
                         n_targets=int(g.sum())))
    order_tab = pd.DataFrame(rows)
    order_tab['p_position_fdr_layers'] = bh_fdr(order_tab['p_position'].values)
    # surrounding-item model: list − isolated surprisal of the same word, clustered by word
    freq = pd.concat([samples.stage1, samples.stage2_base]).drop_duplicates('word').set_index('word')['log_subtlex']
    nsub = ss_iso.n_subtokens()
    prev = t['previous_word'].values
    keep = (t['align_ok'].values & (pos >= 1) & t['word'].isin(iso.index).values &
            pd.Series(prev).map(freq).notna().values & pd.Series(prev).isin(nsub.index).values)
    surround = pd.DataFrame()
    if keep.sum() > 10:
        D = M[keep] - iso.loc[t['word'].values[keep]].values
        X = np.column_stack([np.ones(keep.sum()), zscore_cols(np.column_stack([
            pos[keep], pd.Series(prev[keep]).map(freq).values, pd.Series(prev[keep]).map(nsub).values]))])
        fit = ols_cluster(X, D, t['word'].values[keep])
        names = ['(list − isolated, at mean)', 'list position', 'previous item log_subtlex',
                 'previous item n_subtokens']
        tc = stats.t.ppf(1 - alpha / 2, fit['df'])
        srows = []
        for j, nm in enumerate(names):
            b, se = fit['beta'][j], fit['se_cluster'][j]
            with np.errstate(invalid='ignore', divide='ignore'):
                tt = b / se
            p = p_two_sided_t(tt, fit['df'])
            fdr = bh_fdr(p)
            for l in range(D.shape[1]):
                srows.append(dict(ss_list.ctx(aggregate), term=nm, layer_column=l + 1,
                                  depth=float((l + 1) / ss_list.n_layers), beta_bits_per_sd=b[l], se_cluster=se[l],
                                  ci_low=b[l] - tc * se[l], ci_high=b[l] + tc * se[l], t=tt[l], p=p[l],
                                  p_fdr_layers=fdr[l], n_targets=int(keep.sum()), n_clusters=fit['n_clusters']))
        surround = pd.DataFrame(srows)
    return order_tab, surround


# ════════════════════════════════════════════════════════════════════════════
# TOKENISATION AUDITS
# ════════════════════════════════════════════════════════════════════════════
def tokenization_balance(ss: ScoreSet, samples: Samples, cfg) -> list:
    nsub = ss.n_subtokens()
    rows = []
    for app in 'ABC':
        r = samples.stage1[samples.stage1['appendix'] == app].assign(n_subtokens=lambda x: x['word'].map(nsub))
        contrasts = [('ambiguity', r['ambiguity'] == 'ambiguous', r['ambiguity'] == 'unambiguous')]
        if app != 'A':
            contrasts.append(('senses', r['senses'] == 'many', r['senses'] == 'few'))
        for name, a, b in contrasts:
            rows.append(dict(ss.ctx(None), pool=f'appendix {app}', quantity=f'n_subtokens SMD ({name})',
                             value=standardized_mean_difference(r.loc[a, 'n_subtokens'], r.loc[b, 'n_subtokens']),
                             n=int(r['n_subtokens'].notna().sum())))
    s2 = samples.stage2[cfg.gcide_zero_policy].assign(n_subtokens=lambda x: x['word'].map(nsub))
    for c in ('log_subtlex', 'gcide_etymologies', 'wn_n_all', 'length'):
        rows.append(dict(ss.ctx(None), pool='stage2', quantity=f'r(n_subtokens, {c})', value=_r(s2['n_subtokens'], s2[c]),
                         n=int(s2[['n_subtokens', c]].dropna().shape[0])))
    return rows


def tokenization_bias_audit(ss: ScoreSet, samples: Samples, cfg) -> list:
    """Empirical audit: partial correlation of surprisal with subtoken count given
    frequency and length, per aggregate and layer — does any aggregate (incl. the
    end-of-word correction) remove the tokenisation dependence?"""
    nsub = ss.n_subtokens()
    pools = {'published words': samples.stage1.drop_duplicates('word'),
             'stage2 pool': samples.stage2[cfg.gcide_zero_policy]}
    rows = []
    for agg in AGGREGATES:
        W = ss.word_matrix(agg)
        for pname, fr in pools.items():
            fr = fr[fr['word'].isin(W.index)]
            Z = fr[['log_subtlex', 'length']].values.astype(float)
            ns = fr['word'].map(nsub).values.astype(float)
            Y = W.loc[fr['word']].values
            for l in range(Y.shape[1]):
                r, p, n = partial_corr(ns, Y[:, l], Z)
                rows.append(dict(ss.ctx(agg), pool=pname, layer_column=l + 1, depth=float(ss.depth[l]),
                                 partial_r_n_subtokens=r, p=p, n=n))
    out = pd.DataFrame(rows)
    out['p_fdr_layers'] = out.groupby(['aggregate', 'pool'])['p'].transform(lambda s: bh_fdr(s.values))
    diff = ss.surprisal['whole_word'][:, 1:] - ss.surprisal['sum'][:, 1:]
    ok = ss.targets['align_ok'].values
    size = pd.DataFrame([dict(ss.ctx('whole_word − sum'), pool='all targets', layer_column=l + 1,
                              depth=float(ss.depth[l]), n=int(ok.sum()),
                              eow_correction_mean_bits=float(np.nanmean(diff[ok, l])),
                              eow_correction_sd_bits=float(np.nanstd(diff[ok, l], ddof=1)))
                         for l in range(ss.n_layers)])
    return pd.concat([out, size], ignore_index=True).to_dict('records')


# ════════════════════════════════════════════════════════════════════════════
# RUN ORCHESTRATION — scoring
# ════════════════════════════════════════════════════════════════════════════
def output_dirs(root: str) -> dict:
    d = {k: os.path.join(root, k) for k in ('prompts', 'cache', 'item_scores', 'tables', 'figures', 'logs', 'models')}
    d['root'] = root
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    return d


def tokenizer_diagnostics(sm: ScoringModel, encoded_by_cond: dict) -> dict:
    tok = sm.tokenizer
    dec = lambda ids: [tok.decode([i], clean_up_tokenization_spaces=False) for i in ids]
    out = {'model': sm.spec.alias, 'tokenizer_class': type(tok).__name__,
           'special_tokens': {k: str(v) for k, v in tok.special_tokens_map.items()},
           'bos_token_id': tok.bos_token_id, 'eos_token_id': tok.eos_token_id, 'pad_token_id': tok.pad_token_id,
           'padding': 'manual right padding with attention mask', 'chat_template_applied': sm.chat,
           'chat_template_kwargs': sm.template_kwargs, 'boundary_set_size': int(len(sm.boundary_ids)),
           'conditions': {}}
    for cname, enc in encoded_by_cond.items():
        flat = [(u, t) for u in enc for t in u.targets]
        ok = [x for x in flat if x[1]['ok']]
        picks = {}
        if ok:
            picks['shortest'] = min(ok, key=lambda x: (len(x[1]['word']), x[1]['word']))
            picks['longest'] = max(ok, key=lambda x: (len(x[1]['word']), x[1]['word']))
            picks['most_subtokens'] = max(ok, key=lambda x: (x[1]['n_sub'], x[1]['word']))
            for name, cond in (('single_token_example', lambda t: t['n_sub'] == 1),
                               ('multi_token_example', lambda t: t['n_sub'] >= 2)):
                hit = [x for x in ok if cond(x[1])]
                if hit:
                    picks[name] = hit[0]
        ex = {}
        for name, (u, t) in picks.items():
            a, b = max(0, t['t0'] - 6), min(len(u.input_ids), t['t1'] + 3)
            ex[name] = {'unit_id': u.unit_id, 'word': t['word'], 'target_token_span': [t['t0'], t['t1']],
                        'prediction_positions': list(range(t['t0'] - 1, t['t1'])), 'target_ids': t['target_ids'],
                        'target_tokens': t['target_tokens'], 'decoded_target': t['decoded'],
                        'leading_ws_absorbed': t['leading_ws_absorbed'],
                        'first_token_is_boundary': t['first_token_is_boundary'],
                        'window_start_index': a, 'window_tokens': dec(u.input_ids[a:b])}
        first_ids = enc[0].input_ids[:8] if enc else []
        out['conditions'][cname] = {
            'n_targets': len(flat), 'n_aligned': len(ok),
            'alignment_failures': pd.Series([t['reason'] for _, t in flat if not t['ok']]).value_counts().to_dict(),
            'offset_methods': pd.Series([t['offset_method'] for _, t in flat]).value_counts().to_dict(),
            'n_subtokens_distribution': {int(k): int(v) for k, v in
                                         pd.Series([t['n_sub'] for _, t in ok]).value_counts().sort_index().items()},
            'n_leading_ws_absorbed': int(sum(t['leading_ws_absorbed'] for _, t in ok)),
            'first_input_tokens': dec(first_ids),
            'n_bos_tokens_in_first_unit': int(sum(i == tok.bos_token_id for i in enc[0].input_ids))
            if enc and tok.bos_token_id is not None else 0, 'examples': ex}
    return out


def prompt_examples(sm: ScoringModel, cond: PromptCondition, encoded: list[EncodedUnit], units: list[Unit]) -> dict:
    u, e = units[0], encoded[0]
    return {'model': sm.spec.alias, 'condition': cond.name, 'protocol': cond.protocol, 'role': cond.role,
            'wording_source': cond.wording_source, 'status': cond.status, 'template_sha256': cond.template_sha256,
            'input_construction': ('INSTRUCT: tokenizer.apply_chat_template([{"role": "user", "content": <text>}], '
                                   'add_generation_prompt=True); no system message added by this script (the template '
                                   'may insert its own default)' if sm.chat else
                                   'BASE: the complete text tokenised in one call (add_special_tokens=True)'),
            'chat_template_kwargs': sm.template_kwargs if sm.chat else None, 'example_unit_id': u.unit_id,
            'example_text_or_user_content': u.text, 'example_rendered_model_input': e.full_text,
            'example_input_ids': e.input_ids, 'example_targets': e.targets, 'n_units': len(units),
            'all_rendered_sha256': sha256_json([x.full_text for x in encoded])}


def process_model(spec: ModelSpec, cfg, conds: dict, active: list, samples: Samples, context_df, dirs) -> dict:
    status = {'alias': spec.alias, 'repo_id': spec.repo_id, 'conditions': {}, 'status': 'OK'}
    t_start = time.time()
    sm = ScoringModel(spec, cfg, load_weights=False)
    plan, encoded_by = {}, {}
    for cname in active:
        cond = conds[cname]
        units = build_units(cond, cfg, samples, context_df)
        encoded = [sm.encode_unit(u) for u in units]
        encoded_by[cname] = encoded
        flat = [t for u in encoded for t in u.targets]
        fail_rate = float(1 - np.mean([t['ok'] for t in flat])) if flat else 1.0
        key = {'script_version': SCRIPT_VERSION, 'model_alias': spec.alias, 'repo_id': spec.repo_id,
               'requested_revision': spec.revision, 'resolved_revision': sm.resolved_revision,
               'variant': spec.variant, 'tokenizer': sm.tokenizer_fingerprint(), 'condition': cname,
               'protocol': cond.protocol, 'template_sha256': cond.template_sha256,
               'context_sha256': cfg.context_sha256 if cond.protocol.startswith('context_') else None,
               'units_sha256': units_fingerprint(encoded), 'dtype': cfg.dtype, 'quantization': cfg.quantization,
               'attn_implementation_requested': cfg.attn_implementation, 'num_hidden_layers': sm.num_layers,
               'hidden_size': sm.hidden_size, 'lens_method': LENS_METHOD, 'boundary_definition': BOUNDARY_DEFINITION,
               'representations_saved': cfg.save_representations if cond.protocol != 'list' else 'none',
               'aggregate_definitions_sha256': sha256_json(AGGREGATE_DEFINITIONS)}
        cache = CacheEntry(dirs['root'], spec.alias, cname)
        st = {'protocol': cond.protocol, 'n_units': len(units), 'n_targets': len(flat),
              'alignment_failure_rate': fail_rate}
        atomic_write_json(os.path.join(dirs['prompts'], f"{safe_name(spec.alias)}_{safe_name(cname)}_prompt_examples.json"),
                          prompt_examples(sm, cond, encoded, units))
        if fail_rate > cfg.max_alignment_failure_rate:
            st['status'] = (f'FAILED: target alignment failed for {fail_rate:.1%} of targets '
                            f'(> {cfg.max_alignment_failure_rate:.1%}); see tokenizer diagnostics')
        elif cache.exists() and not cfg.overwrite:
            mism = cache.mismatches(key)
            st['status'] = 'CACHED (manifest matches)' if not mism else \
                f'FAILED: stale cache — manifest keys differ {mism}; refusing to reuse or overwrite (use --overwrite)'
        else:
            plan[cname] = (cond, encoded, key, analysis_key(cfg, cond, units), cache)
        status['conditions'][cname] = st
    atomic_write_json(os.path.join(dirs['prompts'], f"{safe_name(spec.alias)}_tokenizer_diagnostics.json"),
                      tokenizer_diagnostics(sm, encoded_by))
    if plan:
        sm.load_weights()
        prim = encoded_by.get(cfg.primary_condition) or next(iter(encoded_by.values()))
        sm.verify_readout(prim[:8])
        sm.check_padding(prim)
        if not sm.lens_ok:
            logger.error(f"{spec.alias}: layerwise surprisal UNAVAILABLE — {sm.lens_check.get('reason')}")
        desc = sm.describe()
        atomic_write_json(os.path.join(dirs['models'], f"{safe_name(spec.alias)}.json"), desc)
        for cname, (cond, encoded, key, akey, cache) in plan.items():
            want_reps = cond.protocol != 'list' and cfg.save_representations != 'none'
            t0 = time.time()
            sc = sm.score_encoded(encoded, want_reps=want_reps, desc=f"{spec.alias}/{cname}")
            reps = {p: sc[f'reps_{p}'] for p in ('last', 'mean')
                    if want_reps and cfg.save_representations in ('both', p)}
            cache.save(key, akey, {'model': desc, 'condition': asdict(cond), 'seconds': time.time() - t0,
                                   'lens_status': sm.lens_check.get('status'),
                                   'pooling_rules': {'last': 'hidden state at the LAST target subtoken',
                                                     'mean': 'mean over all target subtoken positions'}},
                       pack_scores(encoded, sc), reps)
            st = status['conditions'][cname]
            st['status'] = 'SCORED' if sm.lens_ok else 'SCORED (surprisal UNAVAILABLE; representations only)'
            st['seconds'] = round(time.time() - t0, 1)
            write_csv(item_layer_frame(load_scoreset(cache, cond, desc)),
                      os.path.join(dirs['item_scores'], safe_name(spec.alias), f"{safe_name(cname)}_item_layer_scores.csv"))
        status['lens'] = sm.lens_check.get('status')
        status['padding'] = sm.padding_check.get('status')
    sm.free()
    status['seconds'] = round(time.time() - t_start, 1)
    if any(str(c.get('status', '')).startswith('FAILED') for c in status['conditions'].values()):
        status['status'] = 'PARTIAL'
    return status


# ════════════════════════════════════════════════════════════════════════════
# RUN ORCHESTRATION — analyses
# ════════════════════════════════════════════════════════════════════════════
def discover_scoresets(cfg, conds: dict, samples: Samples, context_df, dirs) -> tuple[list, list]:
    found, skipped = [], []
    root = os.path.join(dirs['root'], 'cache')
    for mdir in sorted(os.listdir(root)):
        for cdir in sorted(os.listdir(os.path.join(root, mdir))):
            ce = CacheEntry(dirs['root'], mdir, cdir)
            if cdir.endswith('.partial') or not ce.exists():
                continue
            man = ce.manifest()
            cname = man['key']['condition']
            if cname not in conds:
                skipped.append((mdir, cdir, 'condition not configured in this run'))
                continue
            mism = ce.mismatches(analysis_key(cfg, conds[cname], build_units(conds[cname], cfg, samples, context_df)),
                                 'analysis_key')
            if mism:
                skipped.append((mdir, cdir, f'stale cache (keys {mism}); not analysed'))
                continue
            found.append(load_scoreset(ce, conds[cname], man['info']['model']))
    return found, skipped


def _lens_available(ss: ScoreSet) -> bool:
    return bool(np.isfinite(ss.surprisal['sum'][:, 1:]).any())


def run_analyses(cfg, conds: dict, samples: Samples, context_df, dirs) -> dict:
    # Small cells (e.g. --max_items subsets) produce undefined SDs, recorded as NaN.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return _run_analyses(cfg, conds, samples, context_df, dirs)


def _stage_designs(cfg, st1: pd.DataFrame, st2: dict) -> list:
    designs = []
    if cfg.run_stage1:
        designs += [design_stage1_A(st1, s) for s in ('regression', 'regression+n_subtokens', 'explor_sim_top2')]
        designs += [design_stage1_BC(st1, a, s) for a in 'BC'
                    for s in ('factorial', 'factorial+covariates', 'factorial+n_subtokens')]
    if cfg.run_stage2:
        for pol in GCIDE_ZERO_POLICIES:
            designs += [design_stage2(st2[pol], s, pol)
                        for s in (STAGE2_SPECS if pol == cfg.gcide_zero_policy else ['primary'])]
    return designs


def _primary_designs(cfg, samples: Samples) -> list:
    out = []
    if cfg.run_stage1:
        out += [design_stage1_A(samples.stage1), design_stage1_BC(samples.stage1, 'B'),
                design_stage1_BC(samples.stage1, 'C')]
    if cfg.run_stage2:
        out.append(design_stage2(samples.stage2[cfg.gcide_zero_policy], 'primary', cfg.gcide_zero_policy))
    return out


def _condition_contrast(d: Design, Yr: np.ndarray, Yo: np.ndarray, depth: np.ndarray, ctx: dict, cfg) -> list:
    """Paired (same words, same resample) change of the ORIENTED standardised effect
    between two conditions: per layer, per summary, and its depth slope."""
    br, bo = boot_std_betas(d, [Yr, Yo], cfg.n_bootstrap, cfg.seed)
    pr, po = std_betas(d, Yr), std_betas(d, Yo)
    rows = []
    for j, term in enumerate(d.terms):
        if term not in ORIENT:
            continue
        dp, db = ORIENT[term] * (po[j] - pr[j]), ORIENT[term] * (bo[:, j, :] - br[:, j, :])
        base = dict(ctx, **design_ctx(d), term=term, n=d.n)
        for l in range(len(depth)):
            lo, hi = percentile_ci(db[:, l], cfg.alpha)
            rows.append(dict(base, statistic='Δ oriented effect at layer', layer_column=l + 1, depth=float(depth[l]),
                             estimate=float(dp[l]), boot_ci_low=float(lo), boot_ci_high=float(hi),
                             p_normal=p_normal(float(dp[l]), float(np.nanstd(db[:, l], ddof=1)))))
        for s in SUMMARIES:
            ev, bv = summary_values(dp, depth, s), summary_values(db, depth, s)
            lo, hi = percentile_ci(bv, cfg.alpha)
            rows.append(dict(base, statistic=f'Δ oriented effect, {s}', estimate=float(ev), boot_ci_low=float(lo),
                             boot_ci_high=float(hi), p_normal=p_normal(float(ev), float(np.nanstd(bv, ddof=1)))))
        ev, bv = depth_slope(dp, depth), depth_slope(db, depth)
        lo, hi = percentile_ci(bv, cfg.alpha)
        rows.append(dict(base, statistic='Δ oriented depth slope (layer × condition)', estimate=float(ev),
                         boot_ci_low=float(lo), boot_ci_high=float(hi),
                         p_normal=p_normal(float(ev), float(np.nanstd(bv, ddof=1))),
                         prediction='≈ 0 (frequency stays shallow)' if term == 'log_subtlex' else '> 0'))
    return rows


def _run_analyses(cfg, conds: dict, samples: Samples, context_df, dirs) -> dict:
    sets, skipped = discover_scoresets(cfg, conds, samples, context_df, dirs)
    for s in skipped:
        logger.warning(f"analysis skipped {s}")
    res = {'skipped_caches': skipped, 'n_scoresets': len(sets), 'tables': {}, 'not_run': {}, 'figures': []}
    if not sets:
        res['not_run']['all_analyses'] = 'NOT_RUN: no valid cached model outputs'
        return res
    alpha, pa, pc = cfg.alpha, cfg.primary_aggregate, cfg.primary_condition
    L_ = {k: [] for k in ('s1', 's2', 'cells', 'hyp', 'depth', 'inter', 'inter_depth', 'dist', 'probe', 'matched',
                          'list_order', 'list_surround', 'tok_bal', 'tok_bias', 'ctx_eff', 'ctx_bal', 'hp', 'hp_prof',
                          'dom', 'excl')}
    L_['excl'] += [samples.exclusions, design_exclusions(samples, cfg)]
    by_model = {}
    for ss in sets:
        by_model.setdefault(ss.alias, {})[ss.condition] = ss
        bad = ss.targets[~ss.targets['align_ok']]
        if len(bad):
            L_['excl'].append(pd.DataFrame({'scope': 'model', 'stage': '', 'step': 'token alignment failure: ' +
                                            bad['align_reason'], 'row_id': '', 'word': bad['word'],
                                            'model': ss.alias, 'condition': ss.condition}))
    matches = build_matches(samples.stage2[cfg.gcide_zero_policy], cfg.match_caliper_sd, cfg.min_matched_pairs) \
        if cfg.run_stage2 and cfg.run_matched else {}
    aggs = [pa] + [a for a in AGGREGATES if a != pa]
    for ss in sets:
        if not _lens_available(ss):
            res['not_run'][f'{ss.alias}/{ss.condition}/surprisal'] = 'UNAVAILABLE: logit lens failed verification'
            continue
        if ss.protocol.startswith('context_'):
            continue
        nsub = ss.n_subtokens()
        st1 = samples.stage1.assign(n_subtokens=samples.stage1['word'].map(nsub))
        st2 = {p: r.assign(n_subtokens=r['word'].map(nsub)) for p, r in samples.stage2.items()}
        for agg in aggs:
            W = ss.word_matrix(agg)
            role = ('confirmatory' if agg == pa else 'sensitivity (aggregate)') if ss.condition == pc else \
                ('secondary (list protocol)' if ss.protocol == 'list' else 'exploratory (task instruction)')
            ctx = dict(ss.ctx(agg), role=role)
            hd, points, boots = {}, {}, {}
            for d in _stage_designs(cfg, st1, st2):
                d = d.restrict(W.index)
                if d.n < len(d.terms) + 3:
                    continue
                Y = W.loc[d.words].values
                boot = None
                if (d.analysis.startswith('stage1') and d.spec in ('regression', 'factorial')) or \
                        (d.analysis == 'stage2' and d.spec == 'primary' and d.zero_policy == cfg.gcide_zero_policy):
                    boot = boot_std_betas(d, [Y], cfg.n_bootstrap, cfg.seed)[0]
                    hd[d.analysis], points[d.analysis], boots[d.analysis] = d, std_betas(d, Y), boot
                    L_['depth'] += secondary_depth_tests(d, points[d.analysis], boot, ss.depth, ctx, alpha)
                    if d.analysis in ('stage1_B', 'stage1_C'):
                        for cell in sorted(set(d.strata)):
                            m = d.strata == cell
                            L_['cells'] += [dict(ctx, **design_ctx(d), cell=cell, n=int(m.sum()), layer_column=l + 1,
                                                 depth=float(ss.depth[l]), mean_bits=float(Y[m, l].mean()),
                                                 sd_bits=float(Y[m, l].std(ddof=1)) if m.sum() > 1 else np.nan)
                                            for l in range(Y.shape[1])]
                L_['s1' if d.analysis.startswith('stage1') else 's2'].append(
                    fit_layerwise(d, Y, ss.depth, ctx, alpha, boot))
            if hd:
                L_['hyp'].append(evaluate_hypotheses(hd, points, boots, ss.depth, ctx, cfg))
            if agg == pa and ss.protocol == 'isolated' and matches:
                for name, m in matches.items():
                    pr = m['pairs'][m['pairs']['treated_word'].isin(W.index) & m['pairs']['control_word'].isin(W.index)]
                    if m['info']['status'] == 'RUN' and len(pr) >= 3:
                        L_['matched'] += paired_layer_table(W.loc[pr['treated_word']].values -
                                                            W.loc[pr['control_word']].values, ss.depth,
                                                            dict(ctx, contrast=name,
                                                                 definition=m['info']['definition']),
                                                            alpha, 'treated − control (matched pairs)')
        if ss.protocol == 'isolated':
            L_['tok_bias'] += tokenization_bias_audit(ss, samples, cfg)
            if ss.condition == pc:
                L_['tok_bal'] += tokenization_balance(ss, samples, cfg)
        if ss.protocol == 'list' and pc in by_model[ss.alias]:
            o, s = list_diagnostics(ss, by_model[ss.alias][pc], samples, pa, alpha)
            L_['list_order'].append(o)
            L_['list_surround'].append(s)
    for alias, cs in by_model.items():
        ref = cs.get(pc)
        if ref is None or not _lens_available(ref):
            continue
        Wref = ref.word_matrix(pa)
        # ── task-instruction manipulation (isolated frames vs the primary frame) ─
        for oname, oss in cs.items():
            if oname == pc or oss.protocol != 'isolated' or not _lens_available(oss):
                continue
            Woth = oss.word_matrix(pa)
            common = Wref.index.intersection(Woth.index)
            ctx = dict(ref.ctx(pa), condition=oname, reference_condition=pc, manipulation='task instruction',
                       condition_status=oss.cond_status)
            for d in _primary_designs(cfg, samples):
                d = d.restrict(common)
                if d.n < len(d.terms) + 3:
                    continue
                Yr, Yo = Wref.loc[d.words].values, Woth.loc[d.words].values
                tab = fit_layerwise(d, Yo - Yr, ref.depth, ctx, alpha, None, include_intercept=True, center_raw=True)
                tab['term'] = tab['term'].replace({'(intercept)': '(condition main effect)'})
                L_['inter'].append(tab)
                L_['inter_depth'] += _condition_contrast(d, Yr, Yo, ref.depth, ctx, cfg)
            if cfg.run_distance:
                ir, io_ = word_rep_index(ref), word_rep_index(oss)
                cw = sorted(set(ir) & set(io_))
                for pooling in sorted(set(ref.reps) & set(oss.reps)):
                    if len(cw) < 10:
                        break
                    Dm = cosine_distance_by_layer(ref.reps[pooling][[ir[w] for w in cw]],
                                                  oss.reps[pooling][[io_[w] for w in cw]])[:, 1:]
                    dctx = dict(ref.ctx(None), condition=oname, reference_condition=pc, pooling=pooling,
                                measure='cross-task-frame cosine distance of the same word',
                                status='exploratory (task-frame effect on the representation; not a usage distance)')
                    L_['dist'] += [dict(dctx, term='mean distance', layer_column=l + 1, depth=float(ref.depth[l]),
                                        estimate=float(np.nanmean(Dm[:, l])), n=len(cw)) for l in range(Dm.shape[1])]
                    Wd = pd.DataFrame(Dm, index=cw, columns=range(1, ref.n_layers + 1))
                    for d in _primary_designs(cfg, samples):
                        d = d.restrict(Wd.index)
                        if d.n >= len(d.terms) + 3:
                            L_['dist'] += fit_layerwise(d, Wd.loc[d.words].values, ref.depth, dctx, alpha) \
                                .to_dict('records')
        # ── contexts: sentence vs bare word in ONE shared frame ────────────────
        bare, sent = cs.get('context_bare'), cs.get('context_sentence')
        if bare is None or sent is None or not _lens_available(sent):
            continue
        t = sent.targets
        no_pre = t['n_pre_target_tokens'] == 0
        if no_pre.any():
            L_['excl'].append(pd.DataFrame({'scope': 'model', 'stage': 'context', 'row_id': '',
                                            'step': 'context has no pre-target words (a causal model cannot use it '
                                                    'at the target position)', 'word': t.loc[no_pre, 'word'],
                                            'model': alias, 'condition': 'context_sentence'}))
        cctx = dict(sent.ctx(pa), reference_condition='context_bare', manipulation='context (same frame)')
        for col in ('n_pre_target_words', 'n_pre_target_tokens', 'pre_target_mean_surprisal_bits', 'n_sub'):
            v = pd.to_numeric(t.loc[t['align_ok'], col], errors='coerce')
            L_['ctx_bal'].append(dict(cctx, group='all contexts', covariate=col, mean=float(v.mean()),
                                      sd=float(v.std(ddof=1)), n=int(v.notna().sum()), smd=np.nan))
            dl = t.loc[t['align_ok'], 'dominance'].str.lower()
            if (dl == 'dominant').any() and (dl == 'subordinate').any():
                L_['ctx_bal'].append(dict(cctx, group='subordinate vs dominant', covariate=col,
                                          mean=float(v[dl == 'subordinate'].mean() - v[dl == 'dominant'].mean()),
                                          sd=np.nan, n=int(v.notna().sum()),
                                          smd=standardized_mean_difference(v[dl == 'subordinate'], v[dl == 'dominant'])))
        Ws, Wb = sent.word_matrix(pa, mask=~no_pre), bare.word_matrix(pa)
        common = Ws.index.intersection(Wb.index)
        if len(common) >= 3:
            D = Ws.loc[common].values - Wb.loc[common].values
            L_['ctx_eff'] += paired_layer_table(D, sent.depth, cctx, alpha, 'context − bare word (same frame)')
            Wd = pd.DataFrame(D, index=common, columns=range(1, sent.n_layers + 1))
            for d in _primary_designs(cfg, samples):
                d = d.restrict(Wd.index)
                if d.n >= len(d.terms) + 3:
                    L_['ctx_eff'] += fit_layerwise(d, Wd.loc[d.words].values, sent.depth,
                                                   dict(cctx, quantity='context effect regressed on item properties'),
                                                   alpha).to_dict('records')
        if cfg.context_info.get('has_meaning_ids') and sent.reps:
            role = 'primary (human-validated annotations)' if cfg.context_info.get('all_human_validated') \
                else 'exploratory (annotations not validated)'
            hp, prof, hbal = homonymy_polysemy_distance(sent, cfg, dict(sent.ctx(None), role=role), context_df)
            L_['ctx_bal'] += hbal
            if hp:
                hp = pd.DataFrame(hp)
                key = (hp['contrast'] == 'different_meaning − different_sense_same_meaning') & (hp['pooling'] == 'last')
                hp['p_holm'] = np.nan
                hp.loc[key, 'p_holm'] = holm(hp.loc[key, 'p_two_sided'].values)
                hp['evidence'] = np.where(~key, 'descriptive', np.where(
                    hp['p_holm'] < alpha, np.where(hp['estimate'] > 0, 'supported', 'contradicted'), 'inconclusive'))
                hp.loc[key & (hp['template_confounded_pair_share'] >= 0.5), 'evidence'] = \
                    'not interpretable (relation type confounded with context templates shared across words)'
                L_['hp'].append(hp)
                L_['hp_prof'] += prof
        if cfg.context_info.get('has_dominance_labels'):
            dl = t['dominance'].str.lower()
            sub = sent.word_matrix(pa, mask=(dl == 'subordinate').values & ~no_pre.values)
            dom = sent.word_matrix(pa, mask=(dl == 'dominant').values & ~no_pre.values)
            cw = sub.index.intersection(dom.index)
            if len(cw) >= 5:
                L_['dom'] += paired_layer_table(sub.loc[cw].values - dom.loc[cw].values, sent.depth, cctx, alpha,
                                                'subordinate − dominant context')
    if cfg.run_probes:
        for alias, cs in by_model.items():
            if pc in cs and cs[pc].reps:
                L_['probe'] += run_probes(cs[pc], samples, cfg)
    # ── assemble tables ──────────────────────────────────────────────────────
    def cat(lst):
        lst = [x for x in lst if x is not None and len(x)]
        return pd.concat([x if isinstance(x, pd.DataFrame) else pd.DataFrame(x) for x in lst], ignore_index=True) \
            if lst else pd.DataFrame()
    T = res['tables']
    s1 = add_global_fdr(cat(L_['s1']), ('analysis', 'spec', 'term'))
    for app in 'ABC':
        T[f'stage1_published_appendix_{app}'] = s1[s1['analysis'] == f'stage1_{app}'] if len(s1) else pd.DataFrame()
    T['stage1_cell_means'] = pd.DataFrame(L_['cells'])
    hyp = cat(L_['hyp'])
    T['confirmatory_hypothesis_tests'] = hyp[hyp['role'] == 'confirmatory'] if len(hyp) else hyp
    T['sensitivity_hypothesis_tests'] = hyp[hyp['role'] != 'confirmatory'] if len(hyp) else hyp
    T['meanings_vs_senses_pattern'] = meanings_senses_pattern(hyp, alpha) if len(hyp) else pd.DataFrame()
    s2 = add_global_fdr(cat(L_['s2']), ('spec', 'zero_policy', 'term'))
    if len(s2) and len(hyp):
        prof = hyp[hyp['family'] == 'stage1_calibration'].groupby(['model', 'condition', 'aggregate'])['evidence'] \
            .agg(lambda e: '; '.join(f"{k} {v}/{len(e)}" for k, v in e.str.split(' ').str[0].value_counts().items())) \
            .rename('stage1_calibration_profile').reset_index()
        s2 = s2.merge(prof, on=['model', 'condition', 'aggregate'], how='left')
    T['stage2_wordnet_derived'] = s2
    coef = [x[x['term'].isin(list(ORIENT)) & x['spec'].isin(['regression', 'factorial', 'primary'])]
            for x in (s1, s2) if len(x)]
    T['layerwise_frequency_ambiguity_coefficients'] = cat(coef)[
        ['model', 'condition', 'protocol', 'aggregate', 'analysis', 'spec', 'zero_policy', 'layer_column', 'depth',
         'readout_type', 'term', 'construct', 'construct_status', 'n', 'beta_std', 'boot_ci_std_low',
         'boot_ci_std_high', 'p_hc3', 'p_fdr_layers']] if coef else pd.DataFrame()
    T['depth_profile_tests'] = pd.DataFrame(L_['depth'])
    T['layerwise_prompt_interactions'] = add_global_fdr(cat(L_['inter']), ('analysis', 'term'))
    T['prompt_interaction_depth_tests'] = pd.DataFrame(L_['inter_depth'])
    T['representation_distance'] = pd.DataFrame(L_['dist'])
    T['context_balance'] = pd.DataFrame(L_['ctx_bal'])
    T['context_effects'] = pd.DataFrame(L_['ctx_eff'])
    T['homonymy_polysemy_distance'] = cat(L_['hp'])
    T['homonymy_polysemy_distance_profile'] = pd.DataFrame(L_['hp_prof'])
    T['context_dominance'] = pd.DataFrame(L_['dom'])
    T['list_protocol_order_diagnostics'] = cat(L_['list_order'])
    T['list_protocol_surrounding_items'] = cat(L_['list_surround'])
    T['probe_results'] = cat(L_['probe'])
    T['matched_robustness'] = pd.DataFrame(L_['matched'])
    if matches:
        T['matched_balance'] = pd.concat([m['balance'] for m in matches.values()], ignore_index=True)
        for name, m in matches.items():
            write_csv(m['pairs'], os.path.join(dirs['tables'], f'matched_items_{name}.csv'))
        res['matching'] = {k: v['info'] for k, v in matches.items()}
    T['tokenization_balance'] = pd.DataFrame(L_['tok_bal'])
    T['tokenization_bias_audit'] = pd.DataFrame(L_['tok_bias'])
    T['exclusions'] = cat(L_['excl'])
    T['construct_registry'] = pd.DataFrame(CONSTRUCTS)
    T['multiple_testing'] = multiple_testing_summary(T, alpha)
    res['not_run'].update(not_run_registry(cfg, T))
    T['analysis_registry'] = pd.DataFrame([{'analysis': k, 'status': v} for k, v in res['not_run'].items()])
    for name, df in T.items():
        if df is not None and len(df):
            write_csv(df, os.path.join(dirs['tables'], f'{name}.csv'))
    test = any(x.model_info.get('test_model_random_weights') for x in sets)
    res['latex_tables'] = make_publication_tables(cfg, T, samples, dirs, test)
    if cfg.run_figures:
        res['figures'] = make_figures(cfg, T, sets, samples, dirs)
    return res


def multiple_testing_summary(T: dict, alpha: float) -> pd.DataFrame:
    rows = []
    specs = {'stage1_published_appendix_A': ('p_hc3', 'p_fdr_layers', 'BH across layers'),
             'stage1_published_appendix_B': ('p_hc3', 'p_fdr_layers', 'BH across layers'),
             'stage1_published_appendix_C': ('p_hc3', 'p_fdr_layers', 'BH across layers'),
             'stage2_wordnet_derived': ('p_hc3', 'p_fdr_layers', 'BH across layers'),
             'layerwise_prompt_interactions': ('p_hc3', 'p_fdr_layers', 'BH across layers'),
             'probe_results': ('p_perm', 'p_fdr_layers', 'BH across layers'),
             'matched_robustness': ('p', 'p_fdr_layers', 'BH across layers'),
             'confirmatory_hypothesis_tests': ('p_two_sided', 'p_holm', 'Holm within prespecified family'),
             'sensitivity_hypothesis_tests': ('p_two_sided', 'p_holm', 'Holm within prespecified family')}
    for name, (pcol, acol, method) in specs.items():
        df = T.get(name)
        if df is None or not len(df):
            continue
        fam = [c for c in ('model', 'condition', 'aggregate', 'analysis', 'spec', 'zero_policy', 'term', 'family',
                           'target', 'pooling', 'contrast') if c in df]
        for kv, g in df.groupby(fam, dropna=False, sort=False):
            rows.append(dict(zip(fam, kv), table=name, method=method, alpha=alpha,
                             n_tests=int(g[pcol].notna().sum()),
                             n_significant_uncorrected=int((g[pcol] < alpha).sum()),
                             n_significant_adjusted=int((g[acol] < alpha).sum())))
    return pd.DataFrame(rows)


def not_run_registry(cfg, T: dict) -> dict:
    nr = {}
    hp = T.get('homonymy_polysemy_distance', pd.DataFrame())
    if not cfg.context_file:
        nr['context_vs_bare_word'] = ('NOT_RUN: no vetted meaning-/sense-disambiguating contexts supplied '
                                      '(--context_file); none are generated')
        nr['homonymy_vs_polysemy_primary'] = ('UNSUPPORTED: the supplied files contain no validated sense-pair or '
                                              'usage annotations; needs --context_file with human-validated '
                                              'meaning_id and sense_id (e.g. sentence pairs of the RAW-C type)')
    elif not cfg.context_info.get('has_meaning_ids'):
        nr['homonymy_vs_polysemy_primary'] = ('UNSUPPORTED: --context_file lacks meaning_id annotations for words with '
                                              'two or more contexts')
    elif not len(hp):
        nr['homonymy_vs_polysemy_primary'] = 'NOT_RUN: no word had two annotated contexts with usable alignment'
    if not (cfg.context_file and cfg.context_info.get('has_dominance_labels')):
        nr['dominant_vs_subordinate'] = ('NOT_RUN: no dominance resource (eDom; Twilley et al., 1994; Gilbert & '
                                         'Rodd, 2022) supplied; dominance labels are never fabricated')
    nr['homonymy_vs_polysemy_proxies'] = ('EXPLORATORY ONLY: Stage 2 specs with sim_top2 / min_sim / mean_sim '
                                          '(WordNet Wu–Palmer similarity; not human-validated)')
    nr['context_frequency_and_syntax_matching'] = ('NOT_AVAILABLE: no word-frequency lexicon for context words and no '
                                                   'parser; supply frame_id per context to mark shared syntactic frames')
    nr['nonword_foils_word_nonword_decision'] = ('NOT_RUN: the pool has no nonword foils and inventing pseudowords is '
                                                 'not allowed, so no lexical decision is simulated')
    nr['decision_site_answer_readout'] = ('NOT_RUN: the source response format echoes the item before WORD/NONWORD, '
                                          'and without nonwords a WORD/NONWORD margin is not a lexical decision')
    nr['behavioural_ease_generation'] = 'NOT_RUN: generated "ease" ratings are not implemented (relation to RT untested)'
    nr['tuned_lens'] = 'NOT_RUN: only the logit lens is implemented (no trained translators)'
    if not cfg.run_list_protocol:
        nr['list_protocol'] = 'NOT_RUN: secondary list protocol disabled (enable with --run_list_protocol)'
    if not cfg.run_prompt_manipulations:
        nr['task_instruction_manipulation'] = 'NOT_RUN: disabled (--no-run_prompt_manipulations)'
    return nr


# ════════════════════════════════════════════════════════════════════════════
# PUBLICATION FIGURES — vector PDF (embedded TrueType) + 300-dpi PNG at ACL
# column widths, captions as sidecar files; drawn only from analyses that ran
# ════════════════════════════════════════════════════════════════════════════
FIG_W = 16.0 / 2.54                                   # inches: ACL text width (16 cm); figures are saved at it
VIZ = {'text': '#0b0b0b', 'text2': '#52514e', 'grid': '#e6e5e0', 'zero': '#8f8e88', 'warn': '#d03b3b',
       'summary_output': '#8a6bbd',                     # fig. 4: output layer vs deep-region mean (black)
       # categorical slots 1–3 of the reference palette (validated all-pairs); colour follows the CONSTRUCT
       'frequency': '#2a78d6', 'meanings': '#eb6834', 'senses': '#1baf7a',
       'ramp': ['#86b6ef', '#2a78d6', '#104281']}          # ordinal blue ramp for frequency tertiles
TERM_STYLE = {'log_subtlex': ('Frequency (log SUBTLEX)', VIZ['frequency']),
              'gcide_etymologies': ('Meanings proxy (GCIDE etymologies)', VIZ['meanings']),
              'ambiguity': ('Meanings (published factor)', VIZ['meanings']),
              'wn_n_all': ('Senses proxy (WordNet senses)', VIZ['senses']),
              'senses': ('Senses (published factor)', VIZ['senses'])}
HYP_SHORT = {'S1A_meanings': 'A-M', 'S1A_senses': 'A-S', 'S1B_senses': 'B-S', 'S1B_meanings_null': 'B-M0',
             'S1B_separation': 'B-Sep', 'S1C_senses': 'C-S', 'S1C_meanings': 'C-M', 'S1C_separation': 'C-Sep',
             'S2_meanings': '2-M', 'S2_senses': '2-S', 'S2_frequency': '2-F', 'S2_separation': '2-Sep',
             'D_frequency_shallow': 'D-F', 'D_meanings_deep': 'D-M', 'D_senses_deep': 'D-S',
             'D_meanings_vs_frequency': 'D-M>F', 'D_senses_vs_frequency': 'D-S>F'}
HYP_COLOR = {h[0]: (VIZ['frequency'] if 'frequency' in h[0] and 'vs' not in h[0] else
                    VIZ['meanings'] if 'meanings' in h[0] else VIZ['senses'] if 'senses' in h[0] else VIZ['text'])
             for h in HYPOTHESES}
EVIDENCE_MARK = {'supported': 'S', 'supported (equivalent to zero within ±SESOI)': 'E', 'contradicted': 'C',
                 'contradicted (non-zero effect)': 'C', 'inconclusive': '·'}


def _plt():
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'], 'font.size': 8,
        'axes.titlesize': 8, 'axes.titleweight': 'bold', 'axes.labelsize': 8, 'xtick.labelsize': 7,
        'ytick.labelsize': 7, 'legend.fontsize': 7, 'legend.frameon': False, 'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'lines.linewidth': 1.2, 'axes.edgecolor': VIZ['text2'],
        'axes.labelcolor': VIZ['text'], 'xtick.color': VIZ['text2'], 'ytick.color': VIZ['text2'],
        'text.color': VIZ['text'], 'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False,
        'axes.spines.right': False})
    return plt


def _model_order(models) -> list:
    reg = [m.alias for m in MODEL_REGISTRY + TEST_MODEL_REGISTRY]
    return sorted(set(models), key=lambda m: (reg.index(m) if m in reg else len(reg), m))


def _depth_axes(ax, ylabel: str, zero: bool = True, xlabel: bool = True):
    ax.grid(axis='y', color=VIZ['grid'], linewidth=0.5)
    ax.set_axisbelow(True)
    if zero:
        ax.axhline(0, color=VIZ['zero'], linewidth=0.6)
    for x in (1 / 3, 2 / 3):
        ax.axvline(x, color=VIZ['zero'], linewidth=0.5, linestyle=':')
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 1 / 3, 2 / 3, 1])
    ax.set_xticklabels(['0', '⅓', '⅔', '1'])
    ax.set_xlabel('Normalised depth' if xlabel else '')
    ax.set_ylabel(ylabel)


def _save(fig, plt, dirs: dict, name: str, caption: str, test: bool) -> str:
    if test:
        fig.text(0.5, 0.5, 'TEST MODELS — NOT RESULTS', ha='center', va='center', rotation=18, fontsize=16,
                 color=VIZ['warn'], alpha=0.25)
    base = os.path.join(dirs['figures'], name)
    fig.savefig(base + '.pdf')                         # no tight bbox: keep the exact text width, so fonts are
    fig.savefig(base + '.png', dpi=300)                # printed at their nominal size with width=\textwidth
    atomic_write_text(base + '_caption.txt', caption + (' TEST MODELS WITH RANDOM WEIGHTS: these values are pipeline '
                                                        'checks, NOT RESULTS.' if test else ''))
    plt.close(fig)
    return base + '.pdf'


def _curves(df: pd.DataFrame, term: str, sign: float = 1.0) -> dict:
    out = {}
    for m, g in df[df['term'] == term].groupby('model'):
        g = g.sort_values('depth')
        lo, hi = sign * g['boot_ci_std_low'].values, sign * g['boot_ci_std_high'].values
        out[m] = (g['depth'].values, sign * g['beta_std'].values, np.minimum(lo, hi), np.maximum(lo, hi))
    return out


def _plot_curves(ax, curves: dict, color: str):
    """One model: line + 95% bootstrap band. Several models: thin per-model lines and
    the across-model median on a common depth grid (descriptive; models share items)."""
    if len(curves) == 1:
        x, y, lo, hi = next(iter(curves.values()))
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, y, color=color, linewidth=1.6)
        return
    grid = np.linspace(0.05, 1.0, 39)
    M = []
    for x, y, _, _ in curves.values():
        ax.plot(x, y, color=color, alpha=0.35, linewidth=0.7)
        M.append(np.interp(grid, x, y, left=np.nan))
    M = np.array(M)
    enough = np.isfinite(M).sum(0) >= 2
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        ax.plot(grid[enough], np.nanmedian(M[:, enough], 0), color=color, linewidth=2.0)


def _axis_power(df: pd.DataFrame) -> int:
    """Power of ten k such that CI limits / 10^k stay below 100 with short tick labels (k = 0 unless tiny)."""
    m = float(np.nanmax(np.abs(df[['ci95_low', 'ci95_high']].values))) if len(df) else 0.0
    return int(np.floor(np.log10(m))) if 0 < m < 0.1 else 0


def _forest(ax, rows: pd.DataFrame, models: list, color: str, offset: float = 0.0, power: int = 0):
    """Point estimate + 95% CI per model, in units of 10^power. Marker encodes the Holm-corrected
    evidence: filled circle = supported, triangle = contradicted, open circle = inconclusive."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    y = {m: i for i, m in enumerate(models)}
    u = 10.0 ** power
    for _, r in rows.iterrows():
        yy = y[r['model']] + offset
        ax.plot([r['ci95_low'] / u, r['ci95_high'] / u], [yy, yy], color=color, linewidth=1.0)
        ev = r['evidence']
        ax.plot(r['estimate'] / u, yy, marker='^' if ev.startswith('contradicted') else 'o', markersize=4.2,
                markerfacecolor='white' if ev == 'inconclusive' else color, markeredgecolor=color,
                markeredgewidth=0.9, linestyle='none')
    ax.axvline(0, color=VIZ['zero'], linewidth=0.6)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.2g}'.replace('-', '−')))
    ax.set_yticks(range(len(models)))
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.grid(axis='x', color=VIZ['grid'], linewidth=0.5)
    ax.set_axisbelow(True)


def make_figures(cfg, T: dict, sets: list, samples: Samples, dirs: dict) -> list:
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    plt = _plt()
    made = []
    pc, pa = cfg.primary_condition, cfg.primary_aggregate
    test = any(s.model_info.get('test_model_random_weights') for s in sets)
    common = (f'Surprisal: logit-lens estimates at intermediate blocks, model output at the final block; aggregate '
              f'"{pa}" (bits); condition "{pc}". Depth = (block index + 1) / number of blocks; dotted lines separate '
              f'the prespecified shallow, middle and deep regions.')
    sel = lambda df: df[(df['condition'] == pc) & (df['aggregate'] == pa)] if len(df) else df
    s1 = sel(pd.concat([T.get('stage1_published_appendix_B', pd.DataFrame()),
                        T.get('stage1_published_appendix_C', pd.DataFrame())], ignore_index=True))
    s1 = s1[s1['spec'] == 'factorial'] if len(s1) else s1
    if len(s1):
        fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 3.9), sharex=True, layout='constrained')
        for k, (ax, (an, term, title)) in enumerate(zip(axes.flat, (
                ('stage1_B', 'ambiguity', 'Exp. 2 (visual)\nmeanings: ambiguous − unambiguous'),
                ('stage1_B', 'senses', 'Exp. 2 (visual)\nsenses: many − few'),
                ('stage1_C', 'ambiguity', 'Exp. 3 (auditory; text input)\nmeanings: ambiguous − unambiguous'),
                ('stage1_C', 'senses', 'Exp. 3 (auditory; text input)\nsenses: many − few')))):
            _plot_curves(ax, _curves(s1[s1['analysis'] == an], term), TERM_STYLE[term][1])
            _depth_axes(ax, 'Effect (SD of surprisal)' if k % 2 == 0 else '', xlabel=k >= 2)
            ax.set_title(title, loc='left')
        n = s1['model'].nunique()
        made.append(_save(fig, plt, dirs, 'fig1_stage1_meanings_senses_by_depth',
                          'Stage 1 (published Rodd et al., 2002 items): effect of the meanings and senses factors on '
                          'target surprisal across depth, in SD of surprisal (positive = higher surprisal). ' +
                          ('Band = 95% stratified bootstrap CI over words. ' if n == 1 else
                           f'Thin lines = {n} models; thick line = across-model median (descriptive; models share '
                           'items). ') + 'Human pattern for sign comparison only: senses negative; meanings null '
                          '(Exp. 2) / positive (Exp. 3). ' + common, test))
    s2 = sel(T.get('stage2_wordnet_derived', pd.DataFrame()))
    if len(s2):
        s2 = s2[(s2['spec'] == 'primary') & (s2['zero_policy'] == cfg.gcide_zero_policy)]
    if len(s2):
        variants = [v for v in ('base', 'instruct') if (s2['variant'] == v).any()]
        terms = ['log_subtlex', 'gcide_etymologies', 'wn_n_all']
        fig, axes = plt.subplots(len(variants), 3, figsize=(FIG_W, 1.9 * len(variants) + 0.4),
                                 sharex=True, sharey='row', squeeze=False, layout='constrained')
        short = {'log_subtlex': 'Frequency\n(log SUBTLEX)', 'gcide_etymologies': 'Meanings proxy\n(GCIDE etymologies)',
                 'wn_n_all': 'Senses proxy\n(WordNet senses)'}
        for i, v in enumerate(variants):
            for j, t in enumerate(terms):
                ax = axes[i][j]
                _plot_curves(ax, _curves(s2[s2['variant'] == v], t, ORIENT[t]), TERM_STYLE[t][1])
                _depth_axes(ax, f'{v.capitalize()} models\noriented effect (SD)' if j == 0 else '',
                            xlabel=i == len(variants) - 1)
                ax.set_title(short[t] if i == 0 else '', loc='left')
        made.append(_save(fig, plt, dirs, 'fig2_stage2_frequency_meanings_senses_by_depth',
                          'Stage 2 (WordNet-derived pool; generalisation on computed proxies, not a replication): '
                          'oriented standardised coefficients (positive = the human-like direction: frequency and '
                          'senses facilitating, meanings inhibiting) from y_k ~ gcide_etymologies + wn_n_all + '
                          f'log_subtlex + length + concreteness (gcide zero policy: {cfg.gcide_zero_policy}). '
                          'Thin lines = individual models; thick = across-model median, or a 95% bootstrap band '
                          'when one model is shown. ' + common, test))
    conf = T.get('confirmatory_hypothesis_tests', pd.DataFrame())
    if len(conf):
        models = _model_order(conf['model'])
        handles = [Line2D([], [], marker='o', color=VIZ['text2'], markerfacecolor=VIZ['text2'], linestyle='none',
                          label='supported (Holm)'),
                   Line2D([], [], marker='o', color=VIZ['text2'], markerfacecolor='white', linestyle='none',
                          label='inconclusive'),
                   Line2D([], [], marker='^', color=VIZ['text2'], markerfacecolor=VIZ['text2'], linestyle='none',
                          label='contradicted (Holm)')]
        dh = [h[0] for h in HYPOTHESES if h[5] == 'depth_hypothesis']
        d = conf[conf['hypothesis'].isin(dh)]
        if len(d):
            fig, axes = plt.subplots(1, len(dh), figsize=(FIG_W, 0.2 * len(models) + 1.3),
                                     sharey=True, squeeze=False, layout='constrained')
            k = _axis_power(d)
            for ax, h in zip(axes[0], dh):
                _forest(ax, d[d['hypothesis'] == h], models, HYP_COLOR[h], power=k)
                ax.set_title({'D_frequency_shallow': 'Frequency\n(pred. < 0)',
                              'D_meanings_deep': 'Meanings\n(pred. > 0)',
                              'D_senses_deep': 'Senses\n(pred. > 0)',
                              'D_meanings_vs_frequency': 'Meanings −\nfrequency\n(pred. > 0)',
                              'D_senses_vs_frequency': 'Senses −\nfrequency\n(pred. > 0)'}[h], loc='left')
            axes[0][0].set_yticklabels(models)
            axes[0][len(dh) // 2].set_xlabel('Slope of the oriented standardised effect per unit of normalised depth'
                                             + (f' (×10$^{{{k}}}$)' if k else ''))
            fig.legend(handles=handles, loc='outside lower center', ncol=3)
            made.append(_save(fig, plt, dirs, 'fig3_depth_hypothesis_tests',
                              'Direct test of the depth hypothesis (Stage 2, prespecified, confirmatory): least-squares '
                              'slope of the oriented standardised effect on normalised depth, per model, with 95% '
                              'bootstrap CIs over words; slope differences test whether the ambiguity effects rise with '
                              'depth relative to frequency. Filled = supported, triangle = contradicted (Holm within '
                              'the depth family); open = inconclusive. ' + common, test))
        sh = ['S1B_separation', 'S1C_separation', 'S2_separation']
        s = conf[conf['hypothesis'].isin(sh) & conf['summary'].isin(['deep', 'output'])]
        if len(s):
            fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 0.24 * len(models) + 1.3), sharey=True,
                                     squeeze=False, layout='constrained')
            summ_col = {'deep': VIZ['text'], 'output': VIZ['summary_output']}
            k = _axis_power(s)
            for ax, h, title in zip(axes[0], sh, ('Stage 1, Exp. 2\n(visual)', 'Stage 1, Exp. 3\n(auditory; text input)',
                                                  'Stage 2\n(computed proxies)')):
                for summ, off in (('deep', -0.17), ('output', 0.17)):
                    _forest(ax, s[(s['hypothesis'] == h) & (s['summary'] == summ)], models, summ_col[summ], off, k)
                ax.set_title(title, loc='left')
            axes[0][0].set_yticklabels(models)
            axes[0][1].set_xlabel('β(meanings) − β(senses), standardised (predicted > 0)'
                                  + (f' (×10$^{{{k}}}$)' if k else ''))
            fig.legend(handles=handles + [Line2D([], [], color=summ_col['deep'], label='deep-region mean'),
                                          Line2D([], [], color=summ_col['output'], label='output layer')],
                       loc='outside lower center', ncol=5)
            made.append(_save(fig, plt, dirs, 'fig4_meanings_vs_senses_separation',
                              'Do meanings and senses separate? Difference between the standardised meanings and senses '
                              'effects (β_meanings − β_senses; positive under the human pattern of inhibitory meanings '
                              'and facilitatory senses), at the deep-region mean and the output layer, with 95% '
                              'bootstrap CIs; filled = supported, triangle = contradicted (Holm); open = '
                              'inconclusive. ' + common, test))
    prim = sorted([x for x in sets if x.condition == pc and _lens_available(x)], key=lambda x: x.alias)
    if prim and cfg.run_stage2:
        rows = samples.stage2[cfg.gcide_zero_policy]
        q = rows['log_subtlex'].quantile([1 / 3, 2 / 3]).values
        tert = pd.cut(rows['log_subtlex'], [-np.inf, q[0], q[1], np.inf], labels=['low', 'mid', 'high'])
        ncols = min(3, len(prim))
        nrows = int(math.ceil(len(prim) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_W, 1.6 * nrows + 0.5), squeeze=False,
                                 layout='constrained')
        for ax in axes.flat[len(prim):]:
            ax.set_visible(False)
        for k, (ax, ss) in enumerate(zip(axes.flat, _model_order([x.alias for x in prim]))):
            ss = next(x for x in prim if x.alias == ss)
            W = ss.word_matrix(pa)
            for i, lev in enumerate(['low', 'mid', 'high']):
                w = [x for x in rows.loc[tert == lev, 'word'] if x in W.index]
                if len(w) >= 3:
                    M = W.loc[w].values
                    mu, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(w))
                    ax.fill_between(ss.depth, mu - 1.96 * se, mu + 1.96 * se, color=VIZ['ramp'][i], alpha=0.2,
                                    linewidth=0)
                    ax.plot(ss.depth, mu, color=VIZ['ramp'][i], label=f'{lev} frequency')
            _depth_axes(ax, 'Surprisal (bits)' if k % ncols == 0 else '', zero=False)
            ax.set_title(ss.alias, loc='left')
        fig.legend(handles=[Line2D([], [], color=VIZ['ramp'][i], label=f'{lev} frequency')
                            for i, lev in enumerate(['low', 'mid', 'high'])], loc='outside lower center', ncol=3)
        made.append(_save(fig, plt, dirs, 'figS1_surprisal_by_frequency_tertile',
                          'Supplementary: mean target surprisal across depth for low, mid and high log-frequency '
                          'tertiles of the Stage 2 pool (bands = mean ± 1.96 SE over words). ' + common, test))
    if len(s2) and s2['model'].nunique() >= 2:
        models = _model_order(s2['model'])
        terms = ['log_subtlex', 'gcide_etymologies', 'wn_n_all']
        grid = np.linspace(0.05, 1.0, 20)
        mats = [np.array([np.interp(grid, *s2[(s2['model'] == m) & (s2['term'] == t)].sort_values('depth')[
            ['depth', 'beta_std']].values.T, left=np.nan) for m in models]) for t in terms]
        vmax = float(np.nanmax([np.nanmax(np.abs(M)) for M in mats])) or 1.0
        fig, axes = plt.subplots(1, 4, figsize=(FIG_W, 0.18 * len(models) + 1.3), squeeze=False,
                                 gridspec_kw={'width_ratios': [1, 1, 1, 0.05]}, layout='constrained')
        cmap = LinearSegmentedColormap.from_list('div', ['#1c5cab', '#f0efec', '#c23434'])
        for ax, t, M in zip(axes[0][:3], terms, mats):
            im = ax.imshow(M, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax, extent=(grid[0], 1.0, len(models), 0))
            ax.set_yticks(np.arange(len(models)) + 0.5)
            ax.set_yticklabels(models if t == terms[0] else [''] * len(models))
            ax.set_title({'log_subtlex': 'Frequency', 'gcide_etymologies': 'Meanings proxy',
                          'wn_n_all': 'Senses proxy'}[t], loc='left')
            ax.set_xlabel('Normalised depth')
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 1 / 3, 2 / 3, 1])
            ax.set_xticklabels(['0', '⅓', '⅔', '1'])
        fig.colorbar(im, cax=axes[0][3], label='Standardised β')
        made.append(_save(fig, plt, dirs, 'figS2_stage2_coefficients_heatmap',
                          'Supplementary: Stage 2 standardised coefficients for every model on a common normalised-depth '
                          'grid (linear interpolation; blank = depth not covered). ' + common, test))
    pid = T.get('prompt_interaction_depth_tests', pd.DataFrame())
    g0 = pid[(pid['statistic'] == 'Δ oriented effect at layer') & (pid['analysis'] == 'stage2')] if len(pid) else pid
    if len(g0):
        panels = [(m, c) for m in _model_order(g0['model']) for c in sorted(g0.loc[g0['model'] == m, 'condition'].unique())]
        ncols = min(3, len(panels))
        nrows = int(math.ceil(len(panels) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_W, 1.8 * nrows + 0.6), squeeze=False,
                                 layout='constrained')
        for ax in axes.flat[len(panels):]:
            ax.set_visible(False)
        for k, (ax, (m, c)) in enumerate(zip(axes.flat, panels)):
            g = g0[(g0['model'] == m) & (g0['condition'] == c)]
            for t in ('log_subtlex', 'gcide_etymologies', 'wn_n_all'):
                gt = g[g['term'] == t].sort_values('depth')
                ax.fill_between(gt['depth'], gt['boot_ci_low'], gt['boot_ci_high'], color=TERM_STYLE[t][1],
                                alpha=0.15, linewidth=0)
                ax.plot(gt['depth'], gt['estimate'], color=TERM_STYLE[t][1], label=TERM_STYLE[t][0])
            _depth_axes(ax, 'Δ oriented effect (SD)' if k % ncols == 0 else '')
            ax.set_title(f'{m}\n{c}', loc='left')
        fig.legend(handles=[Line2D([], [], color=TERM_STYLE[t][1], label=TERM_STYLE[t][0])
                            for t in ('log_subtlex', 'gcide_etymologies', 'wn_n_all')], loc='outside lower center',
                   ncol=3)
        made.append(_save(fig, plt, dirs, 'figS3_task_instruction_by_depth',
                          f'Supplementary (exploratory): change of the oriented Stage 2 effects when the lexical-decision '
                          f'frame ("{pc}") is replaced by another task instruction, paired by word (95% paired bootstrap '
                          'bands). ' + common, test))
    return made


# ════════════════════════════════════════════════════════════════════════════
# PUBLICATION TABLES — LaTeX (booktabs, table*: valid in ACL two-column layout)
# ════════════════════════════════════════════════════════════════════════════
_TEX_UNICODE = {'−': '$-$', 'β': r'$\beta$', '±': r'$\pm$', '×': r'$\times$', 'Δ': r'$\Delta$', '≥': r'$\geq$',
                '≤': r'$\leq$', '≈': r'$\approx$', '–': '--', '—': '---', '·': r'$\cdot$', '⅓': '1/3', '⅔': '2/3',
                '→': r'$\rightarrow$', '>': '$>$', '<': '$<$'}          # < > are ¡ ¿ in OT1 text mode


def _tex(s) -> str:
    s = str(s).replace('\\', '\x00')
    for a, b in (('&', r'\&'), ('%', r'\%'), ('_', r'\_'), ('#', r'\#'), ('{', r'\{'), ('}', r'\}'),
                 ('~', r'\textasciitilde{}'), ('^', r'\textasciicircum{}'), ('$', r'\$')):
        s = s.replace(a, b)
    for a, b in _TEX_UNICODE.items():
        s = s.replace(a, b)
    return s.replace('\x00', r'\textbackslash{}')


def _num(x, d=2) -> str:
    return '--' if not np.isfinite(x) else f'${x:.{d}f}$'


def _est_ci(r, d=2, stacked: bool = False) -> str:
    """estimate [95% CI] with the evidence mark; stacked = CI on a second line (narrow columns)."""
    mark = '$^{*}$' if str(r['evidence']).startswith('supported') else \
        ('$^{\\dagger}$' if str(r['evidence']).startswith('contradicted') else '')
    ci = f"[{_num(r['ci95_low'], d)}, {_num(r['ci95_high'], d)}]"
    if stacked:
        return (f"\\begin{{tabular}}[t]{{@{{}}c@{{}}}}{_num(r['estimate'], d)}{mark}"
                f"\\\\ {{\\scriptsize {ci}}}\\end{{tabular}}")
    return f"{_num(r['estimate'], d)} {ci}{mark}"


def latex_table(header: list, rows: list, caption: str, label: str, colspec: str, notes: str = '',
                groups: list = None, size: str = r'\small') -> str:
    """booktabs table* (full text width; longtable is not allowed in two-column mode). `groups` =
    [(label, n_columns), ...] adds a spanning header row with cmidrule rules above the column header."""
    head = ' & '.join(header) + r' \\'
    if groups:
        spans, rules, c = [], [], 1
        for g, k in groups:
            spans.append(f'\\multicolumn{{{k}}}{{c}}{{{g}}}' if g else '&' * (k - 1))
            if g:
                rules.append(f'\\cmidrule(lr){{{c}-{c + k - 1}}}')
            c += k
        head = ' & '.join(spans) + r' \\ ' + ''.join(rules) + '\n' + head
    out = [r'\begin{table*}[t]', r'\centering', size, f'\\begin{{tabular}}{{{colspec}}}', r'\toprule', head,
           r'\midrule'] + [' & '.join(r) + r' \\' for r in rows] + [r'\bottomrule', r'\end{tabular}']
    if notes:
        out.append(r'\par\smallskip\parbox{\linewidth}{\footnotesize ' + notes + '}')
    return '\n'.join(out + [f'\\caption{{{caption}}}', f'\\label{{{label}}}', r'\end{table*}'])  # ACL: caption below


def write_latex(path: str, tables: list):
    atomic_write_text(path, f"% Generated by {SCRIPT_NAME} v{SCRIPT_VERSION}; requires \\usepackage{{booktabs}}\n"
                      + '\n\n'.join(tables) + '\n')


def make_publication_tables(cfg, T: dict, samples: Samples, dirs: dict, test: bool) -> list:
    pdir = os.path.join(dirs['tables'], 'latex')
    os.makedirs(pdir, exist_ok=True)
    warn = r' \textbf{TEST MODELS WITH RANDOM WEIGHTS: NOT RESULTS.}' if test else ''
    made = []
    rows = []
    rr = lambda x: r'\raggedright ' + _tex(x)       # p{} cells; never the last column, so the row's \\ is intact
    for app in 'ABC':
        r = samples.stage1[samples.stage1['appendix'] == app]
        e, m = APPENDIX_DESIGN[app]
        if app == 'A':
            cells = ', '.join(f'{k} {v}' for k, v in r['ambiguity'].value_counts().items())
            design, n = 'regression: meanings + WordNet senses + controls', design_stage1_A(samples.stage1).n
        else:
            sizes = r.groupby(['ambiguity', 'senses']).size()
            cells = f'{sizes.iloc[0]} per cell' if sizes.nunique() == 1 else ', '.join(map(str, sizes.values))
            design, n = '2 × 2 meanings × senses', design_stage1_BC(samples.stage1, app).n
        rows.append(['Stage 1', f'Rodd2002-{app}', str(e), rr(m + (' (text input)' if m == 'auditory' else '')),
                     rr(design), rr(cells), str(n)])
    rows.append(['Stage 2', 'WordNet 3.0', '--', rr('--'), rr('regression on proxies'),
                 rr(f'{len(samples.stage2_base)} in range; zero policy {cfg.gcide_zero_policy}'),
                 str(design_stage2(samples.stage2[cfg.gcide_zero_policy], 'primary', cfg.gcide_zero_policy).n)])
    p = os.path.join(pdir, 'table1_design_and_samples.tex')
    write_latex(p, [latex_table(['Stage', 'Item set', 'Exp.', 'Modality', 'Design', 'Cells', '$n$'], rows,
                                'Item sets, designs and sample sizes; $n$ = complete cases analysed (recomputed from '
                                'the stimulus CSV).', 'tab:design', 'llcp{2.1cm}p{3.6cm}p{2.9cm}r',
                                notes='Stage 1 items are the published Rodd, Gaskell \\& Marslen-Wilson (2002) '
                                      'stimuli; Stage 2 items are WordNet-derived candidates never used in a human '
                                      'experiment.')])
    made.append(p)
    conf = T.get('confirmatory_hypothesis_tests', pd.DataFrame())
    if not len(conf):
        return made
    models = _model_order(conf['model'])
    hyps = [h[0] for h in HYPOTHESES if h[0] in set(conf['hypothesis'])]
    source = {x[0]: x[6] for x in HYPOTHESES}
    family = {x[0]: x[5] for x in HYPOTHESES}
    fam_label = {'stage1_calibration': ('Stage 1', 'Stage 1 calibration'),      # (column group, legend)
                 'stage2_generalisation': ('Stage 2', 'Stage 2 generalisation'),
                 'depth_hypothesis': ('Depth', 'Depth hypothesis')}
    fams = [f for f in fam_label if any(family[h] == f for h in hyps)]
    hyps = [h for f in fams for h in hyps if family[h] == f]
    col = {h: HYP_SHORT[h] if family[h] == 'stage1_calibration' else HYP_SHORT[h].split('-', 1)[1] for h in hyps}
    rows = []
    for m in models:
        g = conf[conf['model'] == m].set_index(['hypothesis', 'summary'])
        cells = []
        for h in hyps:
            key = (h, 'all blocks (slope per unit depth)') if h.startswith('D_') else (h, 'output')
            cells.append(_tex(EVIDENCE_MARK.get(g.loc[key, 'evidence'], '?')) if key in g.index else '--')
        rows.append([_tex(m)] + cells)
    p = os.path.join(pdir, 'table2_confirmatory_evidence_matrix.tex')
    legend = ' '.join(f"{fam_label[f][1]}: " + '; '.join(
        (col[h] if col[h] == HYP_SHORT[h] else f'{col[h]} ({HYP_SHORT[h]})') + f' = {source[h]}'
        for h in hyps if family[h] == f) + '.' for f in fams)
    write_latex(p, [latex_table(['Model'] + [_tex(col[h]) for h in hyps], rows,
                                'Prespecified confirmatory hypotheses per model (primary condition and aggregate): '
                                'S = supported, C = contradicted, E = equivalent to zero within the smallest effect of '
                                'interest, $\\cdot$ = inconclusive; Holm-adjusted within family. Effects are read at '
                                'the output layer, depth hypotheses as slopes over all blocks.' + warn, 'tab:evidence',
                                'l' + 'c' * len(hyps), notes=_tex(legend),
                                groups=[('', 1)] + [(fam_label[f][0], sum(family[h] == f for h in hyps)) for f in fams],
                                size=r'\footnotesize\setlength{\tabcolsep}{3pt}')])
    made.append(p)
    for name, hset, title, label in (
            ('table3_stage2_effects', ['S2_frequency', 'S2_meanings', 'S2_senses', 'S2_separation'],
             'Stage 2 (generalisation on computed proxies): standardised effects at the output layer', 'tab:stage2'),
            ('table4_depth_slopes', [h for h in hyps if h.startswith('D_')],
             'Depth hypothesis: slopes of oriented standardised effects on normalised depth', 'tab:depth')):
        hs = [h for h in hset if h in hyps]
        if not hs:
            continue
        rows = []
        for m in models:
            g = conf[conf['model'] == m]
            row = [_tex(m)]
            for h in hs:
                r = g[(g['hypothesis'] == h) & (g['summary'].isin(['output', 'all blocks (slope per unit depth)']))]
                row.append(_est_ci(r.iloc[0], 3 if name == 'table4_depth_slopes' else 2, stacked=True)
                           if len(r) else '--')
            rows.append(row)
        p = os.path.join(pdir, f'{name}.tex')
        write_latex(p, [latex_table(['Model'] + [_tex(HYP_SHORT[h]) for h in hs], rows,
                                    f'{title}, estimate [95\\% bootstrap CI]; $^{{*}}$ supported, $^{{\\dagger}}$ '
                                    f'contradicted (Holm within family).' + warn, label, 'l' + 'c' * len(hs),
                                    notes=_tex('; '.join(f'{HYP_SHORT[h]} = {source[h]}' for h in hs)))])
        made.append(p)
    tabs = []
    for m in models:
        g = conf[conf['model'] == m]
        g = g.iloc[np.lexsort((g['summary'].map({k: i for i, k in enumerate(('shallow', 'middle', 'deep', 'output'))})
                               .fillna(4).values, g['hypothesis'].map({h: i for i, h in enumerate(hyps)}).values))]
        rows = [[_tex(HYP_SHORT[r['hypothesis']]), _tex(r['summary'].replace('all blocks (slope per unit depth)',
                                                                               'slope over all blocks')),
                 _est_ci(r, 3), _num(r['p_holm'], 3), _tex(EVIDENCE_MARK.get(r['evidence'], '?'))]
                for _, r in g.iterrows()]
        tabs.append(latex_table(['Hyp.', 'Summary', 'Estimate [95\\% CI]', '$p_\\mathrm{Holm}$', 'Evidence'], rows,
                                f'All prespecified confirmatory tests for {_tex(m)} (primary condition and aggregate); '
                                'codes and evidence marks as in the evidence-matrix table.' + warn,
                                f'tab:confirmatory-{m}', 'llccc', size=r'\footnotesize'))
    p = os.path.join(pdir, 'tableS1_confirmatory_tests_by_model.tex')
    write_latex(p, tabs)
    made.append(p)
    return made


# ════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT, REPORT
# ════════════════════════════════════════════════════════════════════════════
def environment_snapshot() -> dict:
    from importlib import metadata as md
    env = {'python': sys.version, 'platform': platform.platform(), 'script': SCRIPT_NAME,
           'script_version': SCRIPT_VERSION, 'packages': {}, 'hf_hub_offline': os.environ.get('HF_HUB_OFFLINE'),
           'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'cuda_available': False, 'gpus': []}
    for p in ('numpy', 'pandas', 'scipy', 'torch', 'transformers', 'tokenizers', 'accelerate', 'bitsandbytes',
              'huggingface_hub', 'safetensors', 'matplotlib', 'statsmodels', 'scikit-learn'):
        try:
            env['packages'][p] = md.version(p)
        except md.PackageNotFoundError:
            env['packages'][p] = None
    if torch_available():
        torch = _torch()
        env.update({'cuda_available': torch.cuda.is_available(), 'cuda_version': torch.version.cuda,
                    'gpus': [{'index': i, 'name': torch.cuda.get_device_name(i),
                              'capability': list(torch.cuda.get_device_capability(i)),
                              'memory_gb': round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)}
                             for i in range(torch.cuda.device_count())]})
    return env


def _md_table(df: pd.DataFrame, cols=None, max_rows=60) -> str:
    if df is None or not len(df):
        return '_(none)_\n'
    d = (df[[c for c in cols if c in df]] if cols else df).head(max_rows)

    def f(v):
        if isinstance(v, (float, np.floating)):
            return '' if not np.isfinite(v) else f'{v:.3g}'
        return str(v).replace('|', '\\|').replace('\n', ' ')
    lines = ['| ' + ' | '.join(map(str, d.columns)) + ' |', '|' + '---|' * len(d.columns)]
    lines += ['| ' + ' | '.join(f(v) for v in r) + ' |' for r in d.itertuples(index=False)]
    more = f'\n_({len(df) - max_rows} more rows in the CSV)_\n' if len(df) > max_rows else ''
    return '\n'.join(lines) + '\n' + more


def _md_spacing(text: str) -> str:
    """Blank lines around headings, block quotes, lists and tables."""
    out, prev = [], ''
    for line in text.split('\n'):
        starts_block = line.startswith(('#', '> ')) or (line.startswith('|') and not prev.startswith('|')) or \
            (line.startswith('- ') and not prev.startswith(('- ', '  ')))
        if (starts_block and prev.strip()) or (prev.startswith(('#', '> ')) and line.strip()) or \
                (prev.startswith('|') and line.strip() and not line.startswith('|')):
            out.append('')
        out.append(line)
        prev = line
    return '\n'.join(out)


def _evidence_matrix(h: pd.DataFrame) -> pd.DataFrame:
    """Hypothesis × summary grid of 'estimate [evidence]' for one model."""
    h = h.assign(cell=[f"{e:+.3f} [{v.split(' (')[0]}]" if np.isfinite(e) else 'n/a'
                       for e, v in zip(h['estimate'], h['evidence'])])
    g = h.pivot_table(index=['hypothesis', 'expected'], columns='summary', values='cell', aggfunc='first')
    order = [s for s in list(SUMMARIES) + ['all blocks (slope per unit depth)'] if s in g.columns]
    return g[order].reset_index()


def write_report(cfg, dirs, audit, env, conds, model_status, model_errors, res, plan, mode: str):
    L = []
    test = any(str(a).endswith('-TEST') for a in model_status)
    L.append("# Meanings, senses, homonymy/polysemy and frequency across LLM layers — run report\n")
    L.append(f"*{SCRIPT_NAME} v{SCRIPT_VERSION} · {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · "
             f"mode: **{mode}***\n")
    if test:
        L.append('> **TEST MODELS WITH RANDOM WEIGHTS WERE USED. Every number from a `*-TEST` model is a pipeline '
                 'check, not a result.**\n')
    if audit['stage_counts']['subset_debug_run']:
        L.append(f'> **SUBSET RUN (`--max_items {cfg.max_items}`): debug only; not the study sample.**\n')
    if mode == 'audit_only':
        L.append('> **AUDIT ONLY: no model was loaded and no analysis was run.**\n')
    L.append("## 1. Question and constructs\n")
    L.append("Do LLMs distinguish multiple relatively unrelated **meanings** from multiple related **senses**, can "
             "**homonymy**-like ambiguity be distinguished from **polysemy**-like structure, and how do these effects "
             "and **word frequency** vary across depth?\n")
    L.append(_md_table(pd.DataFrame(CONSTRUCTS), ['construct', 'stage', 'variable', 'status', 'limitation']))
    hpst = audit['homonymy_polysemy']['status']
    L.append(f"\n**Homonymy vs polysemy testability:** {hpst}\n")
    L.append("## 2. What was directly measured\n")
    L.append(f"- {LENS_METHOD}.\n"
             "- Primary aggregate (declared in advance): `" + cfg.primary_aggregate + "`. "
             + ' '.join(f"**{k}**: {v}." for k, v in AGGREGATE_DEFINITIONS.items()) + "\n"
             "- In the isolated-item protocol all items share one prefix, so the FIRST subtoken is scored from a "
             "hidden state that is identical for every item: its layerwise profile reflects where the model's "
             "*expectation* for the upcoming word forms, not processing of the word. Later subtokens and the "
             "representations used by probes and distances are computed AT the word's own positions.\n"
             "- Isolated-word surprisal is a baseline measure of word-form predictability under a task frame; it does "
             "not by itself demonstrate ambiguity resolution.\n")
    L.append("## 3. What was inferred, and under which assumptions\n")
    L.append(f"- {plan['stand_in_mapping']}\n"
             "- Stage 2 variables are computed proxies (see §1); Stage 2 is a generalisation analysis on a derived "
             "pool, never a replication of Rodd et al. (2002).\n"
             "- Intermediate-layer values are logit-lens estimates; only the final layer is the model's output.\n"
             f"- The end-of-word correction assumes canonical tokenisation and the boundary set: {BOUNDARY_DEFINITION}.\n")
    L.append("## 4. Data, exclusions and confounds\n")
    f = audit['file']
    L.append(f"File `{f['path']}` — sha256 `{f['sha256']}` — {f['n_rows']} rows × {f['n_columns']} columns.\n")
    L.append(_md_table(pd.DataFrame(audit['stage_counts']['funnel'])))
    gz = audit['gcide_etymologies_zero']
    L.append(f"\n`gcide_etymologies == 0`: {gz['stage2_pool_rows']} pool rows; primary policy `{gz['primary_policy']}`. "
             f"{gz['policy_status']} n by policy: {gz['n_stage2_by_policy']}. Every excluded row and its reason is in "
             f"`tables/exclusions.csv`.\n")
    L.append("\nScreening: " + '; '.join(f"{k}: {v['status']} — {v.get('reason', v.get('rule', ''))}"
                                        for k, v in audit['screening'].items()) + "\n")
    L.append(f"\nRepeated words: {audit['repeated_words']['dependence']}\n")
    bal = pd.DataFrame(audit['stage1_balance'])
    if len(bal):
        L.append("\nLargest published-design covariate imbalances (|SMD|):\n")
        L.append(_md_table(bal.assign(abs_smd=bal['smd'].abs()).sort_values('abs_smd', ascending=False),
                           ['appendix', 'contrast', 'covariate', 'mean_level_1', 'mean_level_2', 'smd'], max_rows=8))
    cc = pd.DataFrame(audit['source_cross_checks']['comparisons'])
    L.append("\nSource numbers that did not reproduce:\n")
    L.append(_md_table(cc[~cc['match']]) if len(cc) and (~cc['match']).any() else '_none_\n')
    for n in audit['source_cross_checks']['notes']:
        L.append(f"- {n}\n")
    v = audit['stage2_collinearity'].get(cfg.gcide_zero_policy, {})
    if v:
        L.append(f"\nStage 2 VIF (n = {v['n']}): " + ', '.join(f"{k} {x:.2f}" for k, x in v['vif'].items()) + "\n")
    L.append("\n## 5. Models, tokenisation and alignment\n")
    L.append(f"torch {env['packages'].get('torch')}, transformers {env['packages'].get('transformers')}, tokenizers "
             f"{env['packages'].get('tokenizers')}; GPUs: {[g['name'] for g in env.get('gpus', [])] or 'none'}; dtype "
             f"{cfg.dtype}; device {cfg.resolved_device}; quantization {cfg.quantization}.\n")
    rows = []
    for alias, st in model_status.items():
        p = os.path.join(dirs['models'], f"{safe_name(alias)}.json")
        info = read_json(p) if os.path.exists(p) else {}
        lv = info.get('lens_verification') or {}
        rows.append({'model': alias, 'revision': info.get('resolved_revision'), 'variant': info.get('variant'),
                     'layers': info.get('num_hidden_layers'), 'status': st.get('status'), 'logit lens': lv.get('status'),
                     'last-layer |Δ log p| (nats)': lv.get('max_abs_diff_target_logprob_nats_last_layer'),
                     'padding': (info.get('padding_check') or {}).get('status'),
                     'max alignment failure rate': max([c.get('alignment_failure_rate', 0)
                                                        for c in st.get('conditions', {}).values()] or [np.nan])})
    L.append(_md_table(pd.DataFrame(rows)))
    for a, e in model_errors.items():
        L.append(f"- **{a} failed**: {e['error']} (traceback: `logs/model_load_errors.json`)\n")
    L.append("\nBASE models: complete text tokenised in one call; INSTRUCT models: the tokenizer's own chat template "
             "(single user message, generation prompt, no system message added here; `date_string` fixed to "
             f"`{cfg.chat_date_string}`), re-tokenisation must equal `apply_chat_template(tokenize=True)`. Target spans "
             "come from validated offsets (decode fallback if offsets are inconsistent); merges across the word "
             "boundary are alignment failures (excluded, listed). Per-item token counts and diagnostics are saved with "
             "every score (`item_scores/`).\n")
    for c in conds.values():
        L.append(f"- `{c.name}` — {c.protocol}, {c.status}; wording: {c.wording_source}\n")
    L.append("\n## 6. Analysis plan\n")
    L.append(f"{plan['status']} Summaries: {plan['summaries']}. Uncertainty: {plan['uncertainty']}. Multiplicity: "
             f"{plan['multiplicity']}. Depth test: {plan['depth_test']}.\n")
    if mode in ('audit_only', 'smoke_test', 'aborted'):
        L.append("\n## 7. Analyses\n")
        L.append(f"_Not run in {mode} mode._\n")
        atomic_write_text(os.path.join(dirs['root'], 'REPORT.md'), _md_spacing(''.join(L)))
        return
    T = res.get('tables', {})
    conf = T.get('confirmatory_hypothesis_tests', pd.DataFrame())
    L.append("\n## 7. Confirmatory results (primary condition and aggregate)\n")
    L.append("Each prespecified hypothesis is classified **supported / contradicted / inconclusive** (Holm within "
             "family); cells show the estimate and its classification. There is no single pass/fail gate.\n")
    for m, g in (conf.groupby('model') if len(conf) else []):
        L.append(f"\n### {m}\n")
        L.append(_md_table(_evidence_matrix(g), max_rows=40))
    if len(conf):
        L.append("\n**Sensitivity — what these item sets can detect.** `mde80` = smallest true standardised effect "
                 "that the unadjusted two-sided test detects with 80% power at the observed bootstrap SE (the "
                 "Holm-adjusted tests need more); median over models, output layer or depth slope.\n")
        key = conf[conf['summary'].isin(['output', 'all blocks (slope per unit depth)'])]
        L.append(_md_table(key.groupby('hypothesis', sort=False).agg(
            n=('n', 'first'), median_boot_se=('boot_se', 'median'), median_mde80=('mde80', 'median')).reset_index()))
        eq = conf[conf['kind'] == 'equivalence']
        if len(eq) and not eq['equivalence_attainable'].astype(bool).any():
            L.append(f"\n**Equivalence is not attainable** for {', '.join(sorted(set(eq['hypothesis'])))} at "
                     f"n = {int(eq['n'].max())} with SESOI = ±{cfg.sesoi_std} SD in any model or summary: "
                     "z(1−α)·SE ≥ SESOI even for an estimate of exactly 0, so this hypothesis can only be "
                     "'inconclusive' or 'contradicted'. This is a property of the published item set (its size), "
                     "not of the models; a null meanings effect cannot be confirmed with it.\n")
    pat = T.get('meanings_vs_senses_pattern', pd.DataFrame())
    L.append("\n## 8. Do meanings and senses separate, or is there one predictability effect?\n")
    L.append(_md_table(pat[pat['role'] == 'confirmatory'] if len(pat) else pat,
                       ['model', 'analysis', 'summary', 'separation_estimate', 'separation_ci95_low',
                        'separation_ci95_high', 'pattern'], max_rows=80))
    L.append("\n## 9. Frequency vs ambiguity across depth (direct test)\n")
    L.append(_md_table(conf[conf['family'] == 'depth_hypothesis'] if len(conf) else conf,
                       ['model', 'hypothesis', 'estimate', 'ci95_low', 'ci95_high', 'p_holm', 'evidence'], max_rows=80))
    L.append("Secondary depth descriptors (region contrasts, |β|, peak depth with CIs): `tables/depth_profile_tests.csv`. "
             "Models are compared on the common normalised depth; they share items and are not independent samples, so "
             "no pooled cross-model test is made.\n")
    L.append("\n## 10. Homonymy vs polysemy\n")
    hp = T.get('homonymy_polysemy_distance', pd.DataFrame())
    L.append(f"{hpst}\n")
    if len(hp):
        L.append(_md_table(hp[hp['evidence'] != 'descriptive'], ['model', 'role', 'contrast', 'summary', 'estimate',
                                                                 'ci95_low', 'ci95_high', 'p_holm', 'evidence']))
    L.append("\n## 11. Exploratory and sensitivity analyses (reported separately)\n")
    sens = T.get('sensitivity_hypothesis_tests', pd.DataFrame())
    if len(sens):
        L.append(_md_table(sens.groupby(['model', 'condition', 'aggregate', 'role'])['evidence']
                           .agg(lambda e: '; '.join(f'{k}: {v}' for k, v in e.str.split(' (', regex=False).str[0]
                                                    .value_counts().items())).reset_index(), max_rows=60))
    for name, desc in (('prompt_interaction_depth_tests', 'task-instruction × layer'),
                       ('context_effects', 'context vs bare word (same frame)'),
                       ('list_protocol_surrounding_items', 'list protocol: position and previous-item effects'),
                       ('probe_results', 'linear probes (decodability is not use)'),
                       ('matched_robustness', 'matched-pair robustness'),
                       ('tokenization_bias_audit', 'tokenisation bias by aggregate')):
        df = T.get(name)
        L.append(f"- {desc}: {'`tables/' + name + '.csv` (' + str(len(df)) + ' rows)' if df is not None and len(df) else 'not run'}\n")
    L.append("\n## 12. Run / skipped / unsupported\n")
    for k, v in res.get('not_run', {}).items():
        L.append(f"- **{k}**: {v}\n")
    for s in res.get('skipped_caches', []):
        L.append(f"- skipped cache: {s}\n")
    L.append("\n## 13. Limitations\n")
    for lim in (
            'Why these are not reaction times: surprisal and probe scores are properties of a model\'s probability '
            'distribution or hidden states; the mapping to human processing cost is a theoretical sign comparison.',
            'Appendix C items were spoken in Experiment 3; here they are text, so modality-specific effects (e.g. '
            'uniqueness point, duration) cannot arise, and C is a test of the item set, not of auditory processing.',
            'No nonword foils: no word/nonword decision is simulated; the lexical-decision prompt is only a frame.',
            'The logit lens can be unreliable in early layers of some families (Belrose et al., 2023); early-layer '
            'profiles may reflect readout mismatch rather than absent information.',
            'Summed surprisal grows with subtoken count, which correlates with frequency; see the n_subtokens '
            'robustness specs and tokenization_bias_audit. The end-of-word correction does not remove this.',
            'A senses "facilitation" in isolated-word surprisal can also arise from contextual diversity (words with '
            'more senses occur in more contexts), which this design cannot separate.',
            'Stage 1 cells were matched on CELEX frequency; SUBTLEX frequency is used here (balance in §4).',
            'The relatedness instruction and the context frame are authored here, not source wording.',
            'Inflected forms and emotional extremes were not screened unless a resource was supplied.'):
        L.append(f"- {lim}\n")
    L.append("\n## 14. Files\n")
    L.append("Figures are drawn at the ACL text width (16 cm) with 7–8 pt embedded TrueType text: include them with "
             "`\\includegraphics[width=\\textwidth]{…}` inside `figure*`, unscaled. Tables are `table*` floats "
             "(`\\usepackage{booktabs}`), checked to compile within the 16 cm text width; `\\input{…}` them.\n\n")
    for fpath in res.get('figures', []):
        L.append(f"- figure (vector PDF; 300-dpi PNG and `_caption.txt` alongside): "
                 f"`{os.path.relpath(fpath, dirs['root'])}`\n")
    for fpath in res.get('latex_tables', []):
        L.append(f"- LaTeX table (booktabs): `{os.path.relpath(fpath, dirs['root'])}`\n")
    L.append("- tables: `tables/` (confirmatory_hypothesis_tests, meanings_vs_senses_pattern, stage1_*, stage2_*, "
             "exclusions, construct_registry, analysis_registry, multiple_testing, …)\n")
    atomic_write_text(os.path.join(dirs['root'], 'REPORT.md'), _md_spacing(''.join(L)))


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════
SCIENTIFIC_KEYS = ('data_sha256', 'seed', 'dtype', 'gcide_zero_policy', 'primary_aggregate', 'primary_condition',
                   'max_items', 'n_bootstrap', 'n_permutations', 'n_folds', 'probe_lambda_scale', 'alpha', 'sesoi_std',
                   'list_size', 'list_repeats', 'context_sha256', 'task_prompts_sha256', 'chat_date_string',
                   'quantization', 'attn_implementation', 'match_caliper_sd', 'screening_sha256',
                   'save_representations', 'script_version')


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='Meanings vs senses, homonymy/polysemy and frequency across LLM layers.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    B = argparse.BooleanOptionalAction
    g = ap.add_argument_group('data and output')
    g.add_argument('--data_csv', default=DEFAULT_DATA_CSV)
    g.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR)
    g.add_argument('--resume', action='store_true', help='reuse caches whose manifests match; add models')
    g.add_argument('--overwrite', action='store_true', help='recompute and replace existing outputs')
    g.add_argument('--max_items', type=int, default=None, help='SUBSET of unique words (debug/smoke only)')
    g.add_argument('--gcide_zero_policy', choices=GCIDE_ZERO_POLICIES, default='drop',
                   help='primary handling of gcide_etymologies == 0 (UNRESOLVED in the source; all policies reported)')
    g.add_argument('--screen_inflected_file', default=None, help='CSV word,is_inflected (optional resource)')
    g.add_argument('--screen_valence_file', default=None, help='CSV word,valence (optional resource)')
    g.add_argument('--valence_extreme_sd', type=float, default=2.0)
    g = ap.add_argument_group('models')
    g.add_argument('--models', nargs='+', default=['all'],
                   help="aliases, 'all', 'base', 'instruct', 'tests', or repo IDs (org/name[@rev][::base|::instruct])")
    g.add_argument('--list_models', action='store_true')
    g.add_argument('--dtype', choices=['bf16', 'fp16', 'fp32'], default=None,
                   help='default: bf16 on CUDA with bf16 support, else fp32')
    g.add_argument('--device', default='auto', help="'auto', 'cpu', 'cuda', 'cuda:N'")
    g.add_argument('--device_map', choices=['auto', 'none'], default='auto',
                   help="'auto' shards across visible GPUs (needs accelerate); 'none' uses --device")
    g.add_argument('--attn_implementation', default=None, help="e.g. 'sdpa', 'eager' (default: model default)")
    g.add_argument('--quantization', choices=['none', '8bit', '4bit'], default='none',
                   help='NOT recommended: changes layerwise states and logits; labelled in every output')
    g.add_argument('--trust_remote_code', action='store_true')
    g.add_argument('--batch_size', type=int, default=16)
    g.add_argument('--lens_chunk', type=int, default=256, help='positions per logit-lens vocabulary projection')
    g.add_argument('--continue_on_model_error', action='store_true')
    g.add_argument('--chat_date_string', default='26 Jul 2024',
                   help='fixed date for chat templates that insert the current date')
    g.add_argument('--padding_tolerance_bits', type=float, default=0.25)
    g.add_argument('--max_alignment_failure_rate', type=float, default=0.01)
    g.add_argument('--save_representations', choices=['both', 'last', 'mean', 'none'], default='both')
    g = ap.add_argument_group('design')
    g.add_argument('--primary_measure', choices=['layerwise_surprisal'], default='layerwise_surprisal')
    g.add_argument('--primary_aggregate', choices=list(AGGREGATES), default='sum')
    g.add_argument('--primary_condition', default='ld_isolated')
    g.add_argument('--task_prompts_json', default=None,
                   help='JSON {name: {template, wording_source}} adding isolated task frames ({ITEM_LIST} at the end)')
    g.add_argument('--context_file', default=None,
                   help='CSV word,context_id,text({TARGET}) [,meaning_id,sense_id,dominance,frame_id,source,'
                        'validation_status]')
    g.add_argument('--run_stage1', action=B, default=True)
    g.add_argument('--run_stage2', action=B, default=True)
    g.add_argument('--run_prompt_manipulations', action=B, default=True)
    g.add_argument('--run_list_protocol', action=B, default=False)
    g.add_argument('--list_size', type=int, default=20)
    g.add_argument('--list_repeats', type=int, default=3)
    g.add_argument('--run_probes', action=B, default=True)
    g.add_argument('--run_distance', action=B, default=True)
    g.add_argument('--run_matched', action=B, default=True)
    g.add_argument('--run_figures', action=B, default=True)
    g = ap.add_argument_group('statistics')
    g.add_argument('--seed', type=int, default=42)
    g.add_argument('--alpha', type=float, default=0.05)
    g.add_argument('--sesoi_std', type=float, default=0.2,
                   help='smallest effect of interest (SD of surprisal) for the equivalence test of the Exp. 2 null')
    g.add_argument('--n_bootstrap', type=int, default=2000)
    g.add_argument('--n_permutations', type=int, default=1000)
    g.add_argument('--n_folds', type=int, default=5)
    g.add_argument('--probe_lambda_scale', type=float, default=1.0)
    g.add_argument('--match_caliper_sd', type=float, default=0.2)
    g.add_argument('--min_matched_pairs', type=int, default=20)
    g = ap.add_argument_group('modes')
    g.add_argument('--audit_only', action='store_true', help='validate data and write the audit; no models')
    g.add_argument('--smoke_test', action='store_true', help='small end-to-end check with PASS/FAIL report')
    g.add_argument('--self_test', action='store_true', help='unit and offline integration tests')
    g.add_argument('--analysis_only', action='store_true', help='analyse existing caches; load no model')
    return ap


def finalize_config(args):
    cfg = args
    if cfg.smoke_test:
        if cfg.output_dir == DEFAULT_OUTPUT_DIR:
            cfg.output_dir = DEFAULT_OUTPUT_DIR + '_smoke'
        cfg.max_items = cfg.max_items or 16
        cfg.overwrite = True
    cuda = bf16_ok = False
    if torch_available():
        torch = _torch()
        cuda = torch.cuda.is_available()
        bf16_ok = cuda and torch.cuda.is_bf16_supported()
    cfg.resolved_device = ('cuda' if cuda else 'cpu') if cfg.device == 'auto' else cfg.device
    if cfg.dtype is None:
        cfg.dtype = 'bf16' if (cfg.resolved_device.startswith('cuda') and bf16_ok) else 'fp32'
    if cfg.resolved_device.startswith('cuda') and not cuda:
        raise CriticalError('--device cuda requested but CUDA is not available')
    if cfg.dtype == 'bf16' and cfg.resolved_device.startswith('cuda') and not bf16_ok:
        raise CriticalError('bf16 requested but this GPU does not support bf16; use --dtype fp16 or fp32')
    if cfg.dtype == 'fp16' and cfg.resolved_device == 'cpu':
        raise CriticalError('fp16 on CPU is not supported reliably; use --dtype fp32 (or bf16)')
    if not cfg.resolved_device.startswith('cuda'):
        cfg.device_map = 'none'
    elif cfg.device_map == 'auto':
        try:
            __import__('accelerate')
        except ImportError:
            logger.warning("accelerate not installed: --device_map auto unavailable, using a single device")
            cfg.device_map = 'none'
    cfg.context_sha256 = sha256_file(cfg.context_file) if cfg.context_file else None
    cfg.context_info = {}
    cfg.task_prompts_sha256 = sha256_file(cfg.task_prompts_json) if cfg.task_prompts_json else None
    cfg.screening_sha256 = sha256_json([sha256_file(p) if p else None for p in
                                        (cfg.screen_inflected_file, cfg.screen_valence_file)] + [cfg.valence_extreme_sd])
    cfg.script_version = SCRIPT_VERSION
    return cfg


def setup_logging(log_dir: str | None):
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_dir:
        fh = logging.FileHandler(os.path.join(log_dir, 'run.log'))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False


def config_record(cfg, plan, conds) -> dict:
    d = {k: (v if isinstance(v, (str, int, float, bool, type(None), list, dict)) else str(v))
         for k, v in vars(cfg).items()}
    d.update({'analysis_plan': plan, 'analysis_plan_sha256': sha256_json(plan),
              'conditions': {n: dict(asdict(c), template_sha256=c.template_sha256) for n, c in conds.items()},
              'scientific_keys': {k: d.get(k) for k in SCIENTIFIC_KEYS}})
    return d


def smoke_report(model_status: dict, dirs: dict) -> tuple[bool, list]:
    rows, ok_all = [], True
    for alias, st in model_status.items():
        info_p = os.path.join(dirs['models'], f"{safe_name(alias)}.json")
        info = read_json(info_p) if os.path.exists(info_p) else {}
        lv = info.get('lens_verification') or {}
        finite = False
        for cname in st.get('conditions', {}):
            ce = CacheEntry(dirs['root'], alias, cname)
            if ce.exists():
                s = aggregate_surprisal(ce.load()[1])['sum']
                finite |= bool(np.isfinite(s[:, -1]).any() and np.isfinite(s[:, 1]).any())
        checks = [('tokenizer rendering + prompt examples written',
                   os.path.exists(os.path.join(dirs['prompts'], f"{safe_name(alias)}_tokenizer_diagnostics.json"))),
                  ('model loading', st.get('status') in ('OK', 'PARTIAL') and bool(info)),
                  ('target span alignment', all(c.get('alignment_failure_rate', 1) == 0
                                                for c in st.get('conditions', {}).values())),
                  ('hidden-state count / width', lv.get('n_hidden_state_tensors') == lv.get('expected_hidden_state_tensors')
                   and lv.get('hidden_size_observed') == info.get('hidden_size')),
                  ('logit lens reproduces model logits at last layer', lv.get('status') == 'OK'),
                  ('padding invariance', (info.get('padding_check') or {}).get('status') == 'PASS'),
                  ('one small layerwise score is finite', finite)]
        for name, ok in checks:
            rows.append((alias, name, 'PASS' if ok else 'FAIL'))
            ok_all &= bool(ok)
    return ok_all, rows


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_models:
        for m in MODEL_REGISTRY + TEST_MODEL_REGISTRY:
            print(f"{m.alias:24s} {m.repo_id:42s} {m.variant:9s} base_of={m.base_of}"
                  + ('  [offline random-weight TEST model]' if m.test_model else ''))
        return 0
    if args.self_test:
        setup_logging(None)
        return 0 if run_self_tests() else 1
    try:
        cfg = finalize_config(args)
    except CriticalError as e:
        print(f"CRITICAL: {e}", file=sys.stderr)
        return 1
    exists = os.path.exists(os.path.join(cfg.output_dir, 'config.json'))
    dirs = output_dirs(cfg.output_dir)
    setup_logging(dirs['logs'])
    logger.info(f"{SCRIPT_NAME} v{SCRIPT_VERSION} → {os.path.abspath(cfg.output_dir)}")
    seed_everything(cfg.seed)
    try:
        df, vrep = load_and_validate(cfg.data_csv)
        context_df = None
        if cfg.context_file:
            context_df, cfg.context_info = load_context_file(cfg.context_file)
    except CriticalError as e:
        logger.error(f"CRITICAL input validation failure: {e}")
        return 2
    cfg.data_sha256 = vrep['sha256']
    if exists and not cfg.overwrite:
        old = read_json(os.path.join(cfg.output_dir, 'config.json')).get('scientific_keys', {})
        diff = sorted(k for k in SCIENTIFIC_KEYS if _clean_json(old.get(k)) != _clean_json(getattr(cfg, k, None)))
        if not (cfg.resume or cfg.analysis_only or cfg.audit_only):
            logger.error("output directory already holds a run; use --resume (same configuration) or --overwrite")
            return 1
        if diff:
            logger.error(f"--resume refused: configuration differs from the existing run in {diff}; use another "
                         f"--output_dir or --overwrite")
            return 1
    samples = build_samples(df, cfg, load_screening_resources(cfg))
    audit = data_audit(df, vrep, samples, cfg)
    atomic_write_json(os.path.join(dirs['root'], 'data_audit.json'), audit)
    write_csv(audit_to_flat(audit), os.path.join(dirs['root'], 'data_audit.csv'))
    env = environment_snapshot()
    atomic_write_json(os.path.join(dirs['root'], 'environment.json'), env)
    conds = build_conditions(cfg)
    plan = analysis_plan(cfg)
    atomic_write_json(os.path.join(dirs['root'], 'config.json'), config_record(cfg, plan, conds))
    for r in samples.funnel:
        logger.info(f"  [{r['stage']}] {r['step']}: {r['rows']} rows / {r['unique_words']} words")
    model_status, model_errors = {}, {}
    if cfg.audit_only:
        write_report(cfg, dirs, audit, env, conds, {}, {}, {}, plan, 'audit_only')
        atomic_write_json(os.path.join(dirs['root'], 'manifest.json'),
                          {'mode': 'audit_only', 'data_sha256': cfg.data_sha256, 'script_version': SCRIPT_VERSION})
        logger.info("audit-only run complete (no model was loaded)")
        return 0
    try:
        models = resolve_models(cfg.models)
    except ValueError as e:
        logger.error(str(e))
        return 1
    if not cfg.analysis_only:
        if not torch_available():
            logger.error("torch and transformers are required to score models")
            return 1
        active = active_conditions(cfg, conds)
        for spec in models:
            logger.info(f"── model {spec.alias} ({spec.repo_id}, {spec.variant})")
            try:
                model_status[spec.alias] = process_model(spec, cfg, conds, active, samples, context_df, dirs)
            except Exception as e:                                   # noqa: BLE001
                model_errors[spec.alias] = {'error': f"{type(e).__name__}: {e}", 'traceback': traceback.format_exc()}
                model_status[spec.alias] = {'alias': spec.alias, 'repo_id': spec.repo_id, 'status': 'FAILED',
                                            'conditions': {}}
                logger.error(f"[{spec.alias}] FAILED: {type(e).__name__}: {e}")
                atomic_write_json(os.path.join(dirs['logs'], 'model_load_errors.json'), model_errors)
                if not cfg.continue_on_model_error:
                    write_report(cfg, dirs, audit, env, conds, model_status, model_errors, {}, plan, 'aborted')
                    return 3
            gc.collect()
        atomic_write_json(os.path.join(dirs['logs'], 'model_load_errors.json'), model_errors)
    sp = os.path.join(dirs['logs'], 'model_status.json')
    if os.path.exists(sp) and not cfg.overwrite:
        model_status = {**read_json(sp), **model_status}
    atomic_write_json(sp, model_status)
    if cfg.smoke_test:
        ok, rows = smoke_report(model_status, dirs)
        print('\nSMOKE TEST')
        for a, n, r in rows:
            print(f"  [{r}] {a}: {n}")
        print(f"SMOKE TEST {'PASSED' if ok else 'FAILED'}  (outputs: {os.path.abspath(cfg.output_dir)})")
        write_report(cfg, dirs, audit, env, conds, model_status, model_errors, {}, plan, 'smoke_test')
        return 0 if ok and not model_errors else 4
    res = run_analyses(cfg, conds, samples, context_df, dirs)
    mode = 'analysis_only' if cfg.analysis_only else ('subset' if samples.subset else 'full')
    write_report(cfg, dirs, audit, env, conds, model_status, model_errors, res, plan, mode)
    atomic_write_json(os.path.join(dirs['root'], 'manifest.json'), {
        'mode': mode, 'script_version': SCRIPT_VERSION, 'data_sha256': cfg.data_sha256,
        'analysis_plan_sha256': sha256_json(plan), 'models': model_status, 'model_errors': list(model_errors),
        'n_scoresets_analysed': res.get('n_scoresets'), 'skipped_caches': res.get('skipped_caches'),
        'tables': sorted(k for k, v in res.get('tables', {}).items() if v is not None and len(v)),
        'figures': [os.path.relpath(p, dirs['root']) for p in res.get('figures', [])], 'not_run': res.get('not_run')})
    logger.info(f"done → {os.path.abspath(cfg.output_dir)}/REPORT.md")
    return 3 if model_errors else 0


# ════════════════════════════════════════════════════════════════════════════
# SELF-TESTS  (python semantic_ambiguity_layerwise.py --self_test)
# ════════════════════════════════════════════════════════════════════════════
def _test_cfg(**kw):
    a = build_parser().parse_args([])
    a.dtype, a.resolved_device, a.device_map = 'fp32', 'cpu', 'none'
    a.context_sha256, a.context_info, a.task_prompts_sha256 = None, {}, None
    a.screening_sha256, a.script_version = None, SCRIPT_VERSION
    a.n_bootstrap, a.n_permutations, a.batch_size = 60, 19, 4
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _find_csv():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (DEFAULT_DATA_CSV, os.path.join(here, DEFAULT_DATA_CSV)):
        if os.path.exists(p):
            return p
    return None


def _synthetic_csv(path, mutate=None):
    rows = []
    rng = np.random.default_rng(0)
    for app in 'ABC':
        for i in range(8):
            w = f"p{app.lower()}" + ''.join(chr(97 + (i * 7 + j) % 26) for j in range(3))
            sub = int(rng.integers(20, 5000))
            rows.append({'word': w, 'provenance': 'published', 'source_set': f'Rodd2002-{app}', 'pool_condition': '',
                         'appendix': app, 'experiment': APPENDIX_DESIGN[app][0], 'modality': APPENDIX_DESIGN[app][1],
                         'ambiguity': 'ambiguous' if i % 2 == 0 else 'unambiguous',
                         'senses': '' if app == 'A' else ('many' if (i // 2) % 2 == 0 else 'few'),
                         'length': len(w), 'subtlex': sub, 'log_subtlex': round(np.log10(sub + 1), 3),
                         'gcide_etymologies': 1 + i % 3, 'wn_n_all': 2 + i, 'concreteness': 3.0 + 0.1 * i})
    for i in range(30):
        w = 'de' + chr(97 + i % 26) + chr(97 + (i // 26) % 26) + ('xyz' if i >= 24 else '')
        sub = int(10 ** rng.uniform(1.5, 4.5))
        rows.append({'word': w, 'provenance': 'wordnet-derived', 'source_set': 'WordNet 3.0',
                     'pool_condition': 'homonym-candidate', 'appendix': '', 'experiment': '', 'modality': '',
                     'ambiguity': '', 'senses': '', 'length': len(w), 'subtlex': sub,
                     'log_subtlex': round(np.log10(sub + 1), 3), 'gcide_etymologies': [0, 1, 2, 3, np.nan][i % 5],
                     'wn_n_all': 1 + i % 7, 'concreteness': 2 + (i % 4)})
    df = pd.DataFrame(rows)
    for c in EXPECTED_COLUMNS:
        if c not in df:
            df[c] = np.nan
    df = df[EXPECTED_COLUMNS]
    if mutate:
        df = mutate(df)
    df.to_csv(path, index=False)
    return df


def run_self_tests(verbose: bool = True) -> bool:
    import tempfile
    import unittest
    csv = _find_csv()
    have_torch = torch_available()
    try:
        import statsmodels.api as sm_api
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        sm_api = multipletests = None
    try:
        from sklearn.linear_model import Ridge
        from sklearn.metrics import roc_auc_score
    except ImportError:
        Ridge = roc_auc_score = None
    tmp_root = tempfile.mkdtemp(prefix='semamb_selftest_')
    shared = {}

    def tiny(kind='llama'):
        spec = ModelSpec(f'tiny-{kind}-TEST', f'local/{kind}', 'test',
                         'instruct' if kind.endswith('_chat') else 'base', test_model=kind)
        sm = ScoringModel(spec, _test_cfg())
        sm.verify_readout([sm.encode_unit(u) for u in build_units(
            build_conditions(_test_cfg())['ld_isolated'], _test_cfg(), _words(['bank', 'zygote', 'paper']), None)])
        return sm

    def _words(ws):
        return Samples(pd.DataFrame({'word': ws}), pd.DataFrame({'word': []}), {}, [], pd.DataFrame(), ws, False, {})

    def real():
        df, rep = load_and_validate(csv)
        cfg = _test_cfg()
        return df, build_samples(df, cfg, load_screening_resources(cfg))

    class T(unittest.TestCase):
        # ── data ────────────────────────────────────────────────────────────
        def test_01_schema_validated(self):
            p = os.path.join(tmp_root, 's.csv')
            _synthetic_csv(p)
            df, rep = load_and_validate(p)
            self.assertFalse(rep['columns_missing'])
            _synthetic_csv(p, lambda d: d.drop(columns=['wn_n_all']))
            with self.assertRaises(CriticalError):
                load_and_validate(p)
            _synthetic_csv(p, lambda d: d.assign(provenance=d['provenance'].replace('published', 'pub')))
            with self.assertRaises(CriticalError):
                load_and_validate(p)
            _synthetic_csv(p, lambda d: d.assign(modality=d['modality'].replace('auditory', 'visual')))
            with self.assertRaises(CriticalError):
                load_and_validate(p)

        def test_02_published_counts_and_cell_membership(self):
            if not csv:
                self.skipTest('ambiguity_items_combined.csv not found')
            df, s = real()
            pub = s.stage1
            self.assertEqual((len(pub), pub['word'].nunique()), (402, 309))
            a = pub[pub['appendix'] == 'A']
            self.assertEqual(a['ambiguity'].value_counts().to_dict(), {'ambiguous': 124, 'unambiguous': 58})
            for app, n in (('B', 32), ('C', 23)):
                cells = pub[pub['appendix'] == app].groupby(['ambiguity', 'senses']).size()
                self.assertEqual(sorted(cells.index.tolist()), [('ambiguous', 'few'), ('ambiguous', 'many'),
                                                                 ('unambiguous', 'few'), ('unambiguous', 'many')])
                self.assertTrue((cells == n).all())
            for app, (e, m) in APPENDIX_DESIGN.items():
                r = pub[pub['appendix'] == app]
                self.assertTrue(((r['experiment'] == e) & (r['modality'] == m)).all())
            self.assertTrue((pub.groupby('word')['ambiguity'].nunique() == 1).all())

        def test_03_stage1_stage2_separation(self):
            if not csv:
                self.skipTest('ambiguity_items_combined.csv not found')
            df, s = real()
            self.assertTrue((s.stage1['provenance'] == 'published').all())
            self.assertTrue((s.stage2_base['provenance'] == 'wordnet-derived').all())
            self.assertFalse(set(s.stage1['word']) & set(s.stage2_base['word']))
            for d in (design_stage1_A(s.stage1), design_stage1_BC(s.stage1, 'B')):
                self.assertTrue(set(d.words) <= set(s.stage1['word']))
            self.assertTrue(set(design_stage2(s.stage2['drop'], 'primary', 'drop').words) <= set(s.stage2_base['word']))

        def test_04_repeated_words_missing_values_and_folds(self):
            if not csv:
                self.skipTest('ambiguity_items_combined.csv not found')
            df, s = real()
            pub = s.stage1
            multi = pub.groupby('word')['appendix'].nunique()
            w = multi.index[multi > 1][0]
            self.assertEqual(len(pub[pub['word'] == w]), multi[w])
            dA = design_stage1_A(pub)
            missing = pub[(pub['appendix'] == 'A') & pub[['log_subtlex', 'concreteness']].isna().any(axis=1)]
            self.assertEqual(dA.n + len(dA.excluded), int((pub['appendix'] == 'A').sum()))
            self.assertEqual(set(dA.excluded['row_id']), set(missing['row_id']))
            self.assertTrue(all('log_subtlex' in m for m in dA.excluded['missing']))
            groups = np.array(['a', 'a', 'b', 'c', 'c', 'c', 'd', 'e', 'e', 'f', 'g', 'h'])
            for labels in (None, np.array([1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1])):
                folds = make_group_folds(groups, 3, 0, labels)
                for g in set(groups):
                    self.assertEqual(len(set(folds[groups == g])), 1)
            with self.assertRaises(ValueError):
                make_group_folds(np.array(['a', 'a']), 2, 0, np.array([0, 1]))

        def test_05_stage2_eligibility_recomputed_and_exclusions_documented(self):
            p = os.path.join(tmp_root, 's5.csv')
            sdf = _synthetic_csv(p)
            df, _ = load_and_validate(p)
            s = build_samples(df, _test_cfg(), load_screening_resources(_test_cfg()))
            d = sdf[sdf['provenance'] == 'wordnet-derived']
            manual = d[d['gcide_etymologies'].notna() & d['wn_n_all'].notna() & d['length'].between(4, 6) &
                       d['log_subtlex'].between(2.0, 4.0)]
            self.assertEqual(len(s.stage2_base), len(manual))
            ex = s.exclusions[s.exclusions['stage'] == 'stage2']
            excluded_by_filters = ex[~ex['step'].str.startswith('gcide_etymologies == 0')]
            self.assertEqual(len(excluded_by_filters) + len(s.stage2_base), len(d))
            self.assertTrue(excluded_by_filters['step'].str.len().gt(0).all())
            if csv:
                df, s = real()
                d = df[df['provenance'] == 'wordnet-derived']
                self.assertEqual(len(s.stage2_base), int((d['gcide_etymologies'].notna() & d['length'].between(4, 6) &
                                                          d['log_subtlex'].between(2.0, 4.0)).sum()))

        def test_06_log_subtlex_not_relogged(self):
            p = os.path.join(tmp_root, 's6.csv')
            _synthetic_csv(p)
            df, rep = load_and_validate(p)
            self.assertTrue(rep['checks']['log_subtlex_is_log10_subtlex_plus_1']['pass'])
            s = build_samples(df, _test_cfg(), load_screening_resources(_test_cfg()))
            d = design_stage2(s.stage2['drop'], 'primary', 'drop')
            exp = s.stage2['drop'].dropna(subset=STAGE2_SPECS['primary'][0])['log_subtlex'].values
            np.testing.assert_array_equal(d.raw['log_subtlex'].values, exp)
            _synthetic_csv(p, lambda x: x.assign(log_subtlex=np.log10(x['log_subtlex'] + 1)))
            with self.assertRaises(CriticalError):
                load_and_validate(p)

        def test_07_gcide_zero_and_missing(self):
            d = pd.DataFrame({'gcide_etymologies': [0, 1, 2, np.nan, 0], 'row_id': list('abcde'), 'word': list('abcde')})
            a, gone = apply_gcide_policy(d, 'drop')
            self.assertEqual(list(a['gcide_etymologies'].fillna(-1)), [1, 2, -1])
            self.assertEqual(list(gone['row_id']), ['a', 'e'])
            self.assertEqual(list(apply_gcide_policy(d, 'recode_to_one')[0]['gcide_etymologies'].fillna(-1)),
                             [1, 1, 2, -1, 1])
            self.assertEqual(list(apply_gcide_policy(d, 'retain_as_zero')[0]['gcide_etymologies'].fillna(-1)),
                             [0, 1, 2, -1, 0])
            with self.assertRaises(ValueError):
                apply_gcide_policy(d, 'impute')
            p = os.path.join(tmp_root, 's7.csv')
            sdf = _synthetic_csv(p)
            df, _ = load_and_validate(p)
            s = build_samples(df, _test_cfg(), load_screening_resources(_test_cfg()))
            na_words = set(sdf.loc[(sdf['provenance'] == 'wordnet-derived') & sdf['gcide_etymologies'].isna(), 'word'])
            self.assertFalse(na_words & set(s.stage2_base['word']))
            self.assertTrue(na_words <= set(s.exclusions.loc[s.exclusions['step'].str.contains('missing'), 'word']))

        def test_08_factorial_contrasts_and_signs(self):
            r = pd.DataFrame({'word': [f'w{i}' for i in range(40)], 'row_id': [f'r{i}' for i in range(40)],
                              'appendix': 'B', 'ambiguity': ['ambiguous', 'unambiguous'] * 20,
                              'senses': (['many'] * 2 + ['few'] * 2) * 10})
            d = design_stage1_BC(r, 'B')
            cell = {('ambiguous', 'many'): 5.0, ('ambiguous', 'few'): 7.0, ('unambiguous', 'many'): 4.0,
                    ('unambiguous', 'few'): 4.0}
            y = np.array([cell[(a, s)] for a, s in zip(r['ambiguity'], r['senses'])])[:, None]
            b = ols_multi(d.X_raw(), y)['beta'][:, 0]
            self.assertAlmostEqual(b[1], (5 + 7) / 2 - (4 + 4) / 2)          # ambiguous − unambiguous
            self.assertAlmostEqual(b[2], (5 + 4) / 2 - (7 + 4) / 2)          # many − few
            self.assertAlmostEqual(b[3], (5 - 7) - (4 - 4))                   # interaction contrast
            self.assertEqual(d.expected, {'senses': '-', 'ambiguity': '0'})
            yz = y + np.random.default_rng(0).normal(0, 0.1, y.shape)
            bs = std_betas(d, yz)[:, 0]
            self.assertAlmostEqual(bs[0], ols_multi(d.X_raw(), yz)['beta'][1, 0] / yz.std(ddof=1), places=10)
            a = pd.DataFrame({'word': [f'a{i}' for i in range(20)], 'row_id': [f'a{i}' for i in range(20)],
                              'appendix': 'A', 'ambiguity': ['ambiguous', 'unambiguous'] * 10, 'wn_n_all': range(20),
                              'log_subtlex': np.linspace(2, 4, 20), 'length': [4, 5] * 10,
                              'concreteness': np.linspace(1, 5, 20)})
            dA = design_stage1_A(a)
            self.assertEqual(dA.raw['ambiguity'].tolist(), [1.0, 0.0] * 10)  # ambiguous = 1

        # ── tokenisation and scoring ────────────────────────────────────────
        def test_09_base_full_text_and_chat_templates(self):
            if not have_torch:
                self.skipTest('torch/transformers not installed')
            cond = build_conditions(_test_cfg())['ld_isolated']
            u = build_units(cond, _test_cfg(), _words(['banking']), None)[0]
            sm = tiny('llama')
            e = sm.encode_unit(u)
            self.assertEqual(e.input_ids, sm.tokenizer(u.text)['input_ids'])
            self.assertEqual(e.input_ids.count(sm.tokenizer.bos_token_id), 1)
            self.assertEqual(e.input_ids[0], sm.tokenizer.bos_token_id)
            q = tiny('qwen2')
            self.assertIsNone(q.tokenizer.bos_token_id)
            self.assertEqual(q.encode_unit(u).input_ids, q.tokenizer(u.text)['input_ids'])
            for kind, marker, trims in (('llama_chat', '<|start_header_id|>user', True),
                                        ('qwen2_chat', '<|im_start|>user', False)):
                c = tiny(kind)
                ec = c.encode_unit(u)
                ref = _as_id_list(c.tokenizer.apply_chat_template([{'role': 'user', 'content': u.text}], tokenize=True,
                                                                  add_generation_prompt=True, **c.template_kwargs))
                self.assertEqual(ec.input_ids, ref)
                self.assertIn(marker, ec.full_text)
                self.assertTrue(ec.targets[0]['ok'])
                self.assertEqual(('banking\n<|' in ec.full_text), not trims)
            self.assertIn('26 Jul 2024', tiny('llama_chat').encode_unit(u).full_text)
            with self.assertRaises(ChatTemplateError):
                ScoringModel(ModelSpec('nochat', 'local/llama', 'test', 'instruct', test_model='llama'), _test_cfg(),
                             load_weights=False)

        def test_10_alignment_across_tokenizer_families(self):
            if not have_torch:
                self.skipTest('torch/transformers not installed')
            # (text, target, leading whitespace expected inside the first target token; None = vocabulary-dependent)
            cases = [('Items:\nbank\n', 'bank', False), ('Text: river', 'river', True),
                     ('the river bank.', 'bank', None), ('a quixotically odd idea', 'quixotically', None)]
            for style in ('llama3', 'qwen2', 'gpt2'):
                tok = build_tiny_tokenizer(style, False, None)
                dec = lambda x, tok=tok: tok.decode(x, clean_up_tokenization_spaces=False)
                for text, w, ws in cases:
                    enc = tok(text, return_offsets_mapping=True)
                    s = text.index(w)
                    al = align_target(enc['offset_mapping'], enc['input_ids'], text, (s, s + len(w)), w, dec)
                    self.assertTrue(al['ok'], (style, text, al['reason']))
                    self.assertEqual(al['leading_ws_absorbed'], dec(al['target_ids'][:1])[:1].isspace(), (style, text))
                    if ws is not None:
                        self.assertEqual(al['leading_ws_absorbed'], ws, (style, text))
                    rebuilt = offsets_by_decoding(enc['input_ids'], text, dec)
                    al2 = align_target(rebuilt, enc['input_ids'], text, (s, s + len(w)), w, dec)
                    self.assertEqual((al['t0'], al['t1']), (al2['t0'], al2['t1']))
                text = '"bank" is here'
                enc = tok(text, return_offsets_mapping=True)
                al = align_target(enc['offset_mapping'], enc['input_ids'], text, (1, 5), 'bank', dec)
                straddles = any(x < 1 < y or x < 5 < y for x, y in enc['offset_mapping'])
                self.assertEqual(al['ok'], not straddles, (style, al['reason']))
                multi = align_target(*[tok('a quixotically', return_offsets_mapping=True)[k]
                                       for k in ('offset_mapping', 'input_ids')], 'a quixotically', (2, 14),
                                     'quixotically', dec)
                self.assertGreaterEqual(multi['n_sub'], 2)

        def test_10b_alignment_rules_on_known_offsets(self):
            text, dec = 'Items:\nbank\n', (lambda ids: 'bank')
            self.assertTrue(align_target([(0, 5), (5, 6), (6, 7), (7, 11), (11, 12)], [0, 1, 2, 3, 4], text, (7, 11),
                                         'bank', dec)['ok'])
            self.assertEqual(align_target([(0, 5), (5, 8), (8, 11), (11, 12)], [0, 1, 2, 3], text, (7, 11), 'bank',
                                          dec)['reason'], 'prefix_merged_into_target_token')
            self.assertEqual(align_target([(0, 5), (5, 6), (6, 7), (7, 12)], [0, 1, 2, 3], text, (7, 11), 'bank',
                                          dec)['reason'], 'suffix_merged_into_target_token')
            nl = align_target([(0, 5), (5, 6), (6, 11), (11, 12)], [0, 1, 2, 3], text, (7, 11), 'bank', dec)
            self.assertTrue(nl['ok'] and nl['leading_ws_absorbed'])
            self.assertEqual(align_target([(0, 11)], [0], text, (7, 11), 'bank', dec)['reason'],
                             'prefix_merged_into_target_token')
            self.assertEqual(align_target([(0, 7), (7, 9), (10, 11)], [0, 1, 2], text, (7, 11), 'bank', dec)['reason'],
                             'span_not_fully_covered')

        def test_11_next_token_scoring(self):
            if not have_torch:
                self.skipTest('torch/transformers not installed')
            torch = _torch()
            cond = build_conditions(_test_cfg())['ld_isolated']
            words = ['bank', 'zygote', 'quixotic', 'a', 'paper']
            for kind in ('llama', 'qwen2', 'llama_chat', 'qwen2_chat'):
                sm = tiny(kind)
                enc = [sm.encode_unit(u) for u in build_units(cond, _test_cfg(), _words(words), None)]
                self.assertTrue(all(e.targets[0]['ok'] for e in enc))
                nsub = [e.targets[0]['n_sub'] for e in enc]
                self.assertIn(1, nsub)
                self.assertTrue(any(n >= 2 for n in nsub))
                agg = aggregate_surprisal(pack_scores(enc, sm.score_encoded(enc, want_reps=False, batch_size=3)))
                with torch.inference_mode():
                    for i, e in enumerate(enc):
                        lp = torch.log_softmax(sm.model(input_ids=torch.tensor([e.input_ids])).logits.float()[0], -1)
                        t = e.targets[0]
                        manual = -sum(float(lp[p - 1, e.input_ids[p]]) for p in range(t['t0'], t['t1'] + 1)) / LN2
                        self.assertAlmostEqual(agg['sum'][i, -1], manual, places=3)
                        self.assertAlmostEqual(agg['first'][i, -1], -float(lp[t['t0'] - 1, e.input_ids[t['t0']]]) / LN2,
                                               places=3)

        def test_12_logit_lens_intermediate_vs_final(self):
            if not have_torch:
                self.skipTest('torch/transformers not installed')
            torch = _torch()
            sm = tiny('llama')
            self.assertEqual(sm.lens_check['status'], 'OK')
            self.assertTrue(sm.last_is_post_norm)
            e = sm.encode_unit(build_units(build_conditions(_test_cfg())['ld_isolated'], _test_cfg(),
                                           _words(['bank']), None)[0])
            ids, am = sm._pad([e])
            with torch.inference_mode():
                hs = sm._hidden_states(ids, am)
                h = hs[2][0, :5]
                lens = sm.lens_logprobs(h, 2)
                self.assertLess(float((lens - torch.log_softmax(sm.head(sm.final_norm(h)).float(), -1)).abs().max()), 1e-5)
                self.assertGreater(float((lens - torch.log_softmax(sm.head(h).float(), -1)).abs().max()), 1e-3)
                ref = torch.log_softmax(sm.model(input_ids=ids, attention_mask=am).logits[0, :5].float(), -1)
                self.assertLess(float((sm.lens_logprobs(hs[-1][0, :5], sm.num_layers) - ref).abs().max()), 1e-4)

        def test_13_subtoken_aggregation_and_word_boundary(self):
            if not have_torch:
                self.skipTest('torch/transformers not installed')
            torch = _torch()
            sm = tiny('llama')
            tok = sm.tokenizer
            B = set(sm.boundary_ids.tolist())
            for s, inside in (('\n', True), (' the', True), ('.', True), ('bank', False)):
                self.assertEqual(tok.convert_tokens_to_ids(tok.tokenize(s))[0] in B, inside, s)
            self.assertTrue(set(tok.all_special_ids) <= B)
            iso = build_units(build_conditions(_test_cfg())['ld_isolated'], _test_cfg(), _words(['bank', 'zygote']), None)
            frame = CONTEXT_FRAME_TEMPLATE.split(TEXT_SLOT)[0]
            ctxu = Unit('ctx|river', frame + 'by the river', [('river', len(frame) + 7, len(frame) + 12)],
                        [{'word': 'river', 'context_char_start': len(frame)}])
            enc = [sm.encode_unit(u) for u in iso + [ctxu]]
            self.assertEqual([e.targets[0]['first_token_is_boundary'] for e in enc], [False, False, True])
            self.assertGreater(len(enc[2].targets[0]['pre_target_token_indices']), 0)
            arr = pack_scores(enc, sm.score_encoded(enc, want_reps=False))
            agg = aggregate_surprisal(arr)
            np.testing.assert_allclose(agg['mean'], agg['sum'] / arr['n_sub'][:, None])
            mask = torch.zeros(sm.head.weight.shape[0], dtype=torch.bool)
            mask[torch.as_tensor(sm.boundary_ids)] = True
            with torch.inference_mode():
                for i, e in enumerate(enc):
                    lp = torch.log_softmax(sm.model(input_ids=torch.tensor([e.input_ids])).logits.float()[0], -1)
                    t = e.targets[0]
                    eow = -float(torch.logsumexp(lp[t['t1']][mask], 0)) / LN2
                    bow = -float(torch.logsumexp(lp[t['t0'] - 1][mask], 0)) / LN2
                    expect = agg['sum'][i, -1] + eow - (bow if t['first_token_is_boundary'] else 0.0)
                    self.assertAlmostEqual(agg['whole_word'][i, -1], expect, places=3)
            # identical prefix ⇒ identical bow mass for every isolated item (no item effect)
            self.assertAlmostEqual(float(arr['bow_logmass'][0, -1]), float(arr['bow_logmass'][1, -1]), places=5)

        def test_14_padding_never_shifts_targets(self):
            if not have_torch:
                self.skipTest('torch/transformers not installed')
            for kind in ('llama', 'qwen2_chat'):
                sm = tiny(kind)
                enc = [sm.encode_unit(u) for u in build_units(build_conditions(_test_cfg())['ld_isolated'], _test_cfg(),
                                                              _words(['a', 'quixotically', 'bank', 'zygote']), None)]
                self.assertEqual(sm.check_padding(enc)['status'], 'PASS')
                ids, am = sm._pad(enc)
                for b, e in enumerate(enc):
                    t = e.targets[0]
                    self.assertTrue(bool(am[b, t['t0'] - 1:t['t1'] + 1].all()))
                    self.assertEqual(ids[b, t['t0']:t['t1'] + 1].tolist(), t['target_ids'])

        def test_15_hidden_state_count_and_width(self):
            if not have_torch:
                self.skipTest('torch/transformers not installed')
            torch = _torch()
            for kind in ('llama', 'qwen2'):
                sm = tiny(kind)
                e = sm.encode_unit(build_units(build_conditions(_test_cfg())['ld_isolated'], _test_cfg(),
                                               _words(['bank']), None)[0])
                with torch.inference_mode():
                    hs = sm._hidden_states(*sm._pad([e]))
                self.assertEqual(len(hs), sm.config.num_hidden_layers + 1)
                self.assertEqual(hs[-1].shape[-1], sm.config.hidden_size)

        # ── contexts, caches, end-to-end ────────────────────────────────────
        def test_16_context_construction_without_fabrication(self):
            p = os.path.join(tmp_root, 'ctx.csv')
            pd.DataFrame({'word': ['bank', 'bank'], 'context_id': ['c1', 'c2'],
                          'text': ['she paid the {TARGET} today', '{TARGET} of the river']}).to_csv(p, index=False)
            t, info = load_context_file(p)
            self.assertFalse(info['has_meaning_ids'] or info['has_sense_ids'] or info['has_dominance_labels'])
            self.assertEqual(info['n_contexts_without_pre_target_words'], 1)
            cfg = _test_cfg(context_file=p, context_info=info)
            self.assertTrue(homonymy_polysemy_support(pd.DataFrame({'word': []}), info)['status'].startswith('UNSUPPORTED'))
            self.assertTrue(not_run_registry(cfg, {})['homonymy_vs_polysemy_primary'].startswith('UNSUPPORTED'))
            self.assertTrue(not_run_registry(cfg, {})['dominant_vs_subordinate'].startswith('NOT_RUN'))
            conds = build_conditions(cfg)
            sam = _words(['bank'])
            bare = build_units(conds['context_bare'], cfg, sam, t)
            sent = build_units(conds['context_sentence'], cfg, sam, t)
            frame = CONTEXT_FRAME_TEMPLATE.split(TEXT_SLOT)[0]
            self.assertTrue(bare[0].text.startswith(frame) and all(u.text.startswith(frame) for u in sent))
            self.assertEqual(bare[0].text, frame + 'bank')
            self.assertEqual({u.meta[0]['meaning_id'] for u in sent}, {''})          # nothing invented
            a = pd.Series({'meaning_id': 'm1', 'sense_id': 's1'})
            self.assertEqual(pair_relation(a, pd.Series({'meaning_id': 'm2', 'sense_id': 's9'})), 'different_meaning')
            self.assertEqual(pair_relation(a, pd.Series({'meaning_id': 'm1', 'sense_id': 's2'})),
                             'different_sense_same_meaning')
            self.assertEqual(pair_relation(a, pd.Series({'meaning_id': 'm1', 'sense_id': ''})),
                             'same_meaning_sense_unannotated')
            for bad in (['the ambiguous {TARGET}'], ['no slot here']):
                pd.DataFrame({'word': ['bank'], 'context_id': ['c1'], 'text': bad}).to_csv(p, index=False)
                with self.assertRaises(CriticalError):
                    load_context_file(p)
            self.assertEqual(load_screening_resources(cfg)['inflected']['status'], 'NOT_AVAILABLE')

        def test_17_cache_invalidation_and_stale_refusal(self):
            if not (have_torch and csv):
                self.skipTest('needs torch/transformers and the CSV')
            out = os.path.join(tmp_root, 'run')
            ctxp = os.path.join(tmp_root, 'TEST_FIXTURE_contexts.csv')
            df, _ = load_and_validate(csv)
            words = sorted(df.loc[df['provenance'] == 'published', 'word'].unique())[:12]
            pd.DataFrame([{'word': w, 'context_id': f'c{j}', 'text': tx, 'meaning_id': m, 'sense_id': s,
                           'dominance': dm, 'validation_status': 'test-fixture'}
                          for w in words for j, (tx, m, s, dm) in enumerate([
                              ('she saw the {TARGET} there', 'm1', 's1', 'dominant'),
                              ('they spoke of the {TARGET} again', 'm1', 's2', 'dominant'),
                              ('nobody liked the {TARGET} much', 'm2', 's3', 'subordinate')])]).to_csv(ctxp, index=False)
            ctx_df, info = load_context_file(ctxp)
            cfg = _test_cfg(output_dir=out, max_items=60, run_list_protocol=True, list_size=5, list_repeats=2,
                            context_file=ctxp, context_sha256=sha256_file(ctxp), context_info=info)
            dirs = output_dirs(out)
            samples = build_samples(df, cfg, load_screening_resources(cfg))
            conds = build_conditions(cfg)
            active = active_conditions(cfg, conds)
            spec = TEST_MODEL_REGISTRY[0]
            st = process_model(spec, cfg, conds, active, samples, ctx_df, dirs)
            self.assertTrue(all(c['status'].startswith('SCORED') for c in st['conditions'].values()), st)
            self.assertTrue(all(c['status'].startswith('CACHED') for c in
                                process_model(spec, cfg, conds, active, samples, ctx_df, dirs)['conditions'].values()))
            ce = CacheEntry(out, spec.alias, 'ld_isolated')
            key = ce.manifest()['key']
            for field_, val in (('dtype', 'bf16'), ('template_sha256', 'x'), ('resolved_revision', 'other'),
                                ('units_sha256', 'other-data'), ('boundary_definition', 'other')):
                self.assertEqual(ce.mismatches(dict(key, **{field_: val})), [field_])
            ak = analysis_key(cfg, conds['ld_isolated'], build_units(conds['ld_isolated'], cfg, samples, None))
            self.assertEqual(ce.mismatches(ak, 'analysis_key'), [])
            for field_, c2 in (('chat_date_string', _test_cfg(**{**vars(cfg), 'chat_date_string': '1 Jan 2025'})),
                               ('dtype', _test_cfg(**{**vars(cfg), 'dtype': 'bf16'}))):
                self.assertIn(field_, ce.mismatches(analysis_key(c2, conds['ld_isolated'], build_units(
                    conds['ld_isolated'], c2, samples, None)), 'analysis_key'))
            changed = PromptCondition('ld_isolated', SOURCE_LD_TEMPLATE.replace('Items:', 'Words:'), 'isolated',
                                      'primary', '', 'confirmatory')
            self.assertIn('template_sha256', ce.mismatches(analysis_key(cfg, changed, build_units(
                changed, cfg, samples, None)), 'analysis_key'))
            man = ce.manifest()
            man['key']['dtype'] = 'bf16'
            atomic_write_json(ce.manifest_path, man)
            st3 = process_model(spec, cfg, conds, active, samples, ctx_df, dirs)
            self.assertIn('stale cache', st3['conditions']['ld_isolated']['status'])
            cfg.overwrite = True
            self.assertTrue(process_model(spec, cfg, conds, active, samples, ctx_df, dirs)['conditions']
                            ['ld_isolated']['status'].startswith('SCORED'))
            cfg.overwrite = False
            process_model(TEST_MODEL_REGISTRY[3], cfg, conds, active, samples, ctx_df, dirs)
            item = pd.read_csv(os.path.join(dirs['item_scores'], safe_name(spec.alias), 'ld_isolated_item_layer_scores.csv'))
            for col in ('n_sub', 'target_tokens', 'offset_method', 'align_reason', 'readout_type',
                        'surprisal_whole_word_bits'):
                self.assertIn(col, item.columns)
            shared.update(cfg=cfg, dirs=dirs, samples=samples, conds=conds, ctx_df=ctx_df)

        def test_18_end_to_end_tables(self):
            if 'cfg' not in shared:
                self.skipTest('depends on test_17 (torch + CSV)')
            c = shared
            res = run_analyses(c['cfg'], c['conds'], c['samples'], c['ctx_df'], c['dirs'])
            T = res['tables']
            df, vrep = load_and_validate(csv)
            write_report(c['cfg'], c['dirs'], data_audit(df, vrep, c['samples'], c['cfg']), environment_snapshot(),
                         c['conds'], read_json(os.path.join(c['dirs']['logs'], 'model_status.json'))
                         if os.path.exists(os.path.join(c['dirs']['logs'], 'model_status.json')) else {}, {}, res,
                         analysis_plan(c['cfg']), 'subset')
            report = open(os.path.join(c['dirs']['root'], 'REPORT.md'), encoding='utf-8').read()
            for section in ('What was directly measured', 'Confirmatory results', 'Homonymy vs polysemy',
                            'Run / skipped / unsupported', 'Limitations'):
                self.assertIn(section, report)
            for name in ('stage1_published_appendix_A', 'stage1_published_appendix_B', 'stage2_wordnet_derived'):
                for col in ('n', 'p_hc3', 'p_fdr_layers', 'p_fdr_global_term', 'beta_std', 'ci_low', 'readout_type',
                            'construct_status'):
                    self.assertIn(col, T[name].columns, (name, col))
            self.assertTrue({'experiment', 'modality'} <= set(T['stage1_published_appendix_C'].columns))
            conf = T['confirmatory_hypothesis_tests']
            self.assertTrue(len(conf) and (conf['role'] == 'confirmatory').all())
            self.assertTrue({'p_holm', 'evidence', 'ci95_low', 'summary', 'mde80'} <= set(conf.columns))
            eq = conf[conf['kind'] == 'equivalence']
            self.assertTrue(conf.loc[conf['kind'] != 'equivalence', 'equivalence_attainable'].isna().all())
            # bootstrap SE of the 60-item subset is far above SESOI/z: TOST can never succeed, and must say so
            self.assertTrue(len(eq) and not eq['equivalence_attainable'].astype(bool).any())
            self.assertIn('Equivalence is not attainable', report)
            self.assertTrue(set(conf['evidence']) <= {'supported', 'contradicted', 'inconclusive',
                                                      'supported (equivalent to zero within ±SESOI)',
                                                      'contradicted (non-zero effect)'})
            self.assertFalse(any(col in ('gate', 'stage1_gate') for df in T.values() if df is not None
                                 for col in df.columns))
            for name in ('meanings_vs_senses_pattern', 'exclusions', 'construct_registry', 'analysis_registry',
                         'tokenization_bias_audit', 'tokenization_balance', 'list_protocol_order_diagnostics',
                         'list_protocol_surrounding_items', 'context_effects', 'context_balance',
                         'homonymy_polysemy_distance', 'context_dominance', 'multiple_testing'):
                self.assertTrue(len(T[name]), name)
            hp = T['homonymy_polysemy_distance']
            self.assertTrue((hp['role'] == 'exploratory (annotations not validated)').all())
            # the fixture reuses the same three templates for every word: the code must refuse to interpret it
            self.assertTrue(hp.loc[hp['evidence'] != 'descriptive', 'evidence'].str.startswith('not interpretable').all())
            # publication figures: exact ACL text width, embedded TrueType text, PNG + caption sidecars
            figs = {os.path.basename(f) for f in res['figures']}
            for name in ('fig1_stage1_meanings_senses_by_depth', 'fig2_stage2_frequency_meanings_senses_by_depth',
                         'fig3_depth_hypothesis_tests', 'fig4_meanings_vs_senses_separation'):
                self.assertIn(name + '.pdf', figs)
                base = os.path.join(c['dirs']['figures'], name)
                self.assertTrue(os.path.exists(base + '.png'), name)
                self.assertIn('NOT RESULTS', open(base + '_caption.txt', encoding='utf-8').read())
                pdf = open(base + '.pdf', 'rb').read()
                self.assertIn(b'/FontFile2', pdf)
                self.assertNotIn(b'/Type3', pdf)
                box = re.search(rb'/MediaBox \[ 0 0 ([\d.]+) ([\d.]+) \]', pdf)
                self.assertAlmostEqual(float(box.group(1)) / 72, FIG_W, places=3)
            # LaTeX tables: balanced environments; every body row has as many cells as the column spec
            self.assertEqual({os.path.basename(f) for f in res['latex_tables']},
                             {'table1_design_and_samples.tex', 'table2_confirmatory_evidence_matrix.tex',
                              'table3_stage2_effects.tex', 'table4_depth_slopes.tex',
                              'tableS1_confirmatory_tests_by_model.tex'})
            for f in res['latex_tables']:
                tex = open(f, encoding='utf-8').read()
                self.assertEqual(tex.count(r'\begin{table*}'), tex.count(r'\end{table*}'))
                self.assertEqual(tex.count('{'), tex.count('}'), f)
                for spec, body in re.findall(r'\\begin\{tabular\}\{([^}]*(?:\{[^}]*\}[^}]*)*)\}\n(.*?)\\end\{tabular\}\n',
                                             tex, re.S):
                    ncol = len(re.findall(r'[lcr]', re.sub(r'p\{[^}]*\}', 'l', spec)))
                    rows = body.split(r'\midrule')[-1].split(r'\bottomrule')[0].strip().split(' \\\\\n')
                    for row in rows:
                        self.assertEqual(len(re.findall(r'(?<!\\)&', row)), ncol - 1, (os.path.basename(f), row))
                    self.assertNotIn('>', re.sub(r'\$[^$]*\$', '', body))
            shared.update(latex_tables=res['latex_tables'], figures=res['figures'])

        def test_18b_publication_outputs_compile(self):
            if 'latex_tables' not in shared:
                self.skipTest('depends on test_18')
            import subprocess
            exe = shutil.which('pdflatex')
            if exe is None:
                self.skipTest('pdflatex not installed (LaTeX compile check)')
            d = tempfile.mkdtemp(dir=tmp_root)
            body = [f'\\input{{{f}}}' for f in shared['latex_tables']] + [
                f'\\begin{{figure*}}[t]\\includegraphics[width=\\textwidth]{{{f}}}\\caption{{x}}\\end{{figure*}}'
                for f in shared['figures']]
            with open(os.path.join(d, 'doc.tex'), 'w', encoding='utf-8') as fh:
                fh.write('\n'.join([r'\documentclass[11pt,twocolumn]{article}',
                                    r'\usepackage[a4paper,textwidth=16cm,textheight=24.7cm,columnsep=0.6cm]{geometry}',
                                    r'\usepackage[T1]{fontenc}', r'\usepackage{booktabs}', r'\usepackage{graphicx}',
                                    r'\begin{document}'] + body + ['Text.', r'\end{document}']))
            run = subprocess.run([exe, '-interaction=nonstopmode', '-halt-on-error', 'doc.tex'], cwd=d,
                                 capture_output=True, text=True, timeout=300)
            log = open(os.path.join(d, 'doc.log'), encoding='latin-1').read()
            self.assertEqual(run.returncode, 0, log[-3000:])
            self.assertNotIn('Overfull \\hbox', log)            # ACL text width (16 cm) is respected

        # ── statistics ─────────────────────────────────────────────────────
        def test_19_regression_uncertainty_and_multiplicity(self):
            r = np.random.default_rng(1)
            X = np.column_stack([np.ones(200), r.normal(size=(200, 3))])
            Y = X @ r.normal(size=(4, 2)) + r.normal(size=(200, 2)) * (1 + np.abs(X[:, [1]]))
            g = np.repeat(np.arange(40), 5)
            p = np.array([0.01, 0.04, 0.03, np.nan, 0.5, 0.001])
            if sm_api is None:
                self.skipTest('statsmodels not installed (reference implementation)')
            f, fc = ols_multi(X, Y), ols_cluster(X, Y, g)
            for l in range(2):
                ref = sm_api.OLS(Y[:, l], X).fit()
                np.testing.assert_allclose(f['beta'][:, l], ref.params, rtol=1e-8)
                np.testing.assert_allclose(f['se'][:, l], ref.bse, rtol=1e-8)
                np.testing.assert_allclose(f['se_hc3'][:, l], ref.get_robustcov_results('HC3').bse, rtol=1e-6)
                cl = sm_api.OLS(Y[:, l], X).fit(cov_type='cluster', cov_kwds={'groups': g})
                np.testing.assert_allclose(fc['se_cluster'][:, l], cl.bse, rtol=1e-6)
            ok = np.isfinite(p)
            np.testing.assert_allclose(bh_fdr(p)[ok], multipletests(p[ok], method='fdr_bh')[1])
            np.testing.assert_allclose(holm(p)[ok], multipletests(p[ok], method='holm')[1])
            self.assertTrue(np.isnan(holm(p)[3]))
            self.assertLess(p_tost_normal(0.01, 0.05, 0.2), 0.05)
            self.assertGreater(p_tost_normal(0.15, 0.1, 0.2), 0.05)
            self.assertAlmostEqual(p_normal(1.959964, 1.0), 0.05, places=5)
            Z = r.normal(size=(300, 2))
            x = Z[:, 0] + r.normal(size=300)
            y = Z[:, 0] + 0.5 * x + r.normal(size=300)
            pr, pp, n = partial_corr(x, y, Z)
            A = np.column_stack([np.ones(300), Z])
            res = lambda v: v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
            self.assertAlmostEqual(pr, np.corrcoef(res(x), res(y))[0, 1], places=10)
            self.assertEqual(n, 300)
            self.assertLess(pp, 1e-6)
            v = vif_table(r.normal(size=(500, 3)), ['a', 'b', 'c'])
            self.assertTrue(all(0.98 < x_ < 1.05 for x_ in v.values()))

        def test_20_hypothesis_classification_and_pattern(self):
            cfg = _test_cfg()
            rng = np.random.default_rng(3)
            n, L = 128, 9
            r = pd.DataFrame({'word': [f'w{i}' for i in range(n)], 'row_id': [f'r{i}' for i in range(n)],
                              'appendix': 'B', 'ambiguity': ['ambiguous', 'unambiguous'] * (n // 2),
                              'senses': (['many'] * 2 + ['few'] * 2) * (n // 4)})
            d = design_stage1_BC(r, 'B')
            senses = np.where(r['senses'] == 'many', -1.0, 1.0)[:, None]
            Y = senses * np.linspace(0.5, 1.5, L) + rng.normal(0, 0.5, (n, L))
            depth = np.arange(1, L + 1) / L
            h = evaluate_hypotheses({'stage1_B': d}, {'stage1_B': std_betas(d, Y)},
                                    {'stage1_B': boot_std_betas(d, [Y], 300, 0)[0]}, depth,
                                    {'model': 'm', 'condition': 'c', 'aggregate': 'sum', 'role': 'confirmatory'}, cfg)
            get = lambda hid, s: h[(h['hypothesis'] == hid) & (h['summary'] == s)].iloc[0]
            self.assertEqual(get('S1B_senses', 'deep')['evidence'], 'supported')
            self.assertEqual(get('S1B_separation', 'deep')['evidence'], 'supported')
            self.assertNotEqual(get('S1B_meanings_null', 'deep')['evidence'], 'contradicted (non-zero effect)')
            pat = meanings_senses_pattern(h, cfg.alpha)
            self.assertEqual(pat[pat['summary'] == 'deep']['pattern'].iloc[0],
                             'distinct effects in the predicted directions')
            Y2 = (np.where(r['ambiguity'] == 'ambiguous', -1.0, 1.0)[:, None] + senses) * np.ones(L) + \
                rng.normal(0, 0.5, (n, L))
            h2 = evaluate_hypotheses({'stage1_B': d}, {'stage1_B': std_betas(d, Y2)},
                                     {'stage1_B': boot_std_betas(d, [Y2], 300, 0)[0]}, depth,
                                     {'model': 'm', 'condition': 'c', 'aggregate': 'sum', 'role': 'confirmatory'}, cfg)
            self.assertEqual(h2[(h2['hypothesis'] == 'S1B_meanings_null') & (h2['summary'] == 'output')]
                             ['evidence'].iloc[0], 'contradicted (non-zero effect)')
            self.assertEqual(meanings_senses_pattern(h2, cfg.alpha).query("summary == 'output'")['pattern'].iloc[0],
                             'one shared effect (meanings and senses not distinguished)')

        def test_21_depth_slope_tests(self):
            depth = np.arange(1, 13) / 12
            d = Design('stage2', 'primary', 'confirmatory', np.array([]), ['log_subtlex', 'gcide_etymologies'],
                       pd.DataFrame(), {}, None, {}, pd.DataFrame())
            beta = np.vstack([-np.exp(-4 * depth), depth ** 2])     # frequency shallow (negative), meanings deep
            sl = dict(hypothesis_values('slope', ('log_subtlex',), d, beta, depth))
            self.assertLess(list(sl.values())[0], 0)                 # oriented frequency effect shrinks with depth
            sm_ = dict(hypothesis_values('slope', ('gcide_etymologies',), d, beta, depth))
            self.assertGreater(list(sm_.values())[0], 0)
            sc = dict(hypothesis_values('slope_contrast', ('gcide_etymologies', 'log_subtlex'), d, beta, depth))
            self.assertGreater(list(sc.values())[0], 0)
            self.assertEqual(sum(region_mask(depth, k).sum() for k in REGIONS), 12)
            self.assertAlmostEqual(depth_slope(2 * depth + 1, depth), 2.0)

        def test_22_probe_ridge_and_auroc(self):
            r = np.random.default_rng(3)
            X = r.normal(size=(120, 30)) * r.uniform(0.5, 3, 30)
            y = X[:, 0] - X[:, 1] + r.normal(size=120)
            folds = np.arange(120) % 4
            pred = ridge_oof(X, y, folds, 0.5)[:, 0]
            self.assertGreater(r2_columns(pred, y)[0], 0.3)
            if Ridge is None:
                self.skipTest('scikit-learn not installed (reference implementation)')
            for f in range(4):
                tr, te = folds != f, folds == f
                mu, sd = X[tr].mean(0), X[tr].std(0)
                Xs = (X[tr] - mu) / sd
                ref = Ridge(alpha=0.5 * np.trace(Xs @ Xs.T) / tr.sum()).fit(Xs, y[tr]).predict((X[te] - mu) / sd)
                np.testing.assert_allclose(pred[te], ref, rtol=1e-6, atol=1e-8)
            lab = (y > 0).astype(int)
            self.assertAlmostEqual(auroc_columns(pred, lab)[0], roc_auc_score(lab, pred), places=10)

        def test_23_matching_balances_covariates(self):
            r = np.random.default_rng(4)
            n = 400
            rows = pd.DataFrame({'word': [f'w{i}' for i in range(n)], 'row_id': [f'id{i}' for i in range(n)],
                                 'gcide_etymologies': r.choice([1, 1, 1, 2, 3], n).astype(float),
                                 'wn_n_all': r.integers(1, 15, n).astype(float), 'log_subtlex': r.uniform(2, 4, n),
                                 'length': r.integers(4, 7, n).astype(float), 'concreteness': r.uniform(1.5, 5, n),
                                 'ortho_n': r.integers(0, 8, n).astype(float)})
            for name, v in build_matches(rows, 0.2, 10).items():
                if v['info']['status'] == 'RUN':
                    self.assertTrue((v['balance']['smd_after'].abs() < 0.2).all(), name)
                    self.assertFalse(set(v['pairs']['treated_word']) & set(v['pairs']['control_word']))

        def test_24_prompts_and_leak_guard(self):
            conds = build_conditions(_test_cfg())
            self.assertEqual(conds['ld_isolated'].template, SOURCE_LD_TEMPLATE)
            for c in conds.values():
                text, spans = render_items(c.template, ['bank', 'paper'])
                self.assertEqual([text[a:b] for a, b in spans], ['bank', 'paper'])
                self.assertIsNone(TEMPLATE_LEAK_RE.search(text))
            for bad in ('Rate how ambiguous this is.\n' + ITEM_SLOT, ITEM_SLOT + ' trailing'):
                with self.assertRaises(CriticalError):
                    check_template('x', bad, ITEM_SLOT)

        def test_25_homonymy_polysemy_distance_contrast(self):
            rng = np.random.default_rng(5)
            rows, reps = [], []
            for w in range(12):
                base = rng.normal(size=8)
                for m, s, shift in [('m1', 's1', 0.1), ('m1', 's2', 0.5), ('m2', 's3', 2.0)]:
                    rows.append({'word': f'w{w}', 'align_ok': True, 'meaning_id': m, 'sense_id': s,
                                 'n_pre_target_tokens': 3})
                    reps.append(np.tile(base + shift * rng.normal(size=8), (4, 1)))
            tg = pd.DataFrame(rows).assign(context_id=[f'c{i % 3}' for i in range(len(rows))],
                                           pre_target_mean_surprisal_bits=5.0)
            texts = [f'context {i} for {{TARGET}}' for i in range(len(rows))]          # word-specific contexts
            ss = ScoreSet('m', 'context_sentence', 'context_sentence', 'exploratory', 3, {}, tg, {},
                          {'last': np.array(reps, np.float16)})
            cdf = tg[['word', 'context_id']].assign(text=texts)
            hp, prof, bal = homonymy_polysemy_distance(ss, _test_cfg(n_bootstrap=200), {'model': 'm'}, cdf)
            key = [r for r in hp if r['contrast'] == 'different_meaning − different_sense_same_meaning']
            self.assertEqual(len(key), len(SUMMARIES))
            self.assertTrue(all(np.isfinite(r['estimate']) and r['estimate'] > 0 for r in key))
            self.assertTrue(prof and bal)
            self.assertEqual(key[0]['template_confounded_pair_share'], 0.0)
            shared = cdf.assign(text=[f'shared template {i % 3} {{TARGET}}' for i in range(len(rows))])
            hp2 = homonymy_polysemy_distance(ss, _test_cfg(n_bootstrap=50), {'model': 'm'}, shared)[0]
            self.assertEqual(hp2[0]['template_confounded_pair_share'], 1.0)    # templates ≡ relation types

    suite = unittest.TestLoader().loadTestsFromTestCase(T)
    res = unittest.TextTestRunner(verbosity=2 if verbose else 1, stream=sys.stdout).run(suite)
    n_skip, n_fail = len(res.skipped), len(res.failures) + len(res.errors)
    print(f"\nSELF-TEST SUMMARY: {res.testsRun - n_fail - n_skip} PASS, {n_fail} FAIL, {n_skip} SKIPPED"
          + (' (' + '; '.join(f"{t.id().split('.')[-1]}: {r}" for t, r in res.skipped) + ')' if n_skip else ''))
    shutil.rmtree(tmp_root, ignore_errors=True)
    return res.wasSuccessful()


if __name__ == '__main__':
    sys.exit(main())
