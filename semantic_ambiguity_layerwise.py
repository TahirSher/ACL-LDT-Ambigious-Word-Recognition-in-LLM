#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
semantic_ambiguity_layerwise.py
===============================
Semantic ambiguity (multiple unrelated MEANINGS vs multiple related SENSES) and
word frequency across the layers of decoder-only LLMs.

RESEARCH QUESTION
  Do semantic ambiguity and word frequency have different layerwise profiles in
  decoder-only LLMs; does the ambiguity profile separate multiple unrelated
  meanings (homonymy-like) from multiple related senses (polysemy-like); and are
  these profiles modulated by task instruction / linguistic context?

AUTHORITATIVE SOURCES (read before changing anything)
  ambiguity_procedure_spec.md   Rodd, Gaskell & Marslen-Wilson (2002) procedure,
                                two-stage design, cautions, data dictionary
  Lexical_Variable.pptx         depth hypothesis, measure choice, Route 2 prompt
                                manipulations, screening decisions
  ambiguity_items_combined.csv  stimulus table (2,035 rows, 35 columns)
  ldt_fresh.py                  reused ONLY for model inventory, loading,
                                logit-lens readout verification, layer
                                conventions and statistical helpers

DESIGN
  Stage 1  calibration — provenance == "published" (Rodd et al., 2002).
           Appendix A (Exp. 1, visual): regression, ambiguity + WordNet senses
           in one model; Appendices B (Exp. 2, visual) and C (Exp. 3, auditory):
           meanings x senses 2x2, kept separate. Appendix membership is kept.
  Stage 2  generalisation — provenance == "wordnet-derived", source-defined
           central range (length 4-6, log_subtlex 2.0-4.0, gcide_etymologies and
           wn_n_all present); per layer
               y_k ~ gcide_etymologies + wn_n_all + log_subtlex + length + concreteness
  The stages are never mixed in one regression.

PRIMARY MEASURE
  Layerwise surprisal of the target word through the logit lens: for every
  block output h_l, logits_l = W_U · norm_f(h_l) (the model's own final norm and
  unembedding; the last block's state is already post-norm in HF and is not
  normalised twice — verified at run time against the model's own logits).
  Target subtoken t is scored from position t-1. Multi-token words: SUM of the
  conditional subtoken surprisals (sequence surprisal, bits); MEAN per subtoken
  and an end-of-word-corrected sum are reported as companions.
  Sign convention: higher surprisal = the target is LESS expected by the model.
  Surprisal is a model-derived quantity, NOT a reaction time.

CLAIM BOUNDARY
  * No LLM measure here is a reaction time. Stage 1 compares the qualitative
    SIGN PATTERN of model-derived effects with the published human pattern
    under an explicitly stated stand-in mapping (higher surprisal ≙ cost).
  * The pool contains no nonword foils and none are generated, so no word /
    nonword decision is simulated; the lexical-decision prompt is used as a
    task frame for scoring the word.
  * Appendix C (auditory) items are presented to the model as text; the
    modality difference cannot be simulated.
  * Stage 2 is a generalisation analysis on a derived pool, never a replication.
  * Decodability by a probe is not use by the model.

INSTALL
  pip install "torch>=2.2" "transformers>=4.53" "tokenizers>=0.19" accelerate numpy pandas scipy matplotlib
  pip install statsmodels scikit-learn      # optional: only used by --self_test reference checks
  huggingface-cli login                     # Llama models are gated: accept the licence on the Hub first

USAGE
  python semantic_ambiguity_layerwise.py --self_test
  python semantic_ambiguity_layerwise.py --audit_only
  python semantic_ambiguity_layerwise.py --smoke_test --models SmolLM2-360M
  python semantic_ambiguity_layerwise.py --models SmolLM2-360M Qwen2.5-1.5B --dtype bf16
  python semantic_ambiguity_layerwise.py --resume --models Llama-3.2-1B     # add a model
  python semantic_ambiguity_layerwise.py --analysis_only --resume           # re-run analyses
  python semantic_ambiguity_layerwise.py --list_models
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_NAME = 'semantic_ambiguity_layerwise.py'
SCRIPT_VERSION = '1.0.0'
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
        import torch  # noqa: F401
        import transformers  # noqa: F401
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


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(str(s).encode('utf-8'))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
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
    atomic_write_text(path, json.dumps(_clean_json(obj), indent=2, sort_keys=False,
                                       default=_json_default))


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
    try:
        torch = _torch()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:                                                # noqa: BLE001
        pass


class CriticalError(RuntimeError):
    """A failure that invalidates the run (non-zero exit)."""


class StaleCacheError(RuntimeError):
    """A cache exists whose manifest does not match the current configuration."""


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
    variant: str                 # 'base' | 'instruct'
    base_of: str | None = None   # instruct variants: alias of the base counterpart
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
# Offline tiny RANDOM-WEIGHT models: they exist only to exercise the pipeline
# end to end without network access. Their outputs are never results.
TEST_MODEL_REGISTRY: list[ModelSpec] = [
    ModelSpec('tiny-llama-TEST', 'local/tiny-llama-TEST', 'test', 'base', test_model='llama'),
    ModelSpec('tiny-qwen2-TEST', 'local/tiny-qwen2-TEST', 'test', 'base', test_model='qwen2'),
    ModelSpec('tiny-llama-chat-TEST', 'local/tiny-llama-chat-TEST', 'test', 'instruct',
              'tiny-llama-TEST', test_model='llama_chat'),
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


def zscore_cols(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, float)
    mu = A.mean(0)
    sd = A.std(0, ddof=1) if A.shape[0] > 1 else np.ones(A.shape[1])
    sd = np.where(sd > 0, sd, np.nan)
    return (A - mu) / sd


def ols_multi(X: np.ndarray, Y: np.ndarray) -> dict:
    """OLS of every column of Y (n x L) on the same design X (n x p, intercept
    included). Classical and HC3 (MacKinnon & White, 1985) standard errors.
    Returns arrays of shape (p, L) plus df, R² (L,), rank."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    n, p = X.shape
    rank = int(np.linalg.matrix_rank(X))
    L = Y.shape[1]
    nan = np.full((p, L), np.nan)
    res = {'beta': nan.copy(), 'se': nan.copy(), 'se_hc3': nan.copy(), 'df': n - p,
           'r2': np.full(L, np.nan), 'n': n, 'rank': rank, 'p_design': p}
    if rank < p or n - p < 1:
        return res
    XtX_inv = np.linalg.inv(X.T @ X)
    B = XtX_inv @ (X.T @ Y)
    E = Y - X @ B
    df = n - p
    rss = (E ** 2).sum(0)
    sigma2 = rss / df
    se = np.sqrt(np.outer(np.diag(XtX_inv), sigma2))
    A = X @ XtX_inv
    h = (A * X).sum(1)
    w = E ** 2 / np.clip(1.0 - h, 1e-12, None)[:, None] ** 2
    se_hc3 = np.sqrt((A ** 2).T @ w)
    tss = ((Y - Y.mean(0)) ** 2).sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        r2 = 1.0 - rss / tss
    res.update({'beta': B, 'se': se, 'se_hc3': se_hc3, 'df': df, 'r2': r2})
    return res


def ols_beta_only(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, Y, rcond=None)[0]


def vif_table(X: np.ndarray, names: list[str]) -> dict:
    """Variance inflation factors of the predictor columns (no intercept in X)."""
    X = np.asarray(X, float)
    out = {}
    for j, nm in enumerate(names):
        y = X[:, j]
        others = np.delete(X, j, axis=1)
        A = np.column_stack([np.ones(len(y)), others])
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
        r = y - A @ beta
        tss = ((y - y.mean()) ** 2).sum()
        r2 = 1 - (r @ r) / tss if tss > 0 else np.nan
        out[nm] = float(1.0 / (1.0 - r2)) if np.isfinite(r2) and r2 < 1 else float('inf')
    return out


def bootstrap_p_two_sided(boot) -> float:
    """Two-sided bootstrap p for H0: θ = 0 (percentile inversion, +1 correction
    of Davison & Hinkley, 1997, so p is never 0). Reused from ldt_fresh.py."""
    b = np.asarray(boot, float)
    b = b[np.isfinite(b)]
    if len(b) == 0:
        return np.nan
    k = min(np.sum(b <= 0), np.sum(b >= 0))
    return float(min(1.0, 2 * (1 + k) / (1 + len(b))))


def percentile_ci(boot, alpha=0.05, axis=0):
    b = np.asarray(boot, float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return (np.nanpercentile(b, 100 * alpha / 2, axis=axis),
                np.nanpercentile(b, 100 * (1 - alpha / 2), axis=axis))


def auroc_columns(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUROC (Mann–Whitney, average ranks for ties) of each column of `scores`
    for binary labels y (1 = positive). Labels may also be a matrix of the
    same shape (one label vector per column, used for permutation nulls)."""
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
    sse = ((Yv - P) ** 2).sum(0)
    sst = ((Yv - Yv.mean(0)) ** 2).sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return 1.0 - sse / sst


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
    caliper |Δx_j| ≤ caliper·SD_j (Austin, 2011). Treated units are processed in
    increasing order of their nearest admissible distance; units without an
    admissible control are rejected. Adapted from ldt_fresh.py (numpy only)."""
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
    """Fold index per row such that all rows of a group (word) share a fold
    (no word can be in train and test). With binary labels the groups are
    dealt round-robin within class (stratified). Labels must be constant
    within a group."""
    groups = np.asarray(groups).astype(str)
    uniq = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    fold_of = {}
    if labels is not None:
        lab = pd.Series(np.asarray(labels)).groupby(groups).agg(lambda s: s.nunique())
        if (lab > 1).any():
            raise ValueError("labels differ within a group")
        glabel = pd.Series(np.asarray(labels)).groupby(groups).first()
        for cls in sorted(glabel.unique(), key=str):
            g = np.array(sorted(glabel.index[glabel == cls]))
            g = g[rng.permutation(len(g))]
            for i, gg in enumerate(g):
                fold_of[gg] = i % k
    else:
        g = uniq[rng.permutation(len(uniq))]
        for i, gg in enumerate(g):
            fold_of[gg] = i % k
    return np.array([fold_of[g] for g in groups], int)


def ridge_oof(X: np.ndarray, Ycols: np.ndarray, folds: np.ndarray, lam_scale: float) -> np.ndarray:
    """Out-of-fold predictions of a linear ridge probe with unpenalised
    intercept, computed in the dual (kernel) form so that many label vectors
    (observed + permutations) share one fit per fold. Features are
    standardised inside each training fold; λ = lam_scale · mean(diag K_train),
    fixed a priori (never tuned on test folds)."""
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
        Xte = (X[te] - mu) / sd
        K = Xtr @ Xtr.T
        lam = lam_scale * float(np.trace(K)) / max(1, K.shape[0])
        lam = max(lam, 1e-8)
        ym = Ycols[tr].mean(0)
        A = np.linalg.solve(K + lam * np.eye(K.shape[0]), Ycols[tr] - ym)
        pred[te] = (Xte @ Xtr.T) @ A + ym
    return pred


# ════════════════════════════════════════════════════════════════════════════
# DATA: schema, validation, stages, audit
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
    """Read the stimulus CSV, validate schema and design labels, add a stable
    row identifier. Returns (df, validation_report). Raises CriticalError on
    schema violations that would invalidate the stage definitions."""
    if not os.path.exists(path):
        raise CriticalError(f"data CSV not found: {path}")
    raw = pd.read_csv(path, dtype={c: str for c in TEXT_COLUMNS}, keep_default_na=True)
    rep = {'path': os.path.abspath(path), 'sha256': sha256_file(path), 'n_rows': int(len(raw)),
           'n_columns': int(raw.shape[1]), 'checks': {}}
    missing = [c for c in EXPECTED_COLUMNS if c not in raw.columns]
    extra = [c for c in raw.columns if c not in EXPECTED_COLUMNS]
    rep['columns_missing'], rep['columns_extra'] = missing, extra
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

    def check(name, ok, detail=''):
        ck[name] = {'pass': bool(ok), 'detail': detail}
        return ok

    critical = []
    if not check('word_lowercase_alpha', bool(df['word'].map(lambda v: isinstance(v, str) and
                                                            re.fullmatch(r'[a-z]+', v) is not None).all()),
                 'every word matches ^[a-z]+$'):
        critical.append('word_lowercase_alpha')
    prov_vals = set(df['provenance'].dropna().unique())
    if not check('provenance_values', prov_vals <= {'published', 'wordnet-derived'} and
                 df['provenance'].notna().all(), f"values={sorted(prov_vals)}"):
        critical.append('provenance_values')
    pub = df['provenance'] == 'published'
    der = df['provenance'] == 'wordnet-derived'
    ok_app = df.loc[pub, 'appendix'].isin(list(APPENDIX_DESIGN)).all()
    exp_ok = all(((df.loc[pub & (df['appendix'] == a), 'experiment'] == e).all() and
                  (df.loc[pub & (df['appendix'] == a), 'modality'] == m).all())
                 for a, (e, m) in APPENDIX_DESIGN.items())
    if not check('published_appendix_experiment_modality', bool(ok_app and exp_ok),
                 'A→Exp1 visual, B→Exp2 visual, C→Exp3 auditory'):
        critical.append('published_appendix_experiment_modality')
    if not check('published_ambiguity_values',
                 df.loc[pub, 'ambiguity'].isin(['ambiguous', 'unambiguous']).all()):
        critical.append('published_ambiguity_values')
    sen_bc = df.loc[pub & df['appendix'].isin(['B', 'C']), 'senses'].isin(['few', 'many']).all()
    sen_a = df.loc[pub & (df['appendix'] == 'A'), 'senses'].isna().all()
    if not check('published_senses_values', bool(sen_bc and sen_a),
                 'few/many in B and C; empty in A (no sense manipulation)'):
        critical.append('published_senses_values')
    der_design_empty = df.loc[der, ['appendix', 'experiment', 'modality', 'ambiguity', 'senses']].isna().all().all()
    if not check('derived_design_columns_empty', bool(der_design_empty)):
        critical.append('derived_design_columns_empty')
    check('derived_pool_condition_values',
          df.loc[der, 'pool_condition'].isin(['homonym-candidate', 'polyseme-candidate']).all() and
          df.loc[pub, 'pool_condition'].isna().all())
    check('length_equals_word_length', bool((df['word'].str.len() == df['length']).all()))
    d = (np.log10(df['subtlex'] + 1) - df['log_subtlex']).abs()
    max_dev = float(np.nanmax(d.values)) if d.notna().any() else np.nan
    if not check('log_subtlex_is_log10_subtlex_plus_1', bool(np.isfinite(max_dev) and max_dev < 0.0015),
                 f"max |log10(subtlex+1) − log_subtlex| = {max_dev:.5f}; log_subtlex is used AS IS "
                 f"and never transformed again"):
        critical.append('log_subtlex_is_log10_subtlex_plus_1')
    df['row_id'] = df['provenance'].astype(str) + '|' + df['source_set'].astype(str) + '|' + df['word'].astype(str)
    dup = int(df['row_id'].duplicated().sum())
    if not check('row_id_unique', dup == 0, f"{dup} duplicated provenance|source_set|word keys"):
        critical.append('row_id_unique')
    overlap = set(df.loc[pub, 'word']) & set(df.loc[der, 'word'])
    check('published_derived_words_disjoint', len(overlap) == 0, f"{len(overlap)} shared words")
    rep['critical_failures'] = critical
    if critical:
        raise CriticalError(f"schema/design validation failed: {critical}")
    return df, rep


def apply_gcide_policy(df: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, dict]:
    """gcide_etymologies == 0 means 'in GCIDE, no etymology recorded' (not zero
    meanings). The procedure says: drop or recode to 1, and say which. A blank
    (not in GCIDE) is never imputed."""
    if policy not in GCIDE_ZERO_POLICIES:
        raise ValueError(f"gcide_zero_policy must be one of {GCIDE_ZERO_POLICIES}")
    out = df.copy()
    zero = out['gcide_etymologies'] == 0
    info = {'policy': policy, 'n_zero': int(zero.sum()), 'n_in': int(len(df))}
    if policy == 'drop':
        out = out[~zero]
    elif policy == 'recode_to_one':
        out.loc[zero, 'gcide_etymologies'] = 1.0
    info['n_out'] = int(len(out))
    return out, info


def load_screening_resources(cfg) -> dict:
    """Optional, user-supplied screening resources. Without them the screens
    are NOT_AVAILABLE: the CSV has no morphological or valence field and none
    is invented here."""
    res = {'inflected': {'status': 'NOT_AVAILABLE',
                         'reason': 'no morphological/lemma field in the CSV and no --screen_inflected_file supplied',
                         'excluded_stage2': 0, 'flagged_stage1': 0, 'words': set()},
           'emotional_extremes': {'status': 'NOT_AVAILABLE',
                                  'reason': 'no valence field in the CSV and no --screen_valence_file supplied',
                                  'excluded_stage2': 0, 'flagged_stage1': 0, 'words': set()}}
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
        z = (v - v.mean()) / v.std(ddof=1)
        flag = z.abs() > cfg.valence_extreme_sd
        res['emotional_extremes'] = {
            'status': 'APPLIED', 'source': os.path.abspath(cfg.screen_valence_file),
            'sha256': sha256_file(cfg.screen_valence_file),
            'rule': (f"exclude |z(valence)| > {cfg.valence_extreme_sd} (z over the supplied file; threshold "
                     f"is a CONFIGURATION choice, not prescribed by the source) in Stage 2; flag only in Stage 1"),
            'words': set(t.loc[flag, 'word'].astype(str).str.lower())}
    return res


@dataclass
class Samples:
    df: pd.DataFrame                 # validated full table
    stage1: pd.DataFrame             # published rows (appendix membership preserved)
    stage2_base: pd.DataFrame        # derived rows in the central range, BEFORE the zero policy
    stage2: dict                     # policy -> rows after zero policy
    funnel: list
    words_to_score: pd.DataFrame     # unique words scored by every model
    subset: bool
    screening: dict


def build_samples(df: pd.DataFrame, cfg, screening: dict) -> Samples:
    funnel = []
    pub = df[df['provenance'] == 'published'].copy()
    der = df[df['provenance'] == 'wordnet-derived'].copy()
    funnel.append({'stage': 'stage1', 'step': 'provenance == "published"', 'rows': len(pub),
                   'unique_words': int(pub['word'].nunique())})
    for a in 'ABC':
        funnel.append({'stage': 'stage1', 'step': f'appendix == {a}', 'rows': int((pub['appendix'] == a).sum()),
                       'unique_words': int(pub.loc[pub['appendix'] == a, 'word'].nunique())})
    s = der
    funnel.append({'stage': 'stage2', 'step': 'provenance == "wordnet-derived"', 'rows': len(s),
                   'unique_words': int(s['word'].nunique())})
    s = s[s['gcide_etymologies'].notna()]
    funnel.append({'stage': 'stage2', 'step': 'gcide_etymologies non-missing', 'rows': len(s),
                   'unique_words': int(s['word'].nunique())})
    s = s[s['wn_n_all'].notna()]
    funnel.append({'stage': 'stage2', 'step': 'wn_n_all non-missing', 'rows': len(s),
                   'unique_words': int(s['word'].nunique())})
    s = s[s['length'].between(*STAGE2_RANGE['length'])]
    funnel.append({'stage': 'stage2', 'step': 'length 4–6', 'rows': len(s), 'unique_words': int(s['word'].nunique())})
    s = s[s['log_subtlex'].between(*STAGE2_RANGE['log_subtlex'])]
    funnel.append({'stage': 'stage2', 'step': 'log_subtlex 2.0–4.0 (source-defined central range)',
                   'rows': len(s), 'unique_words': int(s['word'].nunique())})
    for key in ('inflected', 'emotional_extremes'):
        sc = screening[key]
        if sc['status'] == 'APPLIED':
            n0 = len(s)
            s = s[~s['word'].isin(sc['words'])]
            sc['excluded_stage2'] = n0 - len(s)
            sc['flagged_stage1'] = int(pub['word'].isin(sc['words']).sum())
            pub[f'flag_{key}'] = pub['word'].isin(sc['words'])
            funnel.append({'stage': 'stage2', 'step': f'screen {key} (applied)', 'rows': len(s),
                           'unique_words': int(s['word'].nunique())})
        else:
            funnel.append({'stage': 'stage2', 'step': f'screen {key}: NOT_AVAILABLE (0 excluded)',
                           'rows': len(s), 'unique_words': int(s['word'].nunique())})
    stage2_base = s.copy()
    subset = False
    if cfg.max_items:
        subset = True
        rng = np.random.default_rng(cfg.seed)
        w1 = np.array(sorted(pub['word'].unique()))
        w2 = np.array(sorted(stage2_base['word'].unique()))
        k1 = max(1, cfg.max_items // 2)
        k2 = max(1, cfg.max_items - k1)
        # always keep the shortest and longest words for tokenisation diagnostics
        def pick(words, k):
            if len(words) <= k:
                return set(words)
            by_len = sorted(words, key=lambda w: (len(w), w))
            keep = {by_len[0], by_len[-1]}
            rest = [w for w in words[rng.permutation(len(words))] if w not in keep]
            return keep | set(rest[:max(0, k - len(keep))])
        keep1, keep2 = pick(w1, k1), pick(w2, k2)
        pub = pub[pub['word'].isin(keep1)]
        stage2_base = stage2_base[stage2_base['word'].isin(keep2)]
        funnel.append({'stage': 'both', 'step': f'--max_items {cfg.max_items} SUBSET (debug only)',
                       'rows': len(pub) + len(stage2_base),
                       'unique_words': int(pub['word'].nunique() + stage2_base['word'].nunique())})
    stage2 = {}
    for pol in GCIDE_ZERO_POLICIES:
        rows, info = apply_gcide_policy(stage2_base, pol)
        stage2[pol] = rows
        funnel.append({'stage': 'stage2', 'step': f'gcide_etymologies == 0 policy = {pol}'
                       + (' (PRIMARY, configured)' if pol == cfg.gcide_zero_policy else ' (sensitivity)'),
                       'rows': len(rows), 'unique_words': int(rows['word'].nunique())})
    words = pd.DataFrame({'word': sorted(set(pub['word']) | set(stage2_base['word']))})
    words['in_stage1'] = words['word'].isin(set(pub['word']))
    words['in_stage2'] = words['word'].isin(set(stage2_base['word']))
    return Samples(df=df, stage1=pub, stage2_base=stage2_base, stage2=stage2, funnel=funnel,
                   words_to_score=words, subset=subset, screening=screening)


def _vc(s: pd.Series) -> dict:
    return {('<NA>' if pd.isna(k) else str(k)): int(v) for k, v in s.value_counts(dropna=False).items()}


def data_audit(df: pd.DataFrame, rep: dict, samples: Samples, cfg) -> dict:
    pub = df[df['provenance'] == 'published']
    der = df[df['provenance'] == 'wordnet-derived']
    audit = {'file': {k: rep[k] for k in ('path', 'sha256', 'n_rows', 'n_columns')},
             'schema': {'columns_missing': rep['columns_missing'], 'columns_extra': rep['columns_extra'],
                        'checks': rep['checks']}}
    audit['row_counts'] = {c: _vc(df[c]) for c in ('provenance', 'source_set', 'appendix', 'experiment',
                                                   'modality', 'ambiguity', 'senses', 'pool_condition',
                                                   'ambiguity_class')}
    cells = (pub.assign(senses=pub['senses'].fillna('NA'))
             .groupby(['appendix', 'ambiguity', 'senses']).size().reset_index(name='rows'))
    audit['stage1_cells'] = cells.to_dict('records')
    audit['unique_words'] = {
        'overall': int(df['word'].nunique()),
        'by_provenance': {k: int(v) for k, v in df.groupby('provenance')['word'].nunique().items()},
        'by_source_set': {k: int(v) for k, v in df.groupby('source_set')['word'].nunique().items()}}
    miss = {}
    for c in KEY_PREDICTORS:
        miss[c] = {'all_rows': int(df[c].isna().sum()), 'published': int(pub[c].isna().sum()),
                   'derived': int(der[c].isna().sum()),
                   'stage2_pool_before_zero_policy': int(samples.stage2_base[c].isna().sum()),
                   'stage2_pool_n': int(len(samples.stage2_base))}
    audit['missingness'] = miss
    membership = pub.groupby('word')['appendix'].agg(lambda s: ''.join(sorted(s)))
    audit['duplicates'] = {
        'row_id_duplicates': int(df['row_id'].duplicated().sum()),
        'stable_row_id': 'provenance|source_set|word',
        'published_words_in_multiple_appendices': int((membership.str.len() > 1).sum()),
        'published_appendix_membership_patterns': {k: int(v) for k, v in membership.value_counts().items()},
        'derived_duplicate_words': int(der['word'].duplicated().sum()),
        'note': ('Repeated published words are kept once per appendix; analyses are run per appendix, '
                 'and supervised probes deduplicate by word and use word-grouped folds.')}
    z_all = df['gcide_etymologies'] == 0
    z_pool = samples.stage2_base['gcide_etymologies'] == 0
    audit['gcide_etymologies_zero'] = {
        'meaning': 'word present in GCIDE but no entry carries an <ety> field; NOT zero meanings',
        'rows_whole_file': int(z_all.sum()), 'published_rows': int((z_all & (df['provenance'] == 'published')).sum()),
        'derived_rows': int((z_all & (df['provenance'] == 'wordnet-derived')).sum()),
        'stage2_pool_rows': int(z_pool.sum()),
        'stage2_pool_zero_words': sorted(samples.stage2_base.loc[z_pool, 'word'].tolist()),
        'primary_policy': cfg.gcide_zero_policy,
        'policy_status': ('UNRESOLVED IN SOURCE: the procedure says "drop or recode to 1, and say which" '
                          'without choosing. The primary policy is a CONFIGURATION choice of this run; '
                          'results under every policy are reported as sensitivity analyses.'),
        'n_stage2_by_policy': {p: int(len(r)) for p, r in samples.stage2.items()}}
    audit['screening'] = {k: {kk: vv for kk, vv in v.items() if kk != 'words'}
                          for k, v in samples.screening.items()}
    audit['stage_counts'] = {'funnel': samples.funnel, 'subset_debug_run': samples.subset,
                             'words_to_score': int(len(samples.words_to_score)),
                             'stage1_and_stage2_disjoint': bool(not (set(samples.stage1['word']) &
                                                                    set(samples.stage2_base['word'])))}
    # ── recompute every number stated in the source documents ────────────────
    s2 = samples.stage2_base if not samples.subset else None
    full_der = der[der['gcide_etymologies'].notna() & der['wn_n_all'].notna()]
    in_range = der[der['length'].between(4, 6) & der['log_subtlex'].between(2.0, 4.0)]
    pool596 = in_range[in_range['gcide_etymologies'].notna() & in_range['wn_n_all'].notna()]
    gdist = pool596['gcide_etymologies'].value_counts()
    bc = pub[pub['appendix'].isin(['B', 'C'])]
    amb = (bc['ambiguity'] == 'ambiguous').astype(float)
    sen = (bc['senses'] == 'many').astype(float)

    def r(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else np.nan

    def coll(frame):
        return {'gcide_etymologies~wn_n_all': r(frame['gcide_etymologies'], frame['wn_n_all']),
                'gcide_etymologies~log_subtlex': r(frame['gcide_etymologies'], frame['log_subtlex']),
                'wn_n_all~log_subtlex': r(frame['wn_n_all'], frame['log_subtlex'])}
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
        'collinearity_stage2_pool': coll(pool596),
        'collinearity_all_derived_with_gcide': coll(full_der),
        'validation_BC': {'r_gcide_meanings': r(bc['gcide_etymologies'], amb),
                          'r_gcide_senses': r(bc['gcide_etymologies'], sen),
                          'r_wn_n_all_senses': r(bc['wn_n_all'], sen),
                          'r_wn_n_all_meanings': r(bc['wn_n_all'], amb)},
        'gcide_zero_rows_whole_file': int(z_all.sum()),
        'words_missing_from_gcide': int(df.loc[df['gcide_etymologies'].isna(), 'word'].nunique()),
        'article_exp1_regression_n': int((pub['appendix'] == 'A').sum()),
        'article_table2_unambiguous_n': int(((pub['appendix'] == 'A') & (pub['ambiguity'] == 'unambiguous')).sum()),
    }
    checks = []

    def cmp(key, stated, got, tol=0.0):
        if isinstance(stated, dict):
            for k in stated:
                cmp(f"{key}.{k}", stated[k], got.get(k) if isinstance(got, dict) else None, tol)
            return
        if isinstance(got, list):
            match = got == [stated]
        else:
            match = got is not None and np.isfinite(float(got)) and abs(float(got) - float(stated)) <= tol
        checks.append({'quantity': key, 'stated_in_source': stated, 'recomputed': got, 'match': bool(match)})
    for k, v in SOURCE_STATED.items():
        tol = 0.0105 if k in ('collinearity_stage2_pool',) else (0.0015 if k == 'validation_BC' else 0.0)
        cmp(k, v, recomputed.get(k), tol)
    notes = []
    cs = recomputed['collinearity_stage2_pool']
    ca = recomputed['collinearity_all_derived_with_gcide']
    if not all(abs(cs[k] - SOURCE_STATED['collinearity_stage2_pool'][k]) <= 0.0105 for k in cs):
        notes.append(
            "The collinearity values stated for 'that pool' (.33/.22/.45) do NOT reproduce on the Stage 2 pool "
            f"(recomputed {cs['gcide_etymologies~wn_n_all']:.2f}/{cs['gcide_etymologies~log_subtlex']:.2f}/"
            f"{cs['wn_n_all~log_subtlex']:.2f}); they reproduce on ALL derived words with a GCIDE value "
            f"({ca['gcide_etymologies~wn_n_all']:.2f}/{ca['gcide_etymologies~log_subtlex']:.2f}/"
            f"{ca['wn_n_all~log_subtlex']:.2f}, n={len(full_der)}). In-pool values and VIFs are reported here.")
    if recomputed['stage2_gcide_distribution']['0'] > 0:
        notes.append(f"The stated gcide distribution of the 596-word pool (1/2/3/4+ = 359/129/34/20 = 542) "
                     f"omits the {recomputed['stage2_gcide_distribution']['0']} words with gcide_etymologies == 0 "
                     f"that the 596 count includes.")
    notes.append("Appendix A prints 182 items (124 ambiguous / 58 unambiguous); the article reports n = 184 for "
                 "the Exp. 1 regression and N = 60 unambiguous in its Table 2. The discrepancy is in the original; "
                 "the CSV's 182 rows are used.")
    sim_missing_words = int(df.loc[df['sim_top2'].isna(), 'word'].nunique())
    notes.append(f"sim_top2 is missing for {int(df['sim_top2'].isna().sum())} rows / {sim_missing_words} unique "
                 f"words (the data dictionary says 63 words).")
    audit['source_cross_checks'] = {'comparisons': checks, 'notes': notes,
                                    'recomputed_on_subset': bool(samples.subset)}
    # ── Stage 2 predictor collinearity on the analysed pool (model-independent) ─
    vifs = {}
    for pol, rows in samples.stage2.items():
        cols = ['gcide_etymologies', 'wn_n_all', 'log_subtlex', 'length', 'concreteness']
        cc = rows[cols].dropna()
        if len(cc) > len(cols) + 2:
            vifs[pol] = {'n': int(len(cc)), 'vif': vif_table(cc.values, cols),
                         'correlations': cc.corr().round(3).to_dict()}
    audit['stage2_collinearity'] = vifs
    audit['notes'] = {
        'wn_n_noun_zero': int((df['wn_n_noun'] == 0).sum()),
        'wn_n_noun_zero_stage2_pool': int((samples.stage2_base['wn_n_noun'] == 0).sum()),
        'ambiguity_class': 'derived candidate label, not human-validated; not used as a factorial condition',
        'pool_condition': 'source label for derived rows only; not a published condition',
        'modality': 'Appendix C (auditory) items are given to the LLM as text; modality is not simulated',
        'nonword_foils': 'NOT_AVAILABLE in the pool and not generated (the spec forbids inventing pseudowords)',
        'dominance': 'NOT_AVAILABLE (eDom / Twilley / Gilbert & Rodd norms not supplied)'}
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
# AUTHORED HERE (not source wording): the slides ask for a "relatedness judgment
# instruction" but give no wording and no word pairs. This single-word version
# follows Rodd et al.'s pretest (7-point relatedness of a word's meanings) and
# keeps the source prompt's structure, so the text immediately before the
# target ("Items:\n") is identical and only the task instruction differs.
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

# Item-specific labels that must never reach a model input.
LEAK_RE = re.compile(r'\b(ambiguous|unambiguous|homonym\w*|polysem\w*|polyseme\w*|etymolog\w*|wordnet|'
                     r'synsets?|subtlex|gcide|pool_condition|ambiguity_class|few senses|many senses)\b', re.I)
TEMPLATE_LEAK_RE = re.compile(LEAK_RE.pattern[:-3] + r'|frequency|frequent)\b', re.I)


@dataclass
class PromptCondition:
    name: str
    template: str
    protocol: str            # 'isolated' | 'list' | 'context'
    role: str                # 'primary' | 'task_manipulation' | 'secondary_list' | 'context'
    wording_source: str
    status: str              # 'confirmatory' | 'exploratory' | 'secondary'

    @property
    def template_sha256(self) -> str:
        return sha256_text(self.template)


def check_template(name: str, tpl: str):
    if tpl.count(ITEM_SLOT) != 1:
        raise CriticalError(f"prompt condition {name!r}: template must contain {ITEM_SLOT} exactly once")
    if not tpl.endswith(ITEM_SLOT):
        raise CriticalError(f"prompt condition {name!r}: {ITEM_SLOT} must end the template so that the "
                            f"item list is the last text of the input")
    m = TEMPLATE_LEAK_RE.search(tpl.replace(ITEM_SLOT, ''))
    if m:
        raise CriticalError(f"prompt condition {name!r} contains an analysis label ({m.group(0)!r}); "
                            f"labels and covariates must never enter model inputs")


def build_conditions(cfg) -> dict[str, PromptCondition]:
    conds = {
        'ld_isolated': PromptCondition(
            'ld_isolated', SOURCE_LD_TEMPLATE, 'isolated', 'primary',
            'ambiguity_procedure_spec.md, LLM prompt template (verbatim); one item per forward pass',
            'confirmatory'),
        'relatedness_isolated': PromptCondition(
            'relatedness_isolated', AUTHORED_RELATEDNESS_TEMPLATE, 'isolated', 'task_manipulation',
            'AUTHORED HERE: slides request a relatedness-judgement instruction without wording; '
            'modelled on Rodd et al. (2002) pretest relatedness scale', 'exploratory'),
        'ld_list': PromptCondition(
            'ld_list', SOURCE_LD_TEMPLATE, 'list', 'secondary_list',
            'ambiguity_procedure_spec.md template with a randomised multi-item list (secondary protocol); '
            'words only — the pool has no nonword foils', 'secondary'),
    }
    if cfg.task_prompts_json:
        extra = read_json(cfg.task_prompts_json)
        for name, d in extra.items():
            conds[name] = PromptCondition(name, d['template'], d.get('protocol', 'isolated'),
                                          d.get('role', 'task_manipulation'),
                                          d.get('wording_source', f'user file {cfg.task_prompts_json}'),
                                          d.get('status', 'exploratory'))
    for c in conds.values():
        if c.protocol in ('isolated', 'list'):
            check_template(c.name, c.template)
    if cfg.primary_condition not in conds or conds[cfg.primary_condition].protocol != 'isolated':
        raise CriticalError(f"--primary_condition {cfg.primary_condition!r} must name an isolated-item condition")
    return conds


def active_conditions(cfg, conds: dict) -> list[str]:
    names = [cfg.primary_condition]
    if cfg.run_prompt_manipulations:
        names += [n for n, c in conds.items() if c.protocol == 'isolated' and n != cfg.primary_condition]
    if cfg.run_list_protocol:
        names += [n for n, c in conds.items() if c.protocol == 'list']
    if cfg.context_file:
        names.append('context_file')
    return names


def render_items(template: str, words: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Template text with the item list (one item per line, each followed by
    '\\n'); returns the text and each item's character span."""
    pre = template.split(ITEM_SLOT)[0]
    buf, spans = pre, []
    for w in words:
        s = len(buf)
        buf += w
        spans.append((s, s + len(w)))
        buf += '\n'
    return buf, spans


def load_context_file(path: str) -> tuple[pd.DataFrame, dict]:
    """User-supplied contexts: columns word, context_id, text (with {TARGET} once);
    optional bias, dominance (dominant/subordinate/neutral), source,
    validation_status. Nothing here is generated."""
    t = pd.read_csv(path, dtype=str).fillna('')
    need = {'word', 'context_id', 'text'}
    if not need <= set(t.columns):
        raise CriticalError(f"--context_file needs columns {sorted(need)}")
    t['word'] = t['word'].str.strip().str.lower()
    bad = t[t['text'].str.count(re.escape(TARGET_SLOT)) != 1]
    if len(bad):
        raise CriticalError(f"--context_file: {len(bad)} rows do not contain {TARGET_SLOT} exactly once")
    leak = t[t['text'].map(lambda x: bool(LEAK_RE.search(x)))]
    if len(leak):
        raise CriticalError(f"--context_file: {len(leak)} contexts contain analysis labels "
                            f"(e.g. {leak['text'].iloc[0]!r})")
    for c in ('bias', 'dominance', 'source', 'validation_status'):
        if c not in t.columns:
            t[c] = ''
    validated = bool(len(t)) and (t['validation_status'].str.lower() == 'human-validated').all()
    info = {'path': os.path.abspath(path), 'sha256': sha256_file(path), 'n_contexts': int(len(t)),
            'n_words': int(t['word'].nunique()),
            'has_bias_labels': bool((t['bias'] != '').any()),
            'has_dominance_labels': bool(t['dominance'].str.lower().isin(['dominant', 'subordinate']).any()),
            'all_human_validated': validated,
            'status': 'confirmatory-eligible' if validated else 'exploratory (not marked human-validated)'}
    return t, info


# ════════════════════════════════════════════════════════════════════════════
# SCORING UNITS (one forward pass each) AND TARGET ALIGNMENT
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Unit:
    unit_id: str
    text: str                                   # BASE: whole input; INSTRUCT: user-message content
    targets: list                               # [(word, char_start, char_end)] in `text`
    meta: list                                  # per target: dict (list_position, repeat, context_id, ...)


@dataclass
class EncodedUnit:
    unit_id: str
    input_ids: list
    full_text: str                              # exact text that was tokenised (rendered chat for INSTRUCT)
    targets: list                               # per target: alignment dict (see align_target)


def build_units(cond: PromptCondition, words: list[str], cfg, samples: Samples | None = None,
                context_df: pd.DataFrame | None = None) -> list[Unit]:
    units = []
    if cond.protocol == 'isolated':
        for w in words:
            text, spans = render_items(cond.template, [w])
            units.append(Unit(f"{cond.name}|{w}", text, [(w, *spans[0])], [{'word': w}]))
    elif cond.protocol == 'list':
        # Stage 1 and Stage 2 words never share a list (the stages are never mixed).
        pools = {'stage1': sorted(set(samples.stage1['word'])),
                 'stage2': sorted(set(samples.stage2_base['word']))}
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
                                        'list_length': len(chunk)} for j, w in enumerate(chunk)]))
    elif cond.protocol == 'context':
        for _, r in context_df.iterrows():
            pre, post = r['text'].split(TARGET_SLOT)
            text = pre + r['word'] + post
            units.append(Unit(f"context|{r['word']}|{r['context_id']}", text,
                              [(r['word'], len(pre), len(pre) + len(r['word']))],
                              [{'word': r['word'], 'context_id': r['context_id'], 'bias': r['bias'],
                                'dominance': r['dominance']}]))
    else:
        raise ValueError(cond.protocol)
    return units


def align_target(offsets: list, ids: list, text: str, span: tuple[int, int], word: str, decode) -> dict:
    """Map a character span to the token span that realises it in THIS input.
    A target token may absorb leading whitespace (byte-level BPE ' word');
    any other overlap with neighbouring characters (merge across the word
    boundary) is an alignment failure, never silently accepted."""
    s, e = span
    out = {'word': word, 'char_start': s, 'char_end': e, 'ok': False, 'reason': '', 't0': -1, 't1': -1,
           'n_sub': 0, 'leading_ws_absorbed': False, 'boundary_index': None, 'boundary_id': None,
           'boundary_text': None, 'decoded': None, 'target_ids': []}
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
    if t1 + 1 < len(ids):
        out['boundary_index'] = t1 + 1
        out['boundary_id'] = int(ids[t1 + 1])
        out['boundary_text'] = decode([ids[t1 + 1]])
    return out


def offsets_consistent(offsets: list, ids: list, text: str, decode1) -> bool:
    """Sanity check of a fast tokenizer's offset_mapping: non-empty spans must be
    monotone, reproduce the decoded token text, and cover every non-space
    character of the text."""
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
    tokenizer's offset_mapping fails offsets_consistent). Tokens that are not
    part of `text` (e.g. a BOS added by the tokenizer) get (0, 0)."""
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


# ════════════════════════════════════════════════════════════════════════════
# OFFLINE TINY TEST MODELS (random weights; pipeline tests only)
# ════════════════════════════════════════════════════════════════════════════
TINY_CORPUS_WORDS = ('bank paper river money house table water light school music party watch '
                     'letter string word real english item time each one decide quickly accurately '
                     'whether respond exact format line ease hard easy explain revise answers think '
                     'different meanings related judge completely closely participant experiment '
                     'lexical decision meaning relatedness see will you are a the of for and to is').split()
TINY_CHAT_TEMPLATE = (
    "{{ bos_token }}{% if messages[0]['role'] != 'system' %}<|system|>Default test system block. "
    "Date: {{ date_string | default('01 Jan 2000') }}<|end|>{% endif %}"
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] | trim }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}")


def build_tiny_tokenizer(bos: bool, chat: bool):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast
    corpus = [SOURCE_LD_TEMPLATE.replace(ITEM_SLOT, ''), AUTHORED_RELATEDNESS_TEMPLATE.replace(ITEM_SLOT, ''),
              ' '.join(TINY_CORPUS_WORDS), '\n'.join(TINY_CORPUS_WORDS)] * 4
    specials = ['<|bos|>', '<|eos|>', '<|pad|>', '<|system|>', '<|user|>', '<|assistant|>', '<|end|>']
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=520, special_tokens=specials, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(corpus, trainer)
    if bos:
        tok.post_processor = processors.TemplateProcessing(
            single='<|bos|> $A', special_tokens=[('<|bos|>', tok.token_to_id('<|bos|>'))])
    ft = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token='<|bos|>' if (bos or chat) else None,
                                 eos_token='<|eos|>', pad_token='<|pad|>')
    if chat:
        ft.chat_template = TINY_CHAT_TEMPLATE
    return ft


def build_tiny_test_model(kind: str, seed: int, dtype=None):
    torch = _torch()
    tr = _transformers()
    bos = kind in ('llama', 'llama_chat')
    tok = build_tiny_tokenizer(bos=bos, chat=(kind == 'llama_chat'))
    common = dict(vocab_size=len(tok), hidden_size=48, intermediate_size=96, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=4096, tie_word_embeddings=False,
                  eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    torch.manual_seed(seed + {'llama': 1, 'qwen2': 2, 'llama_chat': 3}[kind])
    if kind in ('llama', 'llama_chat'):
        cfg = tr.LlamaConfig(num_hidden_layers=4, bos_token_id=tok.bos_token_id, **common)
        model = tr.LlamaForCausalLM(cfg)
    elif kind == 'qwen2':
        cfg = tr.Qwen2Config(num_hidden_layers=3, **common)
        model = tr.Qwen2ForCausalLM(cfg)
    else:
        raise ValueError(kind)
    if dtype is not None:
        model = model.to(dtype)
    model.eval()
    return tok, model


# ════════════════════════════════════════════════════════════════════════════
# FROZEN LANGUAGE MODEL WITH A VERIFIED LOGIT LENS
# ════════════════════════════════════════════════════════════════════════════
DTYPE_NAMES = {'bf16': 'bfloat16', 'fp16': 'float16', 'fp32': 'float32'}
LENS_METHOD = ('logit lens (nostalgebraist, 2020): log_softmax(W_U · norm_f(h_l) [+ b]) with the model\'s own '
               'final norm and output embedding; the last block\'s HF hidden state is post-norm (verified) and '
               'is read out without a second norm; model-level logit soft-capping / scaling applied if configured')


class ScoringModel:
    """Frozen decoder-only LM. Column convention of every layerwise array:
    column 0 = embedding output (hidden_states[0]); column k = output of block
    k-1 (block_index = k-1), k = 1..L; depth = k / L = (block_index + 1) / L."""

    def __init__(self, spec: ModelSpec, cfg, load_weights: bool = True):
        self.spec, self.cfg = spec, cfg
        self.model = None
        self.lens_ok = None
        self.lens_check = {}
        self.padding_check = {}
        self.batch_size = cfg.batch_size
        tr = _transformers()
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
            raise RuntimeError(f"{spec.alias}: a fast tokenizer (offset mapping) is required for exact "
                               f"target-span alignment")
        tcfg = getattr(self.config, 'text_config', None) or self.config
        self.num_layers = int(tcfg.num_hidden_layers)
        self.hidden_size = int(tcfg.hidden_size)
        self.vocab_size_config = int(getattr(tcfg, 'vocab_size', -1) or -1)
        self.model_type = str(getattr(tcfg, 'model_type', 'unknown'))
        self.chat = spec.chat
        self.template_kwargs = {}
        if self.chat:
            if not getattr(tok, 'chat_template', None):
                raise ChatTemplateError(f"{spec.alias}: INSTRUCT model without a chat template; refusing to "
                                        f"hand-write one")
            if 'date_string' in tok.chat_template:
                # The Llama-3.x templates insert strftime_now() otherwise, which makes
                # the input (and every score) depend on the calendar date of the run.
                self.template_kwargs['date_string'] = cfg.chat_date_string
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else (
            tok.eos_token_id if tok.eos_token_id is not None else 0)
        if load_weights:
            self.load_weights()

    # ── identity used in cache keys ─────────────────────────────────────────
    def tokenizer_fingerprint(self) -> dict:
        tok = self.tokenizer
        return {'class': type(tok).__name__, 'len': len(tok),
                'vocab_sha256': sha256_json(sorted(tok.get_vocab().items())),
                'special_tokens': {k: str(v) for k, v in tok.special_tokens_map.items()},
                'bos_token_id': tok.bos_token_id, 'eos_token_id': tok.eos_token_id,
                'pad_token_id': tok.pad_token_id,
                'chat_template_sha256': sha256_text(tok.chat_template) if self.chat else None,
                'template_kwargs': self.template_kwargs}

    # ── weights ──────────────────────────────────────────────────────────────
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
            use_map = cfg.device_map == 'auto' and cfg.resolved_device.startswith('cuda')
            if use_map or cfg.quantization != 'none':
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
        self.final_norm = None
        for a in ('norm', 'final_layernorm', 'ln_f', 'final_layer_norm', 'norm_f'):
            m = getattr(self.backbone, a, None)
            if isinstance(m, torch.nn.Module):
                self.final_norm, self.final_norm_name = m, a
                break
        self.input_device = self.model.get_input_embeddings().weight.device
        self.head_device = self.head.weight.device
        self.head_dtype = self.head.weight.dtype
        params = list(self.final_norm.parameters()) if self.final_norm is not None else []
        self.norm_device = params[0].device if params else self.head_device
        self.norm_dtype = params[0].dtype if params else self.head_dtype
        tcfg = getattr(self.model.config, 'text_config', None) or self.model.config
        self.softcap = getattr(tcfg, 'final_logit_softcapping', None)
        self.logit_scale = getattr(tcfg, 'logit_scale', None)
        self.last_is_post_norm = True
        self.n_parameters = int(sum(p.numel() for p in self.model.parameters()))
        self.attn_implementation = str(getattr(self.model.config, '_attn_implementation', None))

    def free(self):
        self.model = self.backbone = self.head = self.final_norm = None
        self._test_model = None
        gc.collect()
        try:
            torch = _torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                                            # noqa: BLE001
            pass

    # ── tokenisation ─────────────────────────────────────────────────────────
    def encode_unit(self, unit: Unit) -> EncodedUnit:
        tok = self.tokenizer
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
                raise ChatTemplateError(f"{self.spec.alias}: the chat template altered the user text so that "
                                        f"its target positions cannot be located")
            enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
            ids = [int(i) for i in enc['input_ids']]
            if ids != ids_tpl:
                raise ChatTemplateError(f"{self.spec.alias}: tokenising the rendered chat string differs from "
                                        f"apply_chat_template(tokenize=True)")
            full_text = rendered
            spans = [(w, s + shift, e + shift) for w, s, e in unit.targets]
        else:
            enc = tok(unit.text, add_special_tokens=True, return_offsets_mapping=True)
            ids = [int(i) for i in enc['input_ids']]
            full_text = unit.text
            spans = list(unit.targets)
        offsets = [tuple(int(v) for v in o) for o in enc['offset_mapping']]
        dec1 = lambda i: tok.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        decn = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        self.offset_method = 'offset_mapping'
        if getattr(self, '_force_decode_offsets', False) or not offsets_consistent(offsets, ids, full_text, dec1):
            rebuilt = offsets_by_decoding(ids, full_text, decn)
            if rebuilt is not None and offsets_consistent(rebuilt, ids, full_text, dec1):
                offsets, self.offset_method = rebuilt, 'decode_fallback'
            else:
                offsets, self.offset_method = [(0, 0)] * len(ids), 'unavailable'
        bos = tok.bos_token_id
        if bos is not None and len(ids) > 1 and ids[0] == bos and ids[1] == bos:
            raise RuntimeError(f"{self.spec.alias}: duplicated BOS token at the start of the input")
        dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        targets = []
        for (w, s, e), meta in zip(spans, unit.meta):
            al = align_target(offsets, ids, full_text, (s, e), w, dec)
            if self.offset_method == 'unavailable':
                al.update({'ok': False, 'reason': 'token_offsets_unavailable'})
            al['offset_method'] = self.offset_method
            al.update({k: v for k, v in meta.items() if k != 'word'})
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
        # RIGHT padding: real tokens start at position 0, so absolute positions
        # (and the target / prediction indices) are unchanged by padding, and
        # under causal masking no real token attends to a pad.
        return ids.to(self.input_device), am.to(self.input_device)

    def _hidden_states(self, ids, am):
        out = self.backbone(input_ids=ids, attention_mask=am, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        if hs is None or len(hs) != self.num_layers + 1:
            raise RuntimeError(f"{self.spec.alias}: expected {self.num_layers + 1} hidden-state tensors, got "
                               f"{None if hs is None else len(hs)}")
        for h in (hs[0], hs[-1]):
            if tuple(h.shape) != (ids.shape[0], ids.shape[1], self.hidden_size):
                raise RuntimeError(f"{self.spec.alias}: hidden-state shape {tuple(h.shape)} != "
                                   f"{(ids.shape[0], ids.shape[1], self.hidden_size)}")
        return hs

    def _postprocess_logits(self, logits):
        torch = _torch()
        if self.logit_scale not in (None, 1, 1.0):
            logits = logits * float(self.logit_scale)
        if self.softcap:
            logits = float(self.softcap) * torch.tanh(logits / float(self.softcap))
        return logits

    def lens_logprobs(self, h, k: int):
        """log-probabilities of the logit lens at layer column k for states h (P, d)."""
        torch = _torch()
        x = h
        if not (k == self.num_layers and self.last_is_post_norm):
            x = self.final_norm(x.to(self.norm_device, self.norm_dtype))
        logits = self.head(x.to(self.head_device, self.head_dtype)).float()
        return torch.log_softmax(self._postprocess_logits(logits), dim=-1)

    def score_encoded(self, encoded: list[EncodedUnit], want_reps: bool, batch_size: int | None = None,
                      desc: str = '') -> dict:
        torch = _torch()
        L1 = self.num_layers + 1
        n_t = sum(len(u.targets) for u in encoded)
        t_start = np.cumsum([0] + [len(u.targets) for u in encoded])
        flat = [t for u in encoded for t in u.targets]
        n_sub = np.array([t['n_sub'] if t['ok'] else 0 for t in flat], int)
        tok_start = np.concatenate([[0], np.cumsum(n_sub)[:-1]]) if n_t else np.zeros(0, int)
        n_tok = int(n_sub.sum())
        tok_logp = np.full((n_tok, L1), np.nan, np.float32)
        boundary = np.full((n_t, L1), np.nan, np.float32)
        entropy = np.full((n_t, L1), np.nan, np.float32)
        reps_last = reps_mean = None
        if want_reps:
            reps_last = np.zeros((n_t, L1, self.hidden_size), np.float16)
            reps_mean = np.zeros((n_t, L1, self.hidden_size), np.float16)
        bs = batch_size or self.batch_size
        order = sorted(range(len(encoded)), key=lambda i: -len(encoded[i].input_ids))
        t0_clock = time.time()
        for bi in range(0, len(order), bs):
            idxs = order[bi:bi + bs]
            batch = [encoded[i] for i in idxs]
            ids, am = self._pad(batch)
            rows_p, pos_p, tgt_p, own_p = [], [], [], []
            rows_b, pos_b, tgt_b, own_b = [], [], [], []
            rows_e, pos_e, own_e = [], [], []
            rep_rows, rep_t0, rep_t1, rep_own = [], [], [], []
            for b, ui in enumerate(idxs):
                u = encoded[ui]
                for j, t in enumerate(u.targets):
                    gi = int(t_start[ui] + j)
                    if not t['ok']:
                        continue
                    for s, tt in enumerate(range(t['t0'], t['t1'] + 1)):
                        rows_p.append(b)
                        pos_p.append(tt - 1)              # position t-1 predicts token t
                        tgt_p.append(u.input_ids[tt])
                        own_p.append(int(tok_start[gi] + s))
                    rows_e.append(b)
                    pos_e.append(t['t0'] - 1)
                    own_e.append(gi)
                    if t['boundary_index'] is not None:
                        rows_b.append(b)
                        pos_b.append(t['t1'])            # last target token predicts the boundary token
                        tgt_b.append(t['boundary_id'])
                        own_b.append(gi)
                    if want_reps:
                        rep_rows.append(b)
                        rep_t0.append(t['t0'])
                        rep_t1.append(t['t1'])
                        rep_own.append(gi)
            with torch.inference_mode():
                hs = self._hidden_states(ids, am)
                P = len(rows_p)
                all_rows = rows_p + rows_b + rows_e
                all_pos = pos_p + pos_b + pos_e
                all_tgt = tgt_p + tgt_b + [-1] * len(rows_e)
                for k in range(L1):
                    h = hs[k]
                    if self.lens_ok and all_rows:
                        r_t = torch.tensor(all_rows, device=h.device)
                        p_t = torch.tensor(all_pos, device=h.device)
                        sel_all, ent_all = [], []
                        for c0 in range(0, len(all_rows), self.cfg.lens_chunk):
                            hh = h[r_t[c0:c0 + self.cfg.lens_chunk], p_t[c0:c0 + self.cfg.lens_chunk]]
                            lp = self.lens_logprobs(hh, k)
                            tg = torch.tensor([max(x, 0) for x in all_tgt[c0:c0 + self.cfg.lens_chunk]],
                                              device=lp.device)
                            sel_all.append(lp.gather(1, tg[:, None])[:, 0].cpu())
                            ent_all.append((-(lp.exp() * lp).sum(-1) / LN2).cpu())
                        sel = torch.cat(sel_all).numpy()
                        ent = torch.cat(ent_all).numpy()
                        tok_logp[own_p, k] = sel[:P]
                        if rows_b:
                            boundary[own_b, k] = sel[P:P + len(rows_b)]
                        if rows_e:
                            entropy[own_e, k] = ent[P + len(rows_b):]
                    if want_reps and rep_rows:
                        hr = h.float()
                        T = hr.shape[1]
                        ar = torch.arange(T, device=hr.device)
                        t0s = torch.tensor(rep_t0, device=hr.device)
                        t1s = torch.tensor(rep_t1, device=hr.device)
                        rr = torch.tensor(rep_rows, device=hr.device)
                        last = hr[rr, t1s]
                        mask = ((ar[None, :] >= t0s[:, None]) & (ar[None, :] <= t1s[:, None])).float()
                        mean = (hr[rr] * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
                        reps_last[rep_own, k] = last.cpu().numpy().astype(np.float16)
                        reps_mean[rep_own, k] = mean.cpu().numpy().astype(np.float16)
                del hs
            n_batches = max(1, math.ceil(len(order) / bs))
            if desc and n_batches >= 20 and (bi // bs) % (n_batches // 10) == 0:
                logger.info(f"  {desc}: {min(bi + bs, len(order))}/{len(order)} units "
                            f"({time.time() - t0_clock:.0f}s)")
        return {'tok_logp': tok_logp, 'tok_start': tok_start, 'n_sub': n_sub, 'boundary_logp': boundary,
                'entropy_first_bits': entropy, 'reps_last': reps_last, 'reps_mean': reps_mean}

    # ── preflight checks ─────────────────────────────────────────────────────
    def verify_readout(self, sample: list[EncodedUnit]) -> dict:
        """Establish empirically (i) the hidden-state count / shapes, (ii)
        whether hidden_states[-1] is post-norm, (iii) that the logit-lens readout
        at the last block reproduces the model's own next-token log-probabilities
        at the exact target-prediction positions. A failure marks the layerwise
        surprisal of this model UNAVAILABLE instead of producing misleading values."""
        torch = _torch()
        res = {'method': LENS_METHOD, 'status': 'UNAVAILABLE', 'reason': ''}
        if self.final_norm is None:
            res['reason'] = 'no final normalisation module found on the backbone'
            self.lens_ok, self.lens_check = False, res
            return res
        batch = [u for u in sample if any(t['ok'] for t in u.targets)][:3]
        if not batch:
            res['reason'] = 'no aligned target available for verification'
            self.lens_ok, self.lens_check = False, res
            return res
        ids, am = self._pad(batch)
        with torch.inference_mode():
            out = self.model(input_ids=ids, attention_mask=am, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
            res['n_hidden_state_tensors'] = len(hs)
            res['expected_hidden_state_tensors'] = self.num_layers + 1
            res['hidden_size_observed'] = int(hs[-1].shape[-1])
            if len(hs) != self.num_layers + 1 or int(hs[-1].shape[-1]) != self.hidden_size:
                res['reason'] = 'hidden-state count or width does not match the configuration'
                self.lens_ok, self.lens_check = False, res
                return res
            rows, pos, tg = [], [], []
            for b, u in enumerate(batch):
                for t in u.targets:
                    if t['ok']:
                        for tt in range(t['t0'], t['t1'] + 1):
                            rows.append(b)
                            pos.append(tt - 1)
                            tg.append(u.input_ids[tt])
            r_t = torch.tensor(rows, device=out.logits.device)
            p_t = torch.tensor(pos, device=out.logits.device)
            ref = out.logits[r_t, p_t].float()
            h_last = hs[-1][r_t.to(hs[-1].device), p_t.to(hs[-1].device)]
            direct = self._postprocess_logits(self.head(h_last.to(self.head_device, self.head_dtype)).float())
            normed = self._postprocess_logits(self.head(
                self.final_norm(h_last.to(self.norm_device, self.norm_dtype)).to(self.head_device,
                                                                                  self.head_dtype)).float())
            rel = lambda a: float((a.to(ref.device) - ref).norm() / ref.norm().clamp_min(1e-6))
            e_direct, e_normed = rel(direct), rel(normed)
            self.last_is_post_norm = bool(e_direct <= e_normed)
            res.update({'relative_error_head_on_last_state': e_direct,
                        'relative_error_head_on_norm_of_last_state': e_normed,
                        'hidden_states_last_is_post_norm': self.last_is_post_norm})
            if min(e_direct, e_normed) > 0.02:
                res['reason'] = (f'the output head does not reproduce the model logits (relative errors '
                                 f'{e_direct:.3g} / {e_normed:.3g})')
                self.lens_ok, self.lens_check = False, res
                return res
            self.lens_ok = True
            lp_lens = self.lens_logprobs(h_last, self.num_layers)
            lp_ref = torch.log_softmax(ref, -1)
            tgt = torch.tensor(tg, device=lp_ref.device)
            d_tgt = (lp_lens.to(lp_ref.device).gather(1, tgt[:, None]) - lp_ref.gather(1, tgt[:, None])).abs()
            res['max_abs_diff_target_logprob_nats_last_layer'] = float(d_tgt.max())
            # intermediate layers: lens must differ from a raw (un-normalised) dot product
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
            self.lens_ok = False
        else:
            res['status'] = 'OK'
        self.lens_check = res
        return res

    def check_padding(self, encoded: list[EncodedUnit]) -> dict:
        """Scores of the same targets in one padded batch vs one unit at a time
        (no padding). A violation beyond tolerance switches this model to
        batch size 1, which removes padding altogether."""
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
            self.padding_check = {'status': 'SKIPPED', 'reason': 'fewer than two units of different length'}
            return self.padding_check
        together = self.score_encoded(pick, want_reps=False, batch_size=len(pick))['tok_logp']
        single = np.concatenate([self.score_encoded([u], want_reps=False, batch_size=1)['tok_logp']
                                 for u in pick], axis=0)
        d = np.nanmax(np.abs(together - single)) / LN2 if together.size else 0.0
        tol = 1e-3 if self.cfg.dtype == 'fp32' else self.cfg.padding_tolerance_bits
        st = {'max_abs_diff_bits': float(d), 'tolerance_bits': tol, 'n_units': len(pick),
              'lengths': [len(u.input_ids) for u in pick], 'padding': 'manual right padding + attention mask'}
        if d > tol:
            st['status'] = 'FAIL→batch_size=1'
            self.batch_size = 1
            logger.warning(f"{self.spec.alias}: batched vs single scores differ by {d:.3g} bits (> {tol}); "
                           f"falling back to batch_size=1 (no padding)")
        else:
            st['status'] = 'PASS'
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
            'vocab_size_config': self.vocab_size_config, 'tokenizer_len': len(tok),
            'tokenizer_class': type(tok).__name__,
            'special_tokens': {k: str(v) for k, v in tok.special_tokens_map.items()},
            'special_token_ids': {'bos': tok.bos_token_id, 'eos': tok.eos_token_id, 'pad': tok.pad_token_id,
                                  'unk': tok.unk_token_id, 'pad_id_used_for_padding': self.pad_id},
            'tokenizer_adds_bos': bool(tok.bos_token_id is not None and probe and probe[0] == tok.bos_token_id),
            'tokenizer_padding_side_default': getattr(tok, 'padding_side', None),
            'padding_used': 'manual right padding with attention mask (positions of real tokens unchanged)',
            'chat_template_present': bool(getattr(tok, 'chat_template', None)),
            'chat_template_applied': self.chat,
            'chat_template_sha256': sha256_text(tok.chat_template) if getattr(tok, 'chat_template', None) else None,
            'chat_template_kwargs': self.template_kwargs,
            'dtype': getattr(self, 'dtype', None), 'device_map': str(getattr(self, 'device_map', None)),
            'attn_implementation': getattr(self, 'attn_implementation', None),
            'quantization': self.cfg.quantization, 'n_parameters': getattr(self, 'n_parameters', None),
            'final_norm_module': getattr(self, 'final_norm_name', None),
            'logit_softcapping': getattr(self, 'softcap', None), 'logit_scale': getattr(self, 'logit_scale', None),
            'lens_verification': self.lens_check, 'padding_check': self.padding_check,
            'batch_size_used': self.batch_size}


# ════════════════════════════════════════════════════════════════════════════
# CACHE (per model × condition; manifest-keyed, stale caches refused)
# ════════════════════════════════════════════════════════════════════════════
AGGREGATES = ('sum', 'mean', 'sum_eow', 'first')
AGGREGATE_DEFINITIONS = {
    'sum': 'Σ_j −log2 p(t_j | context, t_<j): sequence surprisal of all target subtokens (bits)',
    'mean': 'sum / number of subtokens (length-normalised companion)',
    'sum_eow': ('sum + surprisal of the token that follows the word (end-of-word correction, cf. Pimentel & '
                'Meister, 2024); the following token is the item-closing newline for BASE inputs and whatever '
                'the chat template places after the user text for INSTRUCT inputs'),
    'first': 'surprisal of the FIRST subtoken only (diagnostic; never used as the whole-word score)'}


def units_fingerprint(encoded: list[EncodedUnit]) -> str:
    return sha256_json([[u.unit_id, u.input_ids, [[t['t0'], t['t1'], t['ok']] for t in u.targets]]
                        for u in encoded])


def condition_words_fingerprint(units: list[Unit]) -> str:
    return sha256_json([[u.unit_id, [t[0] for t in u.targets]] for u in units])


def analysis_key(cfg, cond: PromptCondition, units: list[Unit]) -> dict:
    """Keys checkable WITHOUT a tokenizer (used by --analysis_only)."""
    return {'script_version': SCRIPT_VERSION, 'condition': cond.name, 'protocol': cond.protocol,
            'template_sha256': cond.template_sha256 if cond.protocol != 'context' else None,
            'context_sha256': cfg.context_sha256 if cond.protocol == 'context' else None,
            'units_words_sha256': condition_words_fingerprint(units), 'dtype': cfg.dtype,
            'quantization': cfg.quantization,
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

    def mismatches(self, expected_key: dict, analysis_only: bool = False) -> list[str]:
        if not self.exists():
            return ['<missing>']
        have = self.manifest()
        if analysis_only:
            hk, ek = have.get('analysis_key', {}), expected_key
        else:
            hk, ek = have.get('key', {}), expected_key
        return sorted(k for k in set(hk) | set(ek) if _clean_json(hk.get(k)) != _clean_json(ek.get(k)))

    def save(self, key: dict, akey: dict, info: dict, arrays: dict, reps: dict):
        tmp = self.dir + '.partial'
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)
        with open(os.path.join(tmp, 'scores.npz'), 'wb') as f:
            np.savez_compressed(f, **arrays)
        saved = {}
        for name, arr in reps.items():
            if arr is not None:
                np.save(os.path.join(tmp, f'reps_{name}.npy'), arr)
                saved[name] = list(arr.shape)
        man = {'key': key, 'analysis_key': akey, 'info': dict(info, representations_saved=saved,
                                                                created_utc=datetime.now(timezone.utc).isoformat())}
        atomic_write_json(os.path.join(tmp, 'manifest.json'), man)
        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)
        os.replace(tmp, self.dir)

    def load(self) -> tuple[dict, dict, dict]:
        man = self.manifest()
        with np.load(os.path.join(self.dir, 'scores.npz'), allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        reps = {}
        for name in ('last', 'mean'):
            p = os.path.join(self.dir, f'reps_{name}.npy')
            if os.path.exists(p):
                reps[name] = np.load(p, mmap_mode='r')
        return man, arrays, reps


def pack_scores(encoded: list[EncodedUnit], sc: dict) -> dict:
    flat = [(u.unit_id, t) for u in encoded for t in u.targets]
    meta_cols = ('list_position', 'repeat', 'pool', 'list_length', 'context_id', 'bias', 'dominance')
    arrays = {'unit_id': np.array([f[0] for f in flat], dtype=str),
              'word': np.array([f[1]['word'] for f in flat], dtype=str),
              'align_ok': np.array([f[1]['ok'] for f in flat], bool),
              'align_reason': np.array([f[1]['reason'] for f in flat], dtype=str),
              'leading_ws_absorbed': np.array([f[1]['leading_ws_absorbed'] for f in flat], bool),
              'has_boundary': np.array([f[1]['boundary_index'] is not None for f in flat], bool),
              'n_sub': sc['n_sub'].astype(int), 'tok_start': sc['tok_start'].astype(int),
              'tok_logp': sc['tok_logp'], 'boundary_logp': sc['boundary_logp'],
              'entropy_first_bits': sc['entropy_first_bits']}
    for c in meta_cols:
        vals = [f[1].get(c, '') for f in flat]
        arrays[f'meta_{c}'] = np.array(['' if v is None else str(v) for v in vals], dtype=str)
    return arrays


def aggregate_surprisal(arrays: dict) -> dict:
    """Per-target layerwise surprisal (bits) for every aggregate; NaN for
    targets that failed alignment. Shape (n_targets, L+1)."""
    tok = arrays['tok_logp'].astype(np.float64)
    n_sub = arrays['n_sub']
    start = arrays['tok_start']
    n_t = len(n_sub)
    L1 = tok.shape[1] if tok.ndim == 2 and tok.shape[0] else arrays['boundary_logp'].shape[1]
    s = np.full((n_t, L1), np.nan)
    first = np.full((n_t, L1), np.nan)
    ok = n_sub > 0
    if ok.any():
        idx = np.where(ok)[0]
        sums = np.add.reduceat(tok, start[idx], axis=0)
        s[idx] = -sums / LN2
        first[idx] = -tok[start[idx]] / LN2
    out = {'sum': s, 'mean': s / np.where(ok, n_sub, np.nan)[:, None], 'first': first}
    b = -arrays['boundary_logp'].astype(np.float64) / LN2
    out['sum_eow'] = s + b
    return out


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS PLAN — fixed in code before this script was run on any model
# ════════════════════════════════════════════════════════════════════════════
REGIONS = {'shallow': (0.0, 1.0 / 3.0), 'middle': (1.0 / 3.0, 2.0 / 3.0), 'deep': (2.0 / 3.0, 1.0)}
STAGE1_TARGETS = {          # human result (Rodd et al., 2002) → expected sign on SURPRISAL under the stand-in
    'stage1_A': {'ambiguity': ('+', 'significant', 'Exp. 1 regression: ambiguity β = +.18, t = 2.7 (slower)'),
                 'wn_n_all': ('-', 'significant', 'Exp. 1 regression: word senses β = −.17, t = −2.4 (faster)')},
    'stage1_B': {'senses': ('-', 'significant', 'Exp. 2 visual: many senses 14 ms faster, significant'),
                 'ambiguity': ('0', 'null', 'Exp. 2 visual: ambiguity 6 ms slower, NOT significant')},
    'stage1_C': {'senses': ('-', 'significant', 'Exp. 3 auditory: many senses 33 ms faster, significant'),
                 'ambiguity': ('+', 'significant', 'Exp. 3 auditory: ambiguity 29 ms slower, significant')},
}
STAGE2_EXPECTED = {'gcide_etymologies': '+', 'wn_n_all': '-', 'log_subtlex': '-'}
STAGE2_SPECS = {
    'primary': (['gcide_etymologies', 'wn_n_all', 'log_subtlex', 'length', 'concreteness'], 'confirmatory'),
    'robust_orthoN': (['gcide_etymologies', 'wn_n_all', 'log_subtlex', 'length', 'concreteness', 'ortho_n'],
                      'robustness'),
    'robust_orthoN_nsyll': (['gcide_etymologies', 'wn_n_all', 'log_subtlex', 'length', 'concreteness',
                             'ortho_n', 'n_syll'], 'robustness'),
    # Summed surprisal grows mechanically with the number of subtokens, and rare
    # words split into more subtokens; this model-specific count is therefore a
    # prespecified robustness covariate (it is not in the source formula).
    'robust_n_subtokens': (['gcide_etymologies', 'wn_n_all', 'log_subtlex', 'length', 'concreteness',
                            'n_subtokens'], 'robustness'),
    'explor_sim_top2': (['gcide_etymologies', 'wn_n_all', 'log_subtlex', 'length', 'concreteness', 'sim_top2'],
                        'exploratory'),
    'explor_min_sim': (['gcide_etymologies', 'wn_n_all', 'log_subtlex', 'length', 'concreteness', 'min_sim'],
                       'exploratory'),
    'explor_mean_sim': (['gcide_etymologies', 'wn_n_all', 'log_subtlex', 'length', 'concreteness', 'mean_sim'],
                        'exploratory'),
}
DEPTH_TERMS = {'frequency': 'log_subtlex', 'meanings_stage2': 'gcide_etymologies', 'senses_stage2': 'wn_n_all'}
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
        'status': ('Fixed in this script (version %s) before it was run on any model. It is NOT a formal '
                   'preregistration: register it (e.g. on OSF) before running the real models if the '
                   'confirmatory labels are to carry weight.' % SCRIPT_VERSION),
        'primary_measure': 'layerwise_surprisal (logit lens), bits; higher = target LESS expected by the model',
        'primary_aggregate': cfg.primary_aggregate, 'aggregate_definitions': AGGREGATE_DEFINITIONS,
        'primary_condition': cfg.primary_condition,
        'stand_in_mapping': ('Human processing cost (slower lexical decision) is mapped to HIGHER surprisal for '
                             'the purpose of comparing SIGNS only. Surprisal is not a reaction time.'),
        'layers': ('analyses use block outputs only; depth = (block_index + 1) / num_hidden_layers; the embedding '
                   'output (depth 0) is stored but not analysed'),
        'regions': {k: f'({a:.4f}, {b:.4f}]' for k, (a, b) in REGIONS.items()},
        'inference': ('OLS per layer; two-sided p from HC3 standard errors (classical p also reported); '
                      'Benjamini–Hochberg FDR across the layers of one model × condition × aggregate × analysis '
                      '× spec × term family (p_fdr_layers) and across all rows of a table for one term '
                      '(p_fdr_global_term); alpha = %.3g' % cfg.alpha),
        'standardised_effects': ('outcome z-scored within layer; continuous predictors z-scored; Stage 1 '
                                 'factors effect-coded ±0.5 (B, C) or dummy 0/1 (A ambiguity)'),
        'bootstrap': ('words resampled with replacement (stratified within design cells in Stage 1), the same '
                      'resample for every layer and — for condition contrasts — for both conditions; B = %d; '
                      'percentile CIs; two-sided bootstrap p with +1 correction' % cfg.n_bootstrap),
        'stage1': {
            'appendix_A': 'surprisal ~ ambiguity(0/1) + wn_n_all + log_subtlex + length + concreteness',
            'appendix_B_C': ('surprisal ~ ambiguity(±.5) * senses(±.5)   [robustness: + log_subtlex + length + '
                             'concreteness; separately + n_subtokens]'),
            'appendix_A_robustness': '+ n_subtokens (model-specific subtoken count of the target)',
            'targets': {a: {t: v[:2] for t, v in d.items()} for a, d in STAGE1_TARGETS.items()},
            'direction_match': 'every effect with an expected sign has that sign (point estimate)',
            'full_match': ('direction_match AND every expected-significant effect has p_fdr_layers < alpha AND the '
                           'expected-null effect (Exp. 2 ambiguity) has unadjusted p >= alpha'),
            'collapsed': ('ambiguity and senses both significant (FDR) with the SAME sign in one appendix = one '
                          'undifferentiated "ambiguity" effect = Stage 1 failure for that appendix'),
            'layer_pass': 'full_match in A AND B AND C at that layer',
            'gate': ('PASS if >= 1 layer passes; the number and depths of passing layers and the final-layer '
                     '(output) verdict are always reported. Stage 2 rows carry the gate of their model × '
                     'condition × aggregate; the source says Stage 2 is only interpretable if Stage 1 passes.')},
        'stage2': {'specs': {k: {'predictors': v[0], 'status': v[1]} for k, v in STAGE2_SPECS.items()},
                   'expected_signs_under_stand_in': STAGE2_EXPECTED,
                   'gcide_zero_policy_primary': cfg.gcide_zero_policy,
                   'gcide_zero_policy_note': 'UNRESOLVED in the source; all three policies are reported'},
        'depth_hypothesis': {
            'peak_depth': 'depth of max |standardised β| over blocks',
            'deep_minus_shallow': 'mean |std β| over deep blocks − mean |std β| over shallow blocks',
            'prediction': ('frequency: deep_minus_shallow < 0 (shallow); meanings and senses: > 0 (deep); '
                           'crossing = dms(ambiguity term) − dms(frequency) > 0; Δpeak = peak(ambiguity term) − '
                           'peak(frequency) > 0'),
            'caveat': ('final-layer surprisal of an isolated word is close to a log-frequency estimate, so a large '
                       'final-layer frequency effect is expected by construction for this measure')},
        'prompt_interaction': {
            'model': 'Δ_k = y_k(condition) − y_k(primary), paired by word; Δ_k ~ centred design (interaction = term × condition)',
            'depth_statistic': ('Δ|std β|(l) = |β_condition(l)| − |β_primary(l)|; prediction: deep − shallow of Δ|std β| '
                                '> 0 for meanings and senses, ≈ 0 for frequency (paired bootstrap)'),
            'status': 'exploratory (the relatedness wording is authored here; no vetted context sentences supplied)',
            'frequency_crossed': ('operationalised as (i) the log_subtlex depth profile and matched high- vs '
                                  'low-frequency contrast under EVERY condition, and (ii) the secondary list protocol, '
                                  'in which high- and low-frequency words are randomly interleaved within one prompt')},
        'probes': {'model': ('linear ridge probe (dual form), unpenalised intercept, features standardised within '
                             'training folds, λ = %.3g · mean diag(K_train), fixed a priori' % cfg.probe_lambda_scale),
                   'cv': '%d word-grouped folds (stratified for binary targets); words deduplicated' % cfg.n_folds,
                   'controls': 'label-permutation null (%d permutations), no-hidden-state baseline probe'
                               % cfg.n_permutations,
                   'status': 'secondary; decodability is not use'},
        'matched': {'contrasts': MATCH_CONTRASTS, 'caliper_sd': cfg.match_caliper_sd,
                    'status': 'robustness (Stage 2 pool, primary zero policy)'},
        'confirmatory': ['Stage 1 calibration verdicts (primary condition, primary aggregate)',
                         'Stage 2 primary spec coefficients (primary condition, aggregate, zero policy)',
                         'depth tests for log_subtlex vs gcide_etymologies / wn_n_all (primary condition)'],
        'exploratory': ['every other condition, aggregate, spec and zero policy', 'prompt interactions',
                        'representation distance', 'relatedness proxies (sim_top2, min_sim, mean_sim)',
                        'list protocol (secondary)', 'probes (secondary)'],
    }


# ════════════════════════════════════════════════════════════════════════════
# SCORE SETS (cached model outputs, ready for analysis)
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class ScoreSet:
    alias: str
    condition: str
    protocol: str
    role: str
    cond_status: str
    n_layers: int
    model_info: dict
    targets: pd.DataFrame
    surprisal: dict
    entropy: np.ndarray
    reps: dict

    @property
    def depth(self) -> np.ndarray:
        return np.arange(1, self.n_layers + 1) / self.n_layers

    def ctx(self, aggregate: str | None = None) -> dict:
        mi = self.model_info
        d = {'model': self.alias, 'repo_id': mi.get('repo_id'), 'variant': mi.get('variant'),
             'family': mi.get('family'), 'base_of': mi.get('base_of'),
             'test_model': bool(mi.get('test_model_random_weights')), 'condition': self.condition,
             'protocol': self.protocol, 'condition_status': self.cond_status}
        if aggregate is not None:
            d['aggregate'] = aggregate
        return d

    def word_matrix(self, aggregate: str, include_embedding: bool = False) -> pd.DataFrame:
        """Words × layer columns (k = 1..L; 0 = embedding if requested). Repeated
        targets of a word (list repeats) are averaged; failed alignments excluded."""
        M = self.surprisal[aggregate]
        cols = list(range(0 if include_embedding else 1, self.n_layers + 1))
        ok = self.targets['align_ok'].values
        df = pd.DataFrame(M[ok][:, cols], columns=cols)
        df['word'] = self.targets.loc[ok, 'word'].values
        return df.groupby('word', sort=True).mean()


def load_scoreset(cache: CacheEntry, cond: PromptCondition, model_info: dict) -> ScoreSet:
    man, arrays, reps = cache.load()
    sur = aggregate_surprisal(arrays)
    n_layers = int(man['key']['num_hidden_layers'])
    tg = pd.DataFrame({'unit_id': arrays['unit_id'], 'word': arrays['word'], 'align_ok': arrays['align_ok'],
                       'align_reason': arrays['align_reason'], 'n_sub': arrays['n_sub']})
    for k in arrays:
        if k.startswith('meta_'):
            tg[k[5:]] = arrays[k]
    return ScoreSet(model_info.get('alias', man['key'].get('model_alias')), cond.name, cond.protocol, cond.role,
                    cond.status, n_layers, model_info, tg, sur, arrays['entropy_first_bits'], reps)


def item_layer_frame(ss: ScoreSet) -> pd.DataFrame:
    """Per-target, per-layer long table (embedding output included, flagged)."""
    L1 = ss.n_layers + 1
    n = len(ss.targets)
    k = np.tile(np.arange(L1), n)
    out = ss.targets.loc[ss.targets.index.repeat(L1)].reset_index(drop=True)
    out.insert(0, 'protocol', ss.protocol)
    out.insert(0, 'condition', ss.condition)
    out.insert(0, 'model', ss.alias)
    out['layer_column'] = k
    out['block_index'] = k - 1
    out['layer_kind'] = np.where(k == 0, 'embedding_output', 'block_output')
    out['depth'] = k / ss.n_layers
    for agg in AGGREGATES:
        out[f'surprisal_{agg}_bits'] = ss.surprisal[agg].reshape(-1)
    out['entropy_first_position_bits'] = ss.entropy.reshape(-1)
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
    row_ids: np.ndarray
    terms: list
    raw: pd.DataFrame
    kinds: dict                    # term -> 'z' (standardised) | 'fixed' (dummy / effect code)
    strata: np.ndarray | None
    expected: dict
    zero_policy: str | None = None
    n_before_complete_cases: int = 0

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
        return Design(self.analysis, self.spec, self.status, self.words[keep], self.row_ids[keep], self.terms,
                      self.raw[keep].reset_index(drop=True), self.kinds,
                      None if self.strata is None else self.strata[keep], self.expected, self.zero_policy,
                      self.n_before_complete_cases)


def design_stage1_A(stage1: pd.DataFrame, spec: str = 'regression') -> Design:
    r = stage1[stage1['appendix'] == 'A']
    cov = ['wn_n_all', 'log_subtlex', 'length', 'concreteness'] + {
        'regression': [], 'explor_sim_top2': ['sim_top2'], 'regression+n_subtokens': ['n_subtokens']}[spec]
    n0 = len(r)
    r = r.dropna(subset=cov + ['ambiguity'])
    raw = pd.DataFrame({'ambiguity': (r['ambiguity'] == 'ambiguous').astype(float).values})
    for c in cov:
        raw[c] = r[c].values
    kinds = {'ambiguity': 'fixed', **{c: 'z' for c in cov}}
    exp = {'ambiguity': '+', 'wn_n_all': '-', 'log_subtlex': '-'}
    status = {'regression': 'confirmatory', 'explor_sim_top2': 'exploratory',
              'regression+n_subtokens': 'robustness'}[spec]
    return Design('stage1_A', spec, status,
                  r['word'].values.astype(str), r['row_id'].values.astype(str), list(raw.columns), raw, kinds,
                  r['ambiguity'].values.astype(str), exp, None, n0)


def design_stage1_BC(stage1: pd.DataFrame, appendix: str, spec: str = 'factorial') -> Design:
    r = stage1[stage1['appendix'] == appendix]
    cov = {'factorial': [], 'factorial+covariates': ['log_subtlex', 'length', 'concreteness'],
           'factorial+n_subtokens': ['n_subtokens']}[spec]
    n0 = len(r)
    r = r.dropna(subset=['ambiguity', 'senses'] + cov)
    a = np.where(r['ambiguity'] == 'ambiguous', 0.5, -0.5)
    s = np.where(r['senses'] == 'many', 0.5, -0.5)
    raw = pd.DataFrame({'ambiguity': a, 'senses': s, 'ambiguity:senses': a * s})
    for c in cov:
        raw[c] = r[c].values
    kinds = {'ambiguity': 'fixed', 'senses': 'fixed', 'ambiguity:senses': 'fixed', **{c: 'z' for c in cov}}
    exp = {'B': {'senses': '-', 'ambiguity': '0'}, 'C': {'senses': '-', 'ambiguity': '+'}}[appendix]
    strata = (r['ambiguity'].astype(str) + '/' + r['senses'].astype(str)).values
    return Design(f'stage1_{appendix}', spec, 'confirmatory' if spec == 'factorial' else 'robustness',
                  r['word'].values.astype(str), r['row_id'].values.astype(str), list(raw.columns), raw, kinds,
                  strata, exp, None, n0)


def design_stage2(rows: pd.DataFrame, spec: str, policy: str) -> Design:
    cols, status = STAGE2_SPECS[spec]
    n0 = len(rows)
    r = rows.dropna(subset=cols)
    raw = pd.DataFrame({c: r[c].values.astype(float) for c in cols})
    return Design('stage2', spec, status, r['word'].values.astype(str), r['row_id'].values.astype(str), cols, raw,
                  {c: 'z' for c in cols}, None, dict(STAGE2_EXPECTED), policy, n0)


# ════════════════════════════════════════════════════════════════════════════
# LAYERWISE FITS
# ════════════════════════════════════════════════════════════════════════════
def _resample(design: Design, rng) -> np.ndarray:
    n = design.n
    if design.strata is None:
        return rng.integers(0, n, n)
    idx = []
    for s in np.unique(design.strata):
        w = np.where(design.strata == s)[0]
        idx.append(w[rng.integers(0, len(w), len(w))])
    return np.concatenate(idx)


def boot_std_betas(design: Design, Ys: list, B: int, seed: int) -> list:
    """Bootstrap of standardised coefficients (terms × layers) for one or more
    outcome matrices with the SAME word resample (joint across layers; paired
    across conditions). Returns a list of arrays (B, k, L)."""
    rng = np.random.default_rng(seed)
    k = len(design.terms)
    out = [np.full((B, k, Y.shape[1]), np.nan) for Y in Ys]
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


def fit_layerwise(design: Design, Y: np.ndarray, depth: np.ndarray, ctx: dict, alpha: float,
                  boot: np.ndarray | None = None, include_intercept: bool = False,
                  center_raw: bool = False) -> pd.DataFrame:
    if not np.isfinite(design.X_std()).all():
        return pd.DataFrame()          # a predictor is constant in this sample: model not identifiable
    fr = ols_multi(design.X_raw(center=center_raw), Y)
    fs = ols_multi(design.X_std(), zscore_cols(Y))
    tcrit = stats.t.ppf(1 - alpha / 2, fr['df']) if fr['df'] > 0 else np.nan
    names = (['(intercept)'] if include_intercept else []) + list(design.terms)
    offs = 0 if include_intercept else 1
    rows = []
    vif = vif_table(design.raw.values.astype(float), design.terms) if len(design.terms) > 1 else {}
    for jj, term in enumerate(names):
        j = jj + offs
        b, se, sh = fr['beta'][j], fr['se'][j], fr['se_hc3'][j]
        with np.errstate(invalid='ignore', divide='ignore'):
            t_h = b / sh
            t_c = b / se
        p_h, p_c = p_two_sided_t(t_h, fr['df']), p_two_sided_t(t_c, fr['df'])
        bs, shs = fs['beta'][j], fs['se_hc3'][j]
        bl = bh = None
        if boot is not None and term != '(intercept)':
            bl, bh = percentile_ci(boot[:, design.terms.index(term), :], alpha)
        for l in range(Y.shape[1]):
            row = dict(ctx)
            row.update({'analysis': design.analysis, 'spec': design.spec, 'spec_status': design.status,
                        'zero_policy': design.zero_policy, 'layer_column': l + 1, 'block_index': l,
                        'depth': float(depth[l]), 'term': term, 'n': design.n,
                        'n_before_complete_cases': design.n_before_complete_cases,
                        'beta': b[l], 'se': se[l], 'se_hc3': sh[l], 't_hc3': t_h[l], 'p_hc3': p_h[l],
                        'p_classical': p_c[l], 'ci_low': b[l] - tcrit * sh[l], 'ci_high': b[l] + tcrit * sh[l],
                        'beta_std': bs[l], 'se_std_hc3': shs[l],
                        'ci_std_low': bs[l] - tcrit * shs[l], 'ci_std_high': bs[l] + tcrit * shs[l],
                        'boot_ci_std_low': np.nan if bl is None else bl[l],
                        'boot_ci_std_high': np.nan if bh is None else bh[l],
                        'r2': fr['r2'][l], 'df_resid': fr['df'], 'vif': vif.get(term, np.nan),
                        'expected_sign': design.expected.get(term, ''),
                        'units': 'bits per predictor unit (beta); SD(y) per SD(x) or per factor contrast (beta_std)'})
            es = design.expected.get(term, '')
            row['sign_match'] = (np.sign(b[l]) == (1 if es == '+' else -1)) if es in ('+', '-') else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df['p_fdr_layers'] = df.groupby('term', sort=False)['p_hc3'].transform(lambda s: bh_fdr(s.values))
    return df


def region_mask(depth: np.ndarray, region: str) -> np.ndarray:
    a, b = REGIONS[region]
    return (depth > a + 1e-12) & (depth <= b + 1e-12)


def depth_stats(beta: np.ndarray, depth: np.ndarray) -> dict:
    """beta: (..., L) standardised coefficients → peak depth and region means of |β|."""
    absb = np.abs(beta)
    sh, dp = region_mask(depth, 'shallow'), region_mask(depth, 'deep')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        peak = np.where(np.isfinite(absb).any(-1), depth[np.nanargmax(np.where(np.isfinite(absb), absb, -1), axis=-1)],
                        np.nan)
        s = np.nanmean(absb[..., sh], -1) if sh.any() else np.full(absb.shape[:-1], np.nan)
        d = np.nanmean(absb[..., dp], -1) if dp.any() else np.full(absb.shape[:-1], np.nan)
    return {'peak_depth': peak, 'shallow_mean_abs': s, 'deep_mean_abs': d, 'deep_minus_shallow': d - s}


def depth_profile_tests(design: Design, point: np.ndarray, boot: np.ndarray, depth: np.ndarray, ctx: dict,
                        alpha: float, freq_term: str | None, amb_terms: list) -> list:
    rows = []
    ps = depth_stats(point, depth)
    bs = depth_stats(boot, depth)
    for j, t in enumerate(design.terms):
        if t not in amb_terms and t != freq_term:
            continue
        for stat in ('peak_depth', 'shallow_mean_abs', 'deep_mean_abs', 'deep_minus_shallow'):
            lo, hi = percentile_ci(bs[stat][:, j], alpha)
            pred = ''
            if stat == 'deep_minus_shallow':
                pred = '< 0 (shallow)' if t == freq_term else '> 0 (deep)'
            rows.append(dict(ctx, analysis=design.analysis, spec=design.spec, zero_policy=design.zero_policy,
                             statistic=stat, term=t, comparison='', estimate=float(ps[stat][j]),
                             boot_ci_low=float(lo), boot_ci_high=float(hi),
                             boot_p=bootstrap_p_two_sided(bs[stat][:, j]) if stat == 'deep_minus_shallow' else np.nan,
                             prediction=pred, n=design.n, n_boot_valid=int(np.isfinite(bs[stat][:, j]).sum())))
    if freq_term in design.terms:
        f = design.terms.index(freq_term)
        for t in amb_terms:
            if t not in design.terms:
                continue
            a = design.terms.index(t)
            for stat, name, pred in (('peak_depth', 'delta_peak_depth', '> 0'),
                                     ('deep_minus_shallow', 'crossing_contrast', '> 0')):
                est = float(ps[stat][a] - ps[stat][f])
                bd = bs[stat][:, a] - bs[stat][:, f]
                lo, hi = percentile_ci(bd, alpha)
                rows.append(dict(ctx, analysis=design.analysis, spec=design.spec, zero_policy=design.zero_policy,
                                 statistic=name, term=t, comparison=f'{t} − {freq_term}', estimate=est,
                                 boot_ci_low=float(lo), boot_ci_high=float(hi), boot_p=bootstrap_p_two_sided(bd),
                                 prediction=pred, n=design.n, n_boot_valid=int(np.isfinite(bd).sum())))
    return rows


def stage1_verdicts(tbl: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Per model × condition × aggregate × layer: preregistered pattern checks."""
    prim = tbl[((tbl['analysis'] == 'stage1_A') & (tbl['spec'] == 'regression')) |
               ((tbl['analysis'].isin(['stage1_B', 'stage1_C'])) & (tbl['spec'] == 'factorial'))]
    keys = ['model', 'condition', 'aggregate', 'layer_column']
    out = []
    for kv, g in prim.groupby(keys + ['depth'], sort=False):
        row = dict(zip(keys + ['depth'], kv))
        all_full, all_dir = True, True
        for app, targets in STAGE1_TARGETS.items():
            ga = g[g['analysis'] == app].set_index('term')
            if ga.empty:
                row.update({f'{app}_direction_match': np.nan, f'{app}_full_match': np.nan, f'{app}_collapsed': np.nan})
                all_full = all_dir = False
                continue
            dmatch, smatch = True, True
            for term, (sign, kind, _) in targets.items():
                b, pf, pu = ga.loc[term, 'beta'], ga.loc[term, 'p_fdr_layers'], ga.loc[term, 'p_hc3']
                row[f'{app}_{term}_beta_std'] = ga.loc[term, 'beta_std']
                row[f'{app}_{term}_p_fdr'] = pf
                if sign in '+-':
                    dmatch &= bool(np.sign(b) == (1 if sign == '+' else -1))
                if kind == 'significant':
                    smatch &= bool(pf < alpha)
                else:
                    smatch &= bool(pu >= alpha)
            terms = list(targets)
            b1, b2 = ga.loc[terms[0], 'beta'], ga.loc[terms[1], 'beta']
            p1, p2 = ga.loc[terms[0], 'p_fdr_layers'], ga.loc[terms[1], 'p_fdr_layers']
            collapsed = bool(np.sign(b1) == np.sign(b2) and p1 < alpha and p2 < alpha)
            row[f'{app}_direction_match'] = dmatch
            row[f'{app}_full_match'] = bool(dmatch and smatch and not collapsed)
            row[f'{app}_collapsed'] = collapsed
            all_full &= row[f'{app}_full_match']
            all_dir &= dmatch
        row['stage1_direction_pass'] = bool(all_dir)
        row['stage1_full_pass'] = bool(all_full)
        out.append(row)
    return pd.DataFrame(out)


def stage1_gate_summary(verd: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, c, a), g in verd.groupby(['model', 'condition', 'aggregate'], sort=False):
        g = g.sort_values('layer_column')
        last = g.iloc[-1]
        rows.append({'model': m, 'condition': c, 'aggregate': a, 'n_layers': len(g),
                     'n_layers_full_pass': int(g['stage1_full_pass'].sum()),
                     'n_layers_direction_pass': int(g['stage1_direction_pass'].sum()),
                     'full_pass_depths': ';'.join(f'{d:.3f}' for d in g.loc[g['stage1_full_pass'], 'depth']),
                     'final_layer_full_pass': bool(last['stage1_full_pass']),
                     'final_layer_direction_pass': bool(last['stage1_direction_pass']),
                     'gate': 'PASS' if g['stage1_full_pass'].any() else 'FAIL'})
    return pd.DataFrame(rows)


def add_global_fdr(df: pd.DataFrame, group_cols=('analysis', 'spec', 'term')) -> pd.DataFrame:
    if df.empty or 'p_hc3' not in df:
        return df
    cols = [c for c in group_cols if c in df]
    df['p_fdr_global_term'] = df.groupby(cols, sort=False, dropna=False)['p_hc3'].transform(
        lambda s: bh_fdr(s.values))
    return df


# ════════════════════════════════════════════════════════════════════════════
# PROBES (secondary)
# ════════════════════════════════════════════════════════════════════════════
def run_probe_target(X_layers, y, binary: bool, groups, cfg, ctx, target, pooling, depth, baseline_X=None):
    """X_layers: callable k -> (n, d) features for layer column k (1..L)."""
    rng = np.random.default_rng([cfg.seed, int(sha256_text(target)[:8], 16)])
    folds = make_group_folds(groups, cfg.n_folds, cfg.seed, labels=y if binary else None)
    yv = np.asarray(y, float)
    yfit = np.where(yv == 1, 1.0, -1.0) if binary else (yv - yv.mean()) / (yv.std() or 1.0)
    perms = np.column_stack([yfit[rng.permutation(len(yfit))] for _ in range(cfg.n_permutations)]) \
        if cfg.n_permutations else np.zeros((len(yfit), 0))
    Ycols = np.column_stack([yfit, perms])
    rows = []
    metric_name = 'auroc_oof' if binary else 'r2_oof'
    for k in range(1, len(depth) + 1):
        pred = ridge_oof(X_layers(k), Ycols, folds, cfg.probe_lambda_scale)
        if binary:
            labs = np.column_stack([(Ycols[:, c] > 0).astype(int) for c in range(Ycols.shape[1])])
            met = auroc_columns(pred, labs)
        else:
            met = r2_columns(pred, Ycols)
        obs, null = met[0], met[1:]
        p = (1 + np.sum(null >= obs)) / (1 + len(null)) if len(null) else np.nan
        rows.append(dict(ctx, target=target, pooling=pooling, layer_column=k, block_index=k - 1,
                         depth=float(depth[k - 1]), metric=metric_name, value=float(obs),
                         perm_null_mean=float(np.mean(null)) if len(null) else np.nan,
                         perm_null_q95=float(np.quantile(null, 0.95)) if len(null) else np.nan,
                         p_perm=p, n=len(yv), n_positive=int((yv == 1).sum()) if binary else np.nan,
                         n_folds=cfg.n_folds, lambda_scale=cfg.probe_lambda_scale,
                         n_permutations=cfg.n_permutations))
    df = pd.DataFrame(rows)
    df['p_fdr_layers'] = bh_fdr(df['p_perm'].values)
    if baseline_X is not None:
        bpred = ridge_oof(baseline_X, yfit, folds, cfg.probe_lambda_scale)
        df['baseline_value'] = float(auroc_columns(bpred, (yfit > 0).astype(int))[0] if binary
                                     else r2_columns(bpred, yfit)[0])
        df['baseline_features'] = 'length, n_subtokens' + ('' if target.endswith('log_subtlex') else ', log_subtlex')
    return df


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
            t_m, c_m = r[col] >= hi, r[col] <= lo
            t_m, c_m = t_m & ~c_m, c_m & ~t_m
        T, C = r[t_m].reset_index(drop=True), r[c_m].reset_index(drop=True)
        pooled = np.sqrt((T[cov].var(ddof=1).values + C[cov].var(ddof=1).values) / 2)
        mt, mc, rej = greedy_caliper_match(T[cov].values, C[cov].values, caliper, pooled)
        info = {'definition': spec['definition'], 'covariates': cov, 'caliper_sd': caliper,
                'n_treated_pool': len(T), 'n_control_pool': len(C), 'n_pairs': len(mt), 'n_rejected': rej}
        bal = []
        for j, c in enumerate(cov):
            bal.append({'contrast': name, 'covariate': c,
                        'smd_before': standardized_mean_difference(T[c], C[c], pooled[j]),
                        'smd_after': standardized_mean_difference(T[c].values[mt], C[c].values[mc], pooled[j]),
                        'mean_treated_after': float(T[c].values[mt].mean()) if len(mt) else np.nan,
                        'mean_control_after': float(C[c].values[mc].mean()) if len(mc) else np.nan})
        pairs = pd.DataFrame({'contrast': name, 'pair': np.arange(len(mt)),
                              'treated_row_id': T['row_id'].values[mt], 'treated_word': T['word'].values[mt],
                              'control_row_id': C['row_id'].values[mc], 'control_word': C['word'].values[mc]})
        info['status'] = 'RUN' if len(mt) >= min_pairs else f'NOT_RUN (only {len(mt)} pairs < {min_pairs})'
        out[name] = {'info': info, 'balance': pd.DataFrame(bal), 'pairs': pairs}
    return out


def matched_effects(match: dict, W: pd.DataFrame, depth: np.ndarray, ctx: dict, alpha: float) -> pd.DataFrame:
    rows = []
    for name, m in match.items():
        if not m['info']['status'].startswith('RUN'):
            continue
        pr = m['pairs']
        ok = pr['treated_word'].isin(W.index) & pr['control_word'].isin(W.index)
        pr = pr[ok]
        if len(pr) < 3:
            continue
        D = W.loc[pr['treated_word']].values - W.loc[pr['control_word']].values
        n = len(D)
        mu, sd = D.mean(0), D.std(0, ddof=1)
        se = sd / np.sqrt(n)
        with np.errstate(invalid='ignore', divide='ignore'):
            t = mu / se
        p = p_two_sided_t(t, n - 1)
        tc = stats.t.ppf(1 - alpha / 2, n - 1)
        fdr = bh_fdr(p)
        for l in range(D.shape[1]):
            rows.append(dict(ctx, contrast=name, definition=m['info']['definition'], layer_column=l + 1,
                             block_index=l, depth=float(depth[l]), n_pairs=n, mean_diff_bits=mu[l], sd_diff=sd[l],
                             ci_low=mu[l] - tc * se[l], ci_high=mu[l] + tc * se[l], t=t[l], p=p[l],
                             p_fdr_layers=fdr[l], dz=mu[l] / sd[l] if sd[l] > 0 else np.nan))
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# REPRESENTATION DISTANCE (secondary)
# ════════════════════════════════════════════════════════════════════════════
def cosine_distance_by_layer(A, B) -> np.ndarray:
    """A, B: (n, L+1, d) → (n, L+1) cosine distance, computed layer by layer."""
    n, L1, _ = A.shape
    out = np.empty((n, L1))
    for k in range(L1):
        a = np.asarray(A[:, k, :], np.float32)
        b = np.asarray(B[:, k, :], np.float32)
        num = (a * b).sum(1)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            out[:, k] = 1.0 - num / den
    return out


def word_rep_index(ss: ScoreSet) -> dict:
    t = ss.targets
    return {w: i for i, w in zip(t.index[t['align_ok']], t.loc[t['align_ok'], 'word'])}


# ════════════════════════════════════════════════════════════════════════════
# LIST PROTOCOL DIAGNOSTICS (secondary)
# ════════════════════════════════════════════════════════════════════════════
def list_order_diagnostics(ss_list: ScoreSet, ss_iso: ScoreSet | None, aggregate: str) -> pd.DataFrame:
    t = ss_list.targets
    ok = t['align_ok'].values
    M = ss_list.surprisal[aggregate][:, 1:]
    pos = pd.to_numeric(t['list_position'], errors='coerce').values
    rep = pd.to_numeric(t['repeat'], errors='coerce').values
    rows = []
    iso = ss_iso.word_matrix(aggregate) if ss_iso is not None else None
    for l in range(ss_list.n_layers):
        y = M[:, l]
        g = ok & np.isfinite(y) & np.isfinite(pos)
        rho, p = stats.spearmanr(pos[g], y[g]) if g.sum() > 3 else (np.nan, np.nan)
        df = pd.DataFrame({'word': t['word'].values[g], 'rep': rep[g], 'y': y[g]})
        wide = df.pivot_table(index='word', columns='rep', values='y', aggfunc='mean')
        rel = np.nan
        if wide.shape[1] >= 2:
            cs = wide.corr().values
            rel = float(np.nanmean(cs[np.triu_indices_from(cs, 1)]))
        r_iso = np.nan
        if iso is not None:
            mw = df.groupby('word')['y'].mean()
            common = mw.index.intersection(iso.index)
            if len(common) > 3:
                r_iso = float(np.corrcoef(mw.loc[common], iso.loc[common, l + 1])[0, 1])
        rows.append(dict(ss_list.ctx(aggregate), layer_column=l + 1, block_index=l,
                         depth=float((l + 1) / ss_list.n_layers),
                         spearman_position=rho, p_position=p, between_repeat_reliability=rel,
                         r_with_isolated=r_iso, n_targets=int(g.sum())))
    out = pd.DataFrame(rows)
    out['p_position_fdr_layers'] = bh_fdr(out['p_position'].values)
    return out


# ════════════════════════════════════════════════════════════════════════════
# RUN ORCHESTRATION — scoring
# ════════════════════════════════════════════════════════════════════════════
def output_dirs(root: str) -> dict:
    d = {k: os.path.join(root, k) for k in ('prompts', 'cache', 'item_scores', 'tables', 'figures', 'logs', 'models')}
    d['root'] = root
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    return d


def condition_words(cond: PromptCondition, samples: Samples) -> list[str]:
    return list(samples.words_to_score['word'])


def tokenizer_diagnostics(sm: ScoringModel, encoded_by_cond: dict) -> dict:
    tok = sm.tokenizer
    out = {'model': sm.spec.alias, 'tokenizer_class': type(tok).__name__,
           'special_tokens': {k: str(v) for k, v in tok.special_tokens_map.items()},
           'bos_token_id': tok.bos_token_id, 'eos_token_id': tok.eos_token_id, 'pad_token_id': tok.pad_token_id,
           'padding': 'manual right padding with attention mask', 'chat_template_applied': sm.chat,
           'chat_template_kwargs': sm.template_kwargs, 'conditions': {}}
    for cname, enc in encoded_by_cond.items():
        flat = [(u, t) for u in enc for t in u.targets]
        ok = [x for x in flat if x[1]['ok']]
        reasons = pd.Series([t['reason'] for _, t in flat if not t['ok']]).value_counts().to_dict()
        nsub = pd.Series([t['n_sub'] for _, t in ok]).value_counts().sort_index().to_dict()
        picks = {}
        if ok:
            picks['shortest'] = min(ok, key=lambda x: (len(x[1]['word']), x[1]['word']))
            picks['longest'] = max(ok, key=lambda x: (len(x[1]['word']), x[1]['word']))
            picks['most_subtokens'] = max(ok, key=lambda x: (x[1]['n_sub'], x[1]['word']))
            single = [x for x in ok if x[1]['n_sub'] == 1]
            multi = [x for x in ok if x[1]['n_sub'] >= 2]
            if single:
                picks['single_token_example'] = single[0]
            if multi:
                picks['multi_token_example'] = multi[0]
        ex = {}
        for name, (u, t) in picks.items():
            a, b = max(0, t['t0'] - 6), min(len(u.input_ids), t['t1'] + 3)
            ex[name] = {'unit_id': u.unit_id, 'word': t['word'], 'target_token_span': [t['t0'], t['t1']],
                        'prediction_positions': list(range(t['t0'] - 1, t['t1'])),
                        'target_ids': t['target_ids'],
                        'target_tokens': [tok.decode([i], clean_up_tokenization_spaces=False) for i in t['target_ids']],
                        'decoded_target': t['decoded'], 'leading_ws_absorbed': t['leading_ws_absorbed'],
                        'boundary_token': t['boundary_text'], 'boundary_id': t['boundary_id'],
                        'window_ids': u.input_ids[a:b],
                        'window_tokens': [tok.decode([i], clean_up_tokenization_spaces=False)
                                          for i in u.input_ids[a:b]],
                        'window_start_index': a}
        first_ids = enc[0].input_ids[:8] if enc else []
        out['conditions'][cname] = {
            'n_targets': len(flat), 'n_aligned': len(ok), 'alignment_failures': reasons,
            'n_subtokens_distribution': {int(k): int(v) for k, v in nsub.items()},
            'n_leading_ws_absorbed': int(sum(t['leading_ws_absorbed'] for _, t in ok)),
            'offset_methods': pd.Series([t.get('offset_method') for _, t in flat]).value_counts().to_dict(),
            'boundary_tokens': pd.Series([t['boundary_text'] for _, t in ok]).value_counts().head(5).to_dict(),
            'first_input_tokens': [tok.decode([i], clean_up_tokenization_spaces=False) for i in first_ids],
            'bos_at_position_0': bool(first_ids and tok.bos_token_id is not None and first_ids[0] == tok.bos_token_id),
            'n_bos_tokens_in_first_unit': int(sum(i == tok.bos_token_id for i in enc[0].input_ids))
            if enc and tok.bos_token_id is not None else 0,
            'examples': ex}
    return out


def prompt_examples(sm: ScoringModel, cond: PromptCondition, encoded: list[EncodedUnit], units: list[Unit]) -> dict:
    u, e = units[0], encoded[0]
    return {'model': sm.spec.alias, 'condition': cond.name, 'protocol': cond.protocol, 'role': cond.role,
            'wording_source': cond.wording_source, 'status': cond.status,
            'template_sha256': cond.template_sha256 if cond.protocol != 'context' else None,
            'input_construction': ('INSTRUCT: tokenizer.apply_chat_template([{"role": "user", "content": <text>}], '
                                   'add_generation_prompt=True) — no system message added by this script; the '
                                   'template may insert its own default'
                                   if sm.chat else 'BASE: the complete text tokenised in one call '
                                                   '(add_special_tokens=True); no chat template'),
            'message_roles': ['user'] if sm.chat else None, 'add_generation_prompt': True if sm.chat else None,
            'chat_template_kwargs': sm.template_kwargs if sm.chat else None,
            'example_unit_id': u.unit_id, 'example_text_or_user_content': u.text,
            'example_rendered_model_input': e.full_text, 'example_rendered_sha256': sha256_text(e.full_text),
            'example_input_ids': e.input_ids, 'example_targets': e.targets,
            'n_units': len(units), 'all_rendered_sha256': sha256_json([x.full_text for x in encoded])}


def process_model(spec: ModelSpec, cfg, conds: dict, active: list, samples: Samples, context_df, dirs) -> dict:
    status = {'alias': spec.alias, 'repo_id': spec.repo_id, 'conditions': {}, 'status': 'OK'}
    t_start = time.time()
    sm = ScoringModel(spec, cfg, load_weights=False)
    words = list(samples.words_to_score['word'])
    plan = {}
    encoded_by = {}
    for cname in active:
        cond = conds[cname] if cname != 'context_file' else PromptCondition(
            'context_file', 'user-supplied contexts', 'context', 'context', 'user --context_file',
            cfg.context_info.get('status', 'exploratory'))
        units = build_units(cond, words, cfg, samples, context_df)
        encoded = [sm.encode_unit(u) for u in units]
        encoded_by[cname] = encoded
        flat = [t for u in encoded for t in u.targets]
        fail_rate = 1 - np.mean([t['ok'] for t in flat]) if flat else 1.0
        key = {'script_version': SCRIPT_VERSION, 'model_alias': spec.alias, 'repo_id': spec.repo_id,
               'requested_revision': spec.revision, 'resolved_revision': sm.resolved_revision,
               'variant': spec.variant, 'tokenizer': sm.tokenizer_fingerprint(), 'condition': cname,
               'protocol': cond.protocol,
               'template_sha256': cond.template_sha256 if cond.protocol != 'context' else cfg.context_sha256,
               'units_sha256': units_fingerprint(encoded), 'dtype': cfg.dtype, 'quantization': cfg.quantization,
               'attn_implementation_requested': cfg.attn_implementation, 'num_hidden_layers': sm.num_layers,
               'hidden_size': sm.hidden_size, 'lens_method': LENS_METHOD,
               'representations_saved': cfg.save_representations if cond.protocol in ('isolated', 'context') else 'none',
               'aggregate_definitions_sha256': sha256_json(AGGREGATE_DEFINITIONS)}
        akey = analysis_key(cfg, cond, units)
        cache = CacheEntry(dirs['root'], spec.alias, cname)
        st = {'protocol': cond.protocol, 'n_units': len(units), 'n_targets': len(flat),
              'alignment_failure_rate': float(fail_rate)}
        atomic_write_json(os.path.join(dirs['prompts'], f"{safe_name(spec.alias)}_{safe_name(cname)}_prompt_examples.json"),
                          prompt_examples(sm, cond, encoded, units))
        if fail_rate > cfg.max_alignment_failure_rate:
            st['status'] = (f'FAILED: target alignment failed for {fail_rate:.1%} of targets '
                            f'(> {cfg.max_alignment_failure_rate:.1%}); see tokenizer diagnostics')
            status['conditions'][cname] = st
            continue
        if cache.exists() and not cfg.overwrite:
            mism = cache.mismatches(key)
            if not mism:
                st['status'] = 'CACHED (manifest matches)'
                status['conditions'][cname] = st
                continue
            st['status'] = (f'FAILED: stale cache — manifest keys differ {mism}; refusing to reuse or overwrite '
                            f'(rerun with --overwrite)')
            status['conditions'][cname] = st
            continue
        plan[cname] = (cond, units, encoded, key, akey, cache, st)
    atomic_write_json(os.path.join(dirs['prompts'], f"{safe_name(spec.alias)}_tokenizer_diagnostics.json"),
                      tokenizer_diagnostics(sm, encoded_by))
    info_path = os.path.join(dirs['models'], f"{safe_name(spec.alias)}.json")
    if plan:
        sm.load_weights()
        prim = encoded_by.get(cfg.primary_condition) or next(iter(encoded_by.values()))
        sm.verify_readout(prim[:8])
        if sm.lens_ok:
            sm.check_padding(prim)
        else:
            sm.padding_check = {'status': 'SKIPPED', 'reason': 'logit lens unavailable'}
            logger.error(f"{spec.alias}: layerwise surprisal UNAVAILABLE — {sm.lens_check.get('reason')}")
        desc = sm.describe()
        atomic_write_json(info_path, desc)
        for cname, (cond, units, encoded, key, akey, cache, st) in plan.items():
            want_reps = cond.protocol in ('isolated', 'context') and cfg.save_representations != 'none'
            t0 = time.time()
            sc = sm.score_encoded(encoded, want_reps=want_reps, desc=f"{spec.alias}/{cname}")
            arrays = pack_scores(encoded, sc)
            reps = {}
            if want_reps:
                if cfg.save_representations in ('both', 'last'):
                    reps['last'] = sc['reps_last']
                if cfg.save_representations in ('both', 'mean'):
                    reps['mean'] = sc['reps_mean']
            cache.save(key, akey, {'model': desc, 'condition': asdict(cond) if hasattr(cond, '__dataclass_fields__')
                                   else {}, 'seconds': time.time() - t0,
                                   'lens_status': sm.lens_check.get('status'),
                                   'pooling_rules': {'last': 'hidden state at the LAST target subtoken position',
                                                     'mean': 'mean over all target subtoken positions'}},
                       arrays, reps)
            st['status'] = 'SCORED' if sm.lens_ok else 'SCORED (surprisal UNAVAILABLE; representations only)'
            st['seconds'] = round(time.time() - t0, 1)
            status['conditions'][cname] = st
            ss = load_scoreset(cache, cond, desc)
            write_csv(item_layer_frame(ss), os.path.join(dirs['item_scores'], safe_name(spec.alias),
                                                         f"{safe_name(cname)}_item_layer_scores.csv"))
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
    if not os.path.isdir(root):
        return found, skipped
    words = list(samples.words_to_score['word'])
    all_conds = dict(conds)
    if cfg.context_file:
        all_conds['context_file'] = PromptCondition('context_file', 'user-supplied contexts', 'context', 'context',
                                                    'user --context_file', cfg.context_info.get('status', 'exploratory'))
    for mdir in sorted(os.listdir(root)):
        for cdir in sorted(os.listdir(os.path.join(root, mdir))):
            if cdir.endswith('.partial'):
                continue
            ce = CacheEntry.__new__(CacheEntry)
            ce.dir = os.path.join(root, mdir, cdir)
            ce.manifest_path = os.path.join(ce.dir, 'manifest.json')
            if not ce.exists():
                continue
            man = ce.manifest()
            cname = man['key']['condition']
            if cname not in all_conds:
                skipped.append((mdir, cdir, 'condition not configured in this run'))
                continue
            cond = all_conds[cname]
            units = build_units(cond, words, cfg, samples, context_df)
            mism = ce.mismatches(analysis_key(cfg, cond, units), analysis_only=True)
            if mism:
                skipped.append((mdir, cdir, f'stale cache (keys {mism}); not analysed'))
                continue
            found.append(load_scoreset(ce, cond, man['info'].get('model', {'alias': man['key']['model_alias']})))
    return found, skipped


def _lens_available(ss: ScoreSet) -> bool:
    return bool(np.isfinite(ss.surprisal['sum'][:, 1:]).any())


def run_analyses(cfg, conds: dict, samples: Samples, context_df, dirs) -> dict:
    # Small cells (e.g. --max_items subsets) produce undefined SDs; those values
    # are NaN in the tables, so numpy's RuntimeWarnings are not informative here.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return _run_analyses(cfg, conds, samples, context_df, dirs)


def _run_analyses(cfg, conds: dict, samples: Samples, context_df, dirs) -> dict:
    sets, skipped = discover_scoresets(cfg, conds, samples, context_df, dirs)
    for s in skipped:
        logger.warning(f"analysis skipped {s}")
    res = {'skipped_caches': skipped, 'n_scoresets': len(sets), 'tables': {}, 'not_run': {}, 'figures': []}
    if not sets:
        res['not_run']['all_analyses'] = 'NOT_RUN: no valid cached model outputs'
        return res
    alpha, B = cfg.alpha, cfg.n_bootstrap
    s1_rows, s2_rows, depth_rows, inter_rows, inter_depth_rows = [], [], [], [], []
    dist_rows, probe_tabs, matched_tabs, list_tabs, dom_rows, cell_rows = [], [], [], [], [], []
    boot_store = {}
    matches = build_matches(samples.stage2[cfg.gcide_zero_policy], cfg.match_caliper_sd, cfg.min_matched_pairs) \
        if cfg.run_stage2 and cfg.run_matched else {}
    by_model = {}
    for ss in sets:
        by_model.setdefault(ss.alias, {})[ss.condition] = ss
    analysis_aggs = [a for a in ('sum', 'mean', 'sum_eow') if a == cfg.primary_aggregate] + \
                    [a for a in ('sum', 'mean', 'sum_eow') if a != cfg.primary_aggregate]
    for ss in sets:
        if not _lens_available(ss):
            res['not_run'][f'{ss.alias}/{ss.condition}/surprisal'] = 'UNAVAILABLE: logit lens failed verification'
            continue
        if ss.protocol == 'context':
            continue                       # handled in the context block below
        depth = ss.depth
        nsub = ss.targets[ss.targets['align_ok']].groupby('word')['n_sub'].mean()
        st1 = samples.stage1.assign(n_subtokens=samples.stage1['word'].map(nsub))
        st2 = {p: r.assign(n_subtokens=r['word'].map(nsub)) for p, r in samples.stage2.items()}
        for agg in analysis_aggs:
            W = ss.word_matrix(agg)
            ctx = ss.ctx(agg)
            primary_agg = agg == cfg.primary_aggregate
            if cfg.run_stage1:
                designs = [design_stage1_A(st1, 'regression'), design_stage1_A(st1, 'regression+n_subtokens'),
                           design_stage1_A(st1, 'explor_sim_top2')]
                for app in 'BC':
                    designs += [design_stage1_BC(st1, app, sp) for sp in
                                ('factorial', 'factorial+covariates', 'factorial+n_subtokens')]
                for d in designs:
                    d = d.restrict(W.index)
                    if d.n < len(d.terms) + 3:
                        continue
                    Y = W.loc[d.words].values
                    boot = None
                    if primary_agg and d.spec in ('regression', 'factorial'):
                        boot = boot_std_betas(d, [Y], B, cfg.seed)[0]
                        point = ols_multi(d.X_std(), zscore_cols(Y))['beta'][1:]
                        amb = ['ambiguity', 'wn_n_all'] if d.analysis == 'stage1_A' else ['ambiguity', 'senses']
                        depth_rows += depth_profile_tests(d, point, boot, depth, ctx, alpha,
                                                          'log_subtlex' if d.analysis == 'stage1_A' else None, amb)
                        boot_store[(ss.alias, ss.condition, d.analysis, d.spec, None)] = (d, boot)
                    s1_rows.append(fit_layerwise(d, Y, depth, ctx, alpha, boot))
                    if d.analysis in ('stage1_B', 'stage1_C') and d.spec == 'factorial':
                        lab = pd.Series(d.strata)
                        for cell in sorted(lab.unique()):
                            m = (lab == cell).values
                            mu = Y[m].mean(0)
                            sd = Y[m].std(0, ddof=1)
                            for l in range(Y.shape[1]):
                                cell_rows.append(dict(ctx, analysis=d.analysis, cell=cell, n=int(m.sum()),
                                                      layer_column=l + 1, depth=float(depth[l]),
                                                      mean_bits=mu[l], sd_bits=sd[l]))
            if cfg.run_stage2:
                for pol in GCIDE_ZERO_POLICIES:
                    specs = list(STAGE2_SPECS) if pol == cfg.gcide_zero_policy else ['primary']
                    for spec in specs:
                        d = design_stage2(st2[pol], spec, pol).restrict(W.index)
                        if d.n < len(d.terms) + 3:
                            continue
                        Y = W.loc[d.words].values
                        boot = None
                        if primary_agg and spec == 'primary' and pol == cfg.gcide_zero_policy:
                            boot = boot_std_betas(d, [Y], B, cfg.seed)[0]
                            point = ols_multi(d.X_std(), zscore_cols(Y))['beta'][1:]
                            depth_rows += depth_profile_tests(d, point, boot, depth, ctx, alpha, 'log_subtlex',
                                                              ['gcide_etymologies', 'wn_n_all'])
                            boot_store[(ss.alias, ss.condition, 'stage2', 'primary', pol)] = (d, boot)
                        s2_rows.append(fit_layerwise(d, Y, depth, ctx, alpha, boot))
            if primary_agg and cfg.run_stage2 and cfg.run_matched and ss.protocol == 'isolated' and matches:
                matched_tabs.append(matched_effects(matches, W, depth, ctx, alpha))
        if ss.protocol == 'list':
            list_tabs.append(list_order_diagnostics(ss, by_model[ss.alias].get(cfg.primary_condition),
                                                    cfg.primary_aggregate))
    # ── prompt / context interactions (paired by word) ──────────────────────
    agg = cfg.primary_aggregate
    for alias, cs in by_model.items():
        ref = cs.get(cfg.primary_condition)
        if ref is None or not _lens_available(ref):
            continue
        others = {n: s for n, s in cs.items() if n != cfg.primary_condition and s.protocol in ('isolated', 'context')
                  and _lens_available(s)}
        Wref = ref.word_matrix(agg)
        for oname, oss in others.items():
            if oss.protocol == 'context':
                t = oss.targets
                biased = (t['bias'].astype(str).str.lower() != 'neutral') if (t['bias'] != '').any() else \
                    pd.Series(True, index=t.index)
                ok = t['align_ok'] & biased
                M = pd.DataFrame(oss.surprisal[agg][ok.values][:, 1:], columns=range(1, oss.n_layers + 1))
                M['word'] = t.loc[ok, 'word'].values
                Woth = M.groupby('word').mean()
            else:
                Woth = oss.word_matrix(agg)
            common = Wref.index.intersection(Woth.index)
            ctx = dict(ref.ctx(agg), condition=oname, protocol=oss.protocol,
                       reference_condition=cfg.primary_condition, condition_status=oss.cond_status)
            designs = []
            if cfg.run_stage1:
                designs += [design_stage1_A(samples.stage1), design_stage1_BC(samples.stage1, 'B'),
                            design_stage1_BC(samples.stage1, 'C')]
            if cfg.run_stage2:
                designs.append(design_stage2(samples.stage2[cfg.gcide_zero_policy], 'primary', cfg.gcide_zero_policy))
            for d in designs:
                d = d.restrict(common)
                if d.n < len(d.terms) + 3:
                    continue
                Yr, Yo = Wref.loc[d.words].values, Woth.loc[d.words].values
                tab = fit_layerwise(d, Yo - Yr, ref.depth, ctx, alpha, None, include_intercept=True, center_raw=True)
                tab['term'] = tab['term'].replace({'(intercept)': '(condition main effect)'})
                tab['quantity'] = np.where(tab['term'] == '(condition main effect)',
                                           'mean difference condition − reference (bits) at mean predictors',
                                           'interaction: term × condition (bits per predictor unit)')
                inter_rows.append(tab)
                br, bo = boot_std_betas(d, [Yr, Yo], max(200, B // 2), cfg.seed)
                pr = ols_multi(d.X_std(), zscore_cols(Yr))['beta'][1:]
                po = ols_multi(d.X_std(), zscore_cols(Yo))['beta'][1:]
                dpt, dbt = np.abs(po) - np.abs(pr), np.abs(bo) - np.abs(br)
                sh, dp = region_mask(ref.depth, 'shallow'), region_mask(ref.depth, 'deep')
                for j, term in enumerate(d.terms):
                    if term not in ('log_subtlex', 'gcide_etymologies', 'wn_n_all', 'ambiguity', 'senses'):
                        continue
                    for region in ('shallow', 'deep'):
                        m = region_mask(ref.depth, region)
                        est = float(np.nanmean(dpt[j, m])) if m.any() else np.nan
                        bd = np.nanmean(dbt[:, j, m], -1) if m.any() else np.full(len(dbt), np.nan)
                        lo, hi = percentile_ci(bd, alpha)
                        inter_depth_rows.append(dict(ctx, analysis=d.analysis, spec=d.spec, term=term,
                                                     statistic=f'mean Δ|std β| in {region} region', estimate=est,
                                                     boot_ci_low=float(lo), boot_ci_high=float(hi),
                                                     boot_p=bootstrap_p_two_sided(bd), n=d.n, prediction=''))
                    if sh.any() and dp.any():
                        est = float(np.nanmean(dpt[j, dp]) - np.nanmean(dpt[j, sh]))
                        bd = np.nanmean(dbt[:, j, dp], -1) - np.nanmean(dbt[:, j, sh], -1)
                        lo, hi = percentile_ci(bd, alpha)
                        inter_depth_rows.append(dict(
                            ctx, analysis=d.analysis, spec=d.spec, term=term,
                            statistic='deep − shallow of Δ|std β| (layer × condition contrast)', estimate=est,
                            boot_ci_low=float(lo), boot_ci_high=float(hi), boot_p=bootstrap_p_two_sided(bd), n=d.n,
                            prediction='≈ 0 (frequency stays shallow)' if term == 'log_subtlex' else '> 0'))
                    for l in range(len(ref.depth)):
                        lo, hi = percentile_ci(dbt[:, j, l], alpha)
                        inter_depth_rows.append(dict(ctx, analysis=d.analysis, spec=d.spec, term=term,
                                                     statistic='Δ|std β| at layer', layer_column=l + 1,
                                                     depth=float(ref.depth[l]), estimate=float(dpt[j, l]),
                                                     boot_ci_low=float(lo), boot_ci_high=float(hi),
                                                     boot_p=np.nan, n=d.n, prediction=''))
            # ── representation distance between conditions (same word) ──────
            if cfg.run_distance:
                for pooling in ('last', 'mean'):
                    if pooling not in ref.reps or pooling not in oss.reps:
                        continue
                    ir, io = word_rep_index(ref), word_rep_index(oss)
                    if oss.protocol == 'context':
                        continue                    # cross-context distance handled below
                    cw = sorted(set(ir) & set(io))
                    if len(cw) < 10:
                        continue
                    Dm = cosine_distance_by_layer(ref.reps[pooling][[ir[w] for w in cw]],
                                                  oss.reps[pooling][[io[w] for w in cw]])[:, 1:]
                    Wd = pd.DataFrame(Dm, index=cw, columns=range(1, ref.n_layers + 1))
                    dctx = dict(ref.ctx(None), condition=oname, reference_condition=cfg.primary_condition,
                                pooling=pooling, measure='cross_task_frame_cosine_distance',
                                status='exploratory (same word under two task frames; NOT the usage distance of the slides)')
                    dist_rows += _distance_tables(Wd, dctx, samples, cfg, ref.depth)
    # ── user-supplied contexts: cross-context distance and dominance ────────
    for alias, cs in by_model.items():
        css = cs.get('context_file')
        if css is None:
            continue
        t = css.targets
        if cfg.run_distance:
            for pooling in ('last', 'mean'):
                if pooling not in css.reps:
                    continue
                per_word = {}
                for w, g in t[t['align_ok']].groupby('word'):
                    idx = g.index.values
                    bias = g['bias'].astype(str).values
                    pairs = [(i, j) for a, i in enumerate(idx) for b, j in enumerate(idx)
                             if a < b and (bias[a] != bias[b] or not (bias != '').any())]
                    if not pairs:
                        continue
                    A = css.reps[pooling][[p[0] for p in pairs]]
                    Bm = css.reps[pooling][[p[1] for p in pairs]]
                    per_word[w] = cosine_distance_by_layer(A, Bm)[:, 1:].mean(0)
                if len(per_word) >= 10:
                    Wd = pd.DataFrame(per_word).T
                    Wd.columns = range(1, css.n_layers + 1)
                    dctx = dict(css.ctx(None), pooling=pooling, measure='cross_context_cosine_distance',
                                status=cfg.context_info.get('status'))
                    dist_rows += _distance_tables(Wd, dctx, samples, cfg, css.depth)
                else:
                    res['not_run'][f'{alias}/context_distance/{pooling}'] = \
                        'NOT_RUN: fewer than 10 words with two differently-biased contexts'
        if cfg.context_info.get('has_dominance_labels') and _lens_available(css):
            dl = t['dominance'].astype(str).str.lower()
            ok = t['align_ok'].values
            M = css.surprisal[cfg.primary_aggregate][:, 1:]
            sub = pd.DataFrame(M[ok & (dl == 'subordinate').values]).assign(word=t.loc[ok & (dl == 'subordinate').values, 'word'].values).groupby('word').mean()
            dom = pd.DataFrame(M[ok & (dl == 'dominant').values]).assign(word=t.loc[ok & (dl == 'dominant').values, 'word'].values).groupby('word').mean()
            cw = sub.index.intersection(dom.index)
            if len(cw) >= 5:
                D = sub.loc[cw].values - dom.loc[cw].values
                n = len(D)
                mu, sd = D.mean(0), D.std(0, ddof=1)
                with np.errstate(invalid='ignore', divide='ignore'):
                    tt = mu / (sd / np.sqrt(n))
                p = p_two_sided_t(tt, n - 1)
                fdr = bh_fdr(p)
                for l in range(D.shape[1]):
                    dom_rows.append(dict(css.ctx(cfg.primary_aggregate), layer_column=l + 1,
                                         depth=float(css.depth[l]), n_words=n,
                                         mean_subordinate_minus_dominant_bits=mu[l], t=tt[l], p=p[l],
                                         p_fdr_layers=fdr[l]))
    # ── probes ───────────────────────────────────────────────────────────────
    if cfg.run_probes:
        for alias, cs in by_model.items():
            ss = cs.get(cfg.primary_condition)
            if ss is None or not ss.reps:
                continue
            probe_tabs += _run_probes(ss, samples, cfg)
    # ── assemble tables ──────────────────────────────────────────────────────
    T = res['tables']

    def cat(lst):
        lst = [x for x in lst if x is not None and len(x)]
        return pd.concat(lst, ignore_index=True) if lst else pd.DataFrame()
    s1 = add_global_fdr(cat(s1_rows))
    for app in 'ABC':
        T[f'stage1_published_appendix_{app}'] = s1[s1['analysis'] == f'stage1_{app}'] if len(s1) else pd.DataFrame()
    T['stage1_cell_means'] = pd.DataFrame(cell_rows)
    verd = stage1_verdicts(s1, alpha) if len(s1) else pd.DataFrame()
    T['stage1_calibration_verdicts'] = verd
    gate = stage1_gate_summary(verd) if len(verd) else pd.DataFrame()
    T['stage1_gate_summary'] = gate
    s2 = add_global_fdr(cat(s2_rows), ('spec', 'zero_policy', 'term'))
    if len(s2) and len(gate):
        s2 = s2.merge(gate[['model', 'condition', 'aggregate', 'gate']].rename(columns={'gate': 'stage1_gate'}),
                      on=['model', 'condition', 'aggregate'], how='left')
    T['stage2_wordnet_derived'] = s2
    coef = []
    for tbl in (s1, s2):
        if len(tbl):
            keep = tbl[tbl['term'].isin(['log_subtlex', 'gcide_etymologies', 'wn_n_all', 'ambiguity', 'senses'])]
            keep = keep[keep['spec'].isin(['regression', 'factorial', 'primary'])]
            coef.append(keep[['model', 'variant', 'family', 'condition', 'protocol', 'aggregate', 'analysis', 'spec',
                              'zero_policy', 'layer_column', 'block_index', 'depth', 'term', 'n', 'beta_std',
                              'ci_std_low', 'ci_std_high', 'boot_ci_std_low', 'boot_ci_std_high', 'beta', 'p_hc3',
                              'p_fdr_layers']])
    T['layerwise_frequency_ambiguity_coefficients'] = cat(coef)
    T['depth_profile_tests'] = pd.DataFrame(depth_rows)
    T['layerwise_prompt_interactions'] = add_global_fdr(cat(inter_rows))
    T['prompt_interaction_depth_tests'] = pd.DataFrame(inter_depth_rows)
    T['representation_distance'] = pd.DataFrame(dist_rows)
    T['probe_results'] = cat(probe_tabs)
    T['matched_robustness'] = cat(matched_tabs)
    T['list_protocol_order_diagnostics'] = cat(list_tabs)
    T['context_dominance'] = pd.DataFrame(dom_rows)
    if matches:
        T['matched_balance'] = pd.concat([m['balance'] for m in matches.values()], ignore_index=True)
        for name, m in matches.items():
            write_csv(m['pairs'], os.path.join(dirs['tables'], f'matched_items_{name}.csv'))
        res['matching'] = {k: v['info'] for k, v in matches.items()}
    T['multiple_testing'] = multiple_testing_summary(T, alpha)
    for name, df in T.items():
        if df is not None and len(df):
            write_csv(df, os.path.join(dirs['tables'], f'{name}.csv'))
    res['boot_store_keys'] = [list(map(str, k)) for k in boot_store]
    res['not_run'].update(not_run_registry(cfg, T, by_model))
    if cfg.run_figures:
        res['figures'] = make_figures(cfg, T, sets, samples, dirs)
    return res


def _distance_tables(Wd: pd.DataFrame, dctx: dict, samples: Samples, cfg, depth) -> list:
    rows = []
    for l, col in enumerate(Wd.columns):
        v = Wd[col].values
        rows.append(dict(dctx, analysis='summary', term='mean_distance', layer_column=int(col),
                         depth=float(depth[l]), estimate=float(np.nanmean(v)),
                         ci_low=float(np.nanmean(v) - 1.96 * np.nanstd(v, ddof=1) / np.sqrt(len(v))),
                         ci_high=float(np.nanmean(v) + 1.96 * np.nanstd(v, ddof=1) / np.sqrt(len(v))),
                         n=int(np.isfinite(v).sum())))
    designs = []
    if cfg.run_stage1:
        designs += [design_stage1_BC(samples.stage1, 'B'), design_stage1_BC(samples.stage1, 'C')]
    if cfg.run_stage2:
        designs.append(design_stage2(samples.stage2[cfg.gcide_zero_policy], 'primary', cfg.gcide_zero_policy))
    for d in designs:
        d = d.restrict(Wd.index)
        if d.n < len(d.terms) + 3:
            continue
        tab = fit_layerwise(d, Wd.loc[d.words].values, depth, dctx, cfg.alpha)
        tab['outcome'] = 'cosine distance'
        rows += tab.rename(columns={'beta': 'estimate'}).to_dict('records')
    return rows


def _run_probes(ss: ScoreSet, samples: Samples, cfg) -> list:
    tabs = []
    idx = word_rep_index(ss)
    nsub = ss.targets[ss.targets['align_ok']].groupby('word')['n_sub'].first()
    depth = ss.depth
    pub = samples.stage1.drop_duplicates('word')
    bc = samples.stage1[samples.stage1['appendix'].isin(['B', 'C'])]
    lab = bc.groupby('word').agg(amb=('ambiguity', 'nunique'), sen=('senses', 'nunique'))
    bcw = bc[bc['word'].isin(lab.index[(lab['amb'] == 1) & (lab['sen'] == 1)])].drop_duplicates('word')
    s2 = samples.stage2[cfg.gcide_zero_policy].drop_duplicates('word')
    targets = [('S1_BC_ambiguity', bcw, (bcw['ambiguity'] == 'ambiguous').astype(int), True, True),
               ('S1_BC_senses', bcw, (bcw['senses'] == 'many').astype(int), True, True),
               ('S1_published_log_subtlex', pub, pub['log_subtlex'], False, False),
               ('S2_log_subtlex', s2, s2['log_subtlex'], False, False),
               ('S2_wn_n_all', s2, s2['wn_n_all'], False, True),
               ('S2_gcide_etymologies', s2, s2['gcide_etymologies'], False, True)]
    for name, frame, y, binary, base_freq in targets:
        m = frame['word'].isin(idx.keys()).values & np.isfinite(np.asarray(y, float))
        if base_freq:
            m &= frame['log_subtlex'].notna().values
        fr = frame[m]
        yv = np.asarray(y, float)[m]
        if len(fr) < 4 * cfg.n_folds or (binary and min((yv == 1).sum(), (yv == 0).sum()) < cfg.n_folds):
            continue
        ridx = np.array([idx[w] for w in fr['word']])
        order = np.argsort(ridx)
        ridx, fr, yv = ridx[order], fr.iloc[order], yv[order]
        base_cols = [fr['length'].values, nsub.loc[fr['word']].values]
        if base_freq:
            base_cols.append(fr['log_subtlex'].values)
        baseX = np.column_stack(base_cols).astype(float)
        for pooling, R in ss.reps.items():
            ctx = dict(ss.ctx(None), status='secondary')
            tabs.append(run_probe_target(lambda k, R=R, ridx=ridx: np.asarray(R[ridx, k, :], np.float64), yv, binary,
                                         fr['word'].values, cfg, ctx, name, pooling, depth, baseX))
    return tabs


def multiple_testing_summary(T: dict, alpha: float) -> pd.DataFrame:
    rows = []
    specs = {'stage1_published_appendix_A': 'p_hc3', 'stage1_published_appendix_B': 'p_hc3',
             'stage1_published_appendix_C': 'p_hc3', 'stage2_wordnet_derived': 'p_hc3',
             'layerwise_prompt_interactions': 'p_hc3', 'probe_results': 'p_perm', 'matched_robustness': 'p'}
    for name, pcol in specs.items():
        df = T.get(name)
        if df is None or not len(df) or pcol not in df:
            continue
        fam = [c for c in ('model', 'condition', 'aggregate', 'analysis', 'spec', 'zero_policy', 'term', 'target',
                           'pooling', 'contrast') if c in df]
        for kv, g in df.groupby(fam, dropna=False, sort=False):
            rows.append(dict(zip(fam, kv if isinstance(kv, tuple) else (kv,)), table=name,
                             family='layers of one ' + ' × '.join(fam), n_tests=int(g[pcol].notna().sum()),
                             method='Benjamini–Hochberg', alpha=alpha,
                             n_significant_uncorrected=int((g[pcol] < alpha).sum()),
                             n_significant_fdr=int((g['p_fdr_layers'] < alpha).sum()) if 'p_fdr_layers' in g else np.nan,
                             n_significant_global_fdr=int((g['p_fdr_global_term'] < alpha).sum())
                             if 'p_fdr_global_term' in g else np.nan))
    return pd.DataFrame(rows)


def not_run_registry(cfg, T: dict, by_model: dict) -> dict:
    nr = {}
    if not cfg.context_file:
        nr['context_vs_isolated'] = ('NOT_RUN: no vetted meaning-biasing context sentences supplied '
                                     '(--context_file); none are generated')
        nr['usage_representation_distance'] = ('NOT_RUN: needs two or more usages/contexts per word '
                                               '(--context_file)')
    if not (cfg.context_file and cfg.context_info.get('has_dominance_labels')):
        nr['dominant_vs_subordinate'] = ('NOT_RUN: no dominance resource (eDom, Twilley et al. 1994, Gilbert & '
                                         'Rodd 2022) supplied; dominance labels are never fabricated')
    nr['nonword_foils_word_nonword_decision'] = ('NOT_RUN: the pool has no nonword foils and the specification '
                                                 'forbids inventing pseudowords')
    nr['decision_site_answer_readout'] = ('NOT_RUN: not implemented — the source response format echoes the item '
                                          'before WORD/NONWORD, and without nonwords a WORD/NONWORD margin is not a '
                                          'lexical decision')
    nr['behavioural_ease_generation'] = ('NOT_RUN: generated "ease" self-reports not implemented (the source says '
                                         'their relation to RT is untested)')
    nr['tuned_lens'] = 'NOT_RUN: only the logit lens is implemented (no trained translators)'
    if not cfg.run_list_protocol:
        nr['list_protocol'] = 'NOT_RUN: secondary list protocol disabled (enable with --run_list_protocol)'
    if not cfg.run_prompt_manipulations:
        nr['prompt_manipulations'] = 'NOT_RUN: disabled (--no-run_prompt_manipulations)'
    return nr


# ════════════════════════════════════════════════════════════════════════════
# FIGURES (drawn only from analyses that ran; never placeholders)
# ════════════════════════════════════════════════════════════════════════════
VIZ = {'surface': '#fcfcfb', 'text': '#0b0b0b', 'text2': '#52514e', 'grid': '#e6e5e0', 'zero': '#a3a29c',
       # categorical slots 1–3 of the reference palette (validated all-pairs); colour follows the ENTITY
       'frequency': '#2a78d6', 'meanings': '#eb6834', 'senses': '#1baf7a',
       # ordinal blue ramp (steps 250 / 450 / 650) for frequency tertiles
       'ramp': ['#86b6ef', '#2a78d6', '#104281']}
TERM_STYLE = {'log_subtlex': ('frequency (log_subtlex)', VIZ['frequency']),
              'gcide_etymologies': ('meanings (gcide_etymologies)', VIZ['meanings']),
              'ambiguity': ('meanings (ambiguity factor)', VIZ['meanings']),
              'wn_n_all': ('senses (wn_n_all)', VIZ['senses']),
              'senses': ('senses (few/many factor)', VIZ['senses'])}


def _plt():
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.facecolor': VIZ['surface'], 'axes.facecolor': VIZ['surface'],
                         'savefig.facecolor': VIZ['surface'], 'axes.edgecolor': VIZ['grid'],
                         'axes.labelcolor': VIZ['text2'], 'xtick.color': VIZ['text2'], 'ytick.color': VIZ['text2'],
                         'text.color': VIZ['text'], 'font.size': 8.5, 'axes.titlesize': 9,
                         'axes.titleweight': 'bold', 'legend.frameon': False, 'lines.linewidth': 1.6})
    return plt


def _grid(plt, n, w=3.3, h=2.5):
    ncols = min(4, max(1, n))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows + 0.9), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes.flat[:n]:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color=VIZ['grid'], linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 1.0)
    return fig, axes.flat


def _finish(fig, plt, path, title, subtitle, handles=None):
    import textwrap
    W, H = fig.get_figwidth(), fig.get_figheight()
    sub = '\n'.join(textwrap.wrap(subtitle, width=max(40, int(W * 15.5))))
    n_lines = sub.count('\n') + 1
    fig.text(0.01, 1 - 0.12 / H, title, ha='left', va='top', fontsize=11, fontweight='bold', color=VIZ['text'])
    fig.text(0.01, 1 - 0.42 / H, sub, ha='left', va='top', fontsize=8, color=VIZ['text2'], linespacing=1.35)
    top_in = 0.55 + 0.16 * n_lines
    bottom_in = 0.40 if handles else 0.05
    if handles:
        fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=8,
                   bbox_to_anchor=(0.5, 0.08 / H))
    fig.tight_layout(rect=(0, bottom_in / H, 1, 1 - top_in / H))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _line(ax, x, y, lo, hi, color, label):
    ax.plot(x, y, color=color, label=label)
    if lo is not None and np.isfinite(lo).any():
        ax.fill_between(x, lo, hi, color=color, alpha=0.13, linewidth=0)


def make_figures(cfg, T: dict, sets: list, samples: Samples, dirs: dict) -> list:
    plt = _plt()
    from matplotlib.lines import Line2D
    made = []
    pc, pa = cfg.primary_condition, cfg.primary_aggregate
    test_note = ' — TEST MODELS (random weights): NOT RESULTS' if any(
        s.model_info.get('test_model_random_weights') for s in sets) else ''
    sub = (f"condition={pc}, aggregate={pa}; surprisal in bits (higher = less expected); "
           f"depth=(block+1)/L; bands = 95% bootstrap CI over words{test_note}")
    s2 = T.get('stage2_wordnet_derived', pd.DataFrame())
    s2p = s2[(s2['condition'] == pc) & (s2['aggregate'] == pa) & (s2['spec'] == 'primary') &
             (s2['zero_policy'] == cfg.gcide_zero_policy)] if len(s2) else s2
    models = sorted(s2p['model'].unique()) if len(s2p) else []

    # 1 — raw layerwise surprisal by frequency tertile (Stage 2 pool)
    prim = [s for s in sets if s.condition == pc and s.protocol == 'isolated' and _lens_available(s)]
    if prim and cfg.run_stage2:
        rows = samples.stage2[cfg.gcide_zero_policy]
        q = rows['log_subtlex'].quantile([1 / 3, 2 / 3]).values
        tert = pd.cut(rows['log_subtlex'], [-np.inf, q[0], q[1], np.inf], labels=['low', 'mid', 'high'])
        fig, axes = _grid(plt, len(prim))
        for ax, ss in zip(axes, sorted(prim, key=lambda s: s.alias)):
            W = ss.word_matrix(pa)
            for i, lev in enumerate(['low', 'mid', 'high']):
                w = [x for x in rows.loc[tert == lev, 'word'] if x in W.index]
                if len(w) < 3:
                    continue
                M = W.loc[w].values
                mu, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(w))
                _line(ax, ss.depth, mu, mu - 1.96 * se, mu + 1.96 * se, VIZ['ramp'][i], lev)
            ax.set_title(ss.alias)
            ax.set_xlabel('normalised depth')
            ax.set_ylabel('mean surprisal (bits)')
        h = [Line2D([], [], color=VIZ['ramp'][i], label=f'{lev} frequency tertile (Stage 2 pool)')
             for i, lev in enumerate(['low', 'mid', 'high'])]
        p = os.path.join(dirs['figures'], 'layerwise_surprisal_profiles.png')
        _finish(fig, plt, p, 'Layerwise target surprisal by frequency tertile',
                sub.replace('bands = 95% bootstrap CI over words', 'bands = mean ± 1.96 SE over words'), h)
        made.append(p)

    def coef_fig(fname, title, terms, absval=False):
        if not models:
            return
        fig, axes = _grid(plt, len(models))
        for ax, m in zip(axes, models):
            g = s2p[s2p['model'] == m]
            for t in terms:
                gt = g[g['term'] == t].sort_values('depth')
                y = gt['beta_std'].values
                lo, hi = gt['boot_ci_std_low'].values, gt['boot_ci_std_high'].values
                if absval:
                    y = np.abs(y)
                    lo = hi = None
                _line(ax, gt['depth'].values, y, lo, hi, TERM_STYLE[t][1], TERM_STYLE[t][0])
            if absval:
                for x in (1 / 3, 2 / 3):
                    ax.axvline(x, color=VIZ['zero'], linewidth=0.6, linestyle=':')
            else:
                ax.axhline(0, color=VIZ['zero'], linewidth=0.8)
            ax.set_title(m)
            ax.set_xlabel('normalised depth')
            ax.set_ylabel('|standardised β|' if absval else 'standardised β')
        h = [Line2D([], [], color=TERM_STYLE[t][1], label=TERM_STYLE[t][0]) for t in terms]
        p = os.path.join(dirs['figures'], fname)
        _finish(fig, plt, p, title, sub + f'; Stage 2 primary spec, gcide zero policy={cfg.gcide_zero_policy}'
                + ('; dotted lines = shallow/middle/deep boundaries' if absval else ''), h)
        made.append(p)
    coef_fig('meanings_vs_senses_profiles.png', 'Meanings vs senses: Stage 2 layerwise coefficients',
             ['gcide_etymologies', 'wn_n_all'])
    coef_fig('frequency_vs_ambiguity_depth.png', 'Frequency vs ambiguity: effect magnitude across depth',
             ['log_subtlex', 'gcide_etymologies', 'wn_n_all'], absval=True)

    # 4 — Stage 1 factorial effects (Appendices B and C)
    s1 = pd.concat([T.get('stage1_published_appendix_B', pd.DataFrame()),
                    T.get('stage1_published_appendix_C', pd.DataFrame())], ignore_index=True)
    if len(s1):
        s1 = s1[(s1['condition'] == pc) & (s1['aggregate'] == pa) & (s1['spec'] == 'factorial')]
    if len(s1):
        ms = sorted(s1['model'].unique())
        fig, axes = plt.subplots(2, len(ms), figsize=(max(7.0, 3.3 * len(ms)), 6.0), squeeze=False)
        for j, m in enumerate(ms):
            for i, app in enumerate(['stage1_B', 'stage1_C']):
                ax = axes[i][j]
                ax.spines[['top', 'right']].set_visible(False)
                ax.grid(axis='y', color=VIZ['grid'], linewidth=0.6)
                g = s1[(s1['model'] == m) & (s1['analysis'] == app)]
                for t in ('ambiguity', 'senses'):
                    gt = g[g['term'] == t].sort_values('depth')
                    _line(ax, gt['depth'].values, gt['beta_std'].values, gt['boot_ci_std_low'].values,
                          gt['boot_ci_std_high'].values, TERM_STYLE[t][1], TERM_STYLE[t][0])
                ax.axhline(0, color=VIZ['zero'], linewidth=0.8)
                ax.set_xlim(0, 1)
                ax.set_title(f"{m}\nAppendix {app[-1]} · " + ('Exp. 2, visual' if app.endswith('B')
                                                                else 'Exp. 3, auditory (text input)'), fontsize=8)
                ax.set_xlabel('normalised depth')
                ax.set_ylabel('effect (SD of surprisal)')
        h = [Line2D([], [], color=TERM_STYLE[t][1], label=TERM_STYLE[t][0] + (' (ambiguous − unambiguous)' if t ==
             'ambiguity' else ' (many − few)')) for t in ('ambiguity', 'senses')]
        p = os.path.join(dirs['figures'], 'stage1_factorial_effects.png')
        _finish(fig, plt, p, 'Stage 1: meanings and senses in the published 2×2 designs',
                sub + '; human pattern: senses facilitatory (negative), ambiguity null (B) / inhibitory (C)', h)
        made.append(p)

    # 5 — cross-model heatmap of Stage 2 coefficients on a common depth grid
    if models:
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('div', ['#1c5cab', '#f0efec', '#c23434'])
        terms = ['log_subtlex', 'gcide_etymologies', 'wn_n_all']
        grid = np.linspace(0.05, 1.0, 20)
        mats = []
        for t in terms:
            M = []
            for m in models:
                gt = s2p[(s2p['model'] == m) & (s2p['term'] == t)].sort_values('depth')
                M.append(np.interp(grid, gt['depth'].values, gt['beta_std'].values,
                                   left=np.nan) if len(gt) else np.full(len(grid), np.nan))
            mats.append(np.array(M))
        vmax = np.nanmax([np.nanmax(np.abs(M)) for M in mats]) or 1.0
        fig, axes = plt.subplots(1, 4, figsize=(12, 0.45 * len(models) + 2.6), squeeze=False,
                                 gridspec_kw={'width_ratios': [1, 1, 1, 0.05]})
        for ax, t, M in zip(axes[0][:3], terms, mats):
            im = ax.imshow(M, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax, extent=(grid[0], 1.0, len(models), 0))
            ax.set_yticks(np.arange(len(models)) + 0.5)
            ax.set_yticklabels(models if t == terms[0] else [''] * len(models))
            ax.set_title(TERM_STYLE[t][0])
            ax.set_xlabel('normalised depth')
            ax.set_xlim(0, 1)
        fig.colorbar(im, cax=axes[0][3], label='standardised β (blue < 0 < red)')
        p = os.path.join(dirs['figures'], 'stage2_generalisation.png')
        _finish(fig, plt, p, 'Stage 2 generalisation (derived pool): coefficients across models',
                sub.replace('; bands = 95% bootstrap CI over words', '') +
                '; linear interpolation to a common depth grid; blank = depth not covered by that model')
        made.append(p)

    # 6 — layer × condition interaction (Δ|std β|), Stage 2 terms
    pid = T.get('prompt_interaction_depth_tests', pd.DataFrame())
    if len(pid):
        g0 = pid[(pid['statistic'] == 'Δ|std β| at layer') & (pid['analysis'] == 'stage2')]
        panels = sorted({(m, c) for m, c in zip(g0['model'], g0['condition'])})
        if panels:
            fig, axes = _grid(plt, len(panels))
            for ax, (m, c) in zip(axes, panels):
                g = g0[(g0['model'] == m) & (g0['condition'] == c)]
                for t in ('log_subtlex', 'gcide_etymologies', 'wn_n_all'):
                    gt = g[g['term'] == t].sort_values('depth')
                    _line(ax, gt['depth'].values, gt['estimate'].values, gt['boot_ci_low'].values,
                          gt['boot_ci_high'].values, TERM_STYLE[t][1], TERM_STYLE[t][0])
                ax.axhline(0, color=VIZ['zero'], linewidth=0.8)
                for x in (1 / 3, 2 / 3):
                    ax.axvline(x, color=VIZ['zero'], linewidth=0.6, linestyle=':')
                ax.set_title(f'{m}\n{c} vs {pc}', fontsize=8)
                ax.set_xlabel('normalised depth')
                ax.set_ylabel('Δ|standardised β|')
            h = [Line2D([], [], color=TERM_STYLE[t][1], label=TERM_STYLE[t][0])
                 for t in ('log_subtlex', 'gcide_etymologies', 'wn_n_all')]
            p = os.path.join(dirs['figures'], 'prompt_layer_interactions.png')
            _finish(fig, plt, p, 'Layer × condition: change in effect magnitude under another task frame',
                    f'Δ|β| = |β(condition)| − |β({pc})|, Stage 2 primary spec, paired bootstrap over words, '
                    f'aggregate={pa}{test_note}', h)
            made.append(p)
    return made


# ════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT, REPORT
# ════════════════════════════════════════════════════════════════════════════
def environment_snapshot() -> dict:
    from importlib import metadata as md
    env = {'python': sys.version, 'platform': platform.platform(), 'script': SCRIPT_NAME,
           'script_version': SCRIPT_VERSION, 'packages': {}, 'hf_hub_offline': os.environ.get('HF_HUB_OFFLINE'),
           'timestamp_utc': datetime.now(timezone.utc).isoformat()}
    for p in ('numpy', 'pandas', 'scipy', 'torch', 'transformers', 'tokenizers', 'accelerate', 'bitsandbytes',
              'huggingface_hub', 'safetensors', 'matplotlib', 'statsmodels', 'scikit-learn'):
        try:
            env['packages'][p] = md.version(p)
        except md.PackageNotFoundError:
            env['packages'][p] = None
    try:
        torch = _torch()
        env['cuda_available'] = torch.cuda.is_available()
        env['cuda_version'] = torch.version.cuda
        env['cudnn'] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
        env['gpus'] = [{'index': i, 'name': torch.cuda.get_device_name(i),
                        'capability': list(torch.cuda.get_device_capability(i)),
                        'memory_gb': round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1),
                        'bf16_supported': bool(torch.cuda.is_bf16_supported())}
                       for i in range(torch.cuda.device_count())]
    except Exception as e:                                           # noqa: BLE001
        env['torch_error'] = repr(e)
    return env


def _md_table(df: pd.DataFrame, cols=None, floatfmt='{:.3g}', max_rows=60) -> str:
    if df is None or not len(df):
        return '_(none)_\n'
    d = df[cols] if cols else df
    d = d.head(max_rows)

    def f(v):
        if isinstance(v, (float, np.floating)):
            return '' if not np.isfinite(v) else floatfmt.format(v)
        return str(v).replace('|', '\\|').replace('\n', ' ')
    lines = ['| ' + ' | '.join(map(str, d.columns)) + ' |', '|' + '---|' * len(d.columns)]
    lines += ['| ' + ' | '.join(f(v) for v in r) + ' |' for r in d.itertuples(index=False)]
    more = f'\n_({len(df) - max_rows} more rows in the CSV)_\n' if len(df) > max_rows else ''
    return '\n'.join(lines) + '\n' + more


def write_report(cfg, dirs, audit, env, conds, model_status, model_errors, res, plan, mode: str):
    L = []
    test = any(str(s.get('alias', '')).endswith('-TEST') for s in model_status.values())
    L.append(f"# Semantic ambiguity and word frequency across LLM layers — run report\n")
    L.append(f"*{SCRIPT_NAME} v{SCRIPT_VERSION} · {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · "
             f"mode: **{mode}***\n")
    banners = []
    if test:
        banners.append('**TEST MODELS WITH RANDOM WEIGHTS WERE USED. Every number below that comes from a '
                       '`*-TEST` model is a pipeline check, not a result.**')
    if audit['stage_counts']['subset_debug_run']:
        banners.append(f'**SUBSET RUN (`--max_items {cfg.max_items}`): debug only; sample sizes are not the '
                       f'study sample.**')
    if mode == 'audit_only':
        banners.append('**AUDIT ONLY: no model was loaded and no analysis was run.**')
    for b in banners:
        L.append(f"> {b}\n")
    L.append("## 1. Question and claim boundary\n")
    L.append("Do semantic ambiguity and word frequency have different layerwise profiles in decoder-only LLMs, "
             "and does the ambiguity profile separate multiple unrelated **meanings** (homonymy-like; "
             "`ambiguity` in Stage 1, `gcide_etymologies` in Stage 2) from multiple related **senses** "
             "(polysemy-like; `senses` in Stage 1, `wn_n_all` in Stage 2)? Are the profiles modulated by task "
             "instruction / context?\n")
    L.append("- **No measure here is a reaction time.** Layerwise surprisal (bits; higher = target less expected) "
             "is a model-derived quantity. Stage 1 compares only the qualitative *sign pattern* with Rodd et al. "
             "(2002), under the stated stand-in mapping (higher surprisal ≙ processing cost).\n"
             "- No nonword foils exist in the pool and none were generated, so no word/nonword decision is "
             "simulated; the lexical-decision prompt is a task frame for scoring the word.\n"
             "- Appendix C (auditory, Exp. 3) items are given to the model as text.\n"
             "- Stage 2 is a **generalisation analysis on a derived pool**, not a replication; its predictors are "
             "computed (GCIDE, WordNet), not normed.\n"
             "- A large final-layer frequency effect on isolated-word surprisal is expected by construction "
             "(the output distribution approximates word probability), so the depth comparison — not the "
             "existence of a frequency effect — is the informative test.\n")
    L.append("## 2. Data\n")
    f = audit['file']
    L.append(f"- File: `{f['path']}` — sha256 `{f['sha256']}` — {f['n_rows']} rows × {f['n_columns']} columns\n")
    L.append("### Sample funnel (recomputed from the CSV; nothing hard-coded)\n")
    L.append(_md_table(pd.DataFrame(audit['stage_counts']['funnel'])))
    gz = audit['gcide_etymologies_zero']
    L.append(f"\n**`gcide_etymologies == 0`** ({gz['stage2_pool_rows']} rows in the Stage 2 pool): "
             f"primary policy = `{gz['primary_policy']}`. {gz['policy_status']} n by policy: "
             f"{gz['n_stage2_by_policy']}.\n")
    L.append("\n**Screening:** " + '; '.join(f"{k}: {v['status']}" + (f" — {v.get('reason', v.get('rule', ''))}")
                                              for k, v in audit['screening'].items()) + "\n")
    L.append("\n### Source cross-checks (stated in the documents vs recomputed)\n")
    cc = pd.DataFrame(audit['source_cross_checks']['comparisons'])
    L.append(_md_table(cc[~cc['match']] if len(cc) else cc, max_rows=40) if len(cc) and (~cc['match']).any()
             else '_All stated numbers reproduced._\n')
    for n in audit['source_cross_checks']['notes']:
        L.append(f"- {n}\n")
    if audit.get('stage2_collinearity'):
        v = audit['stage2_collinearity'].get(cfg.gcide_zero_policy, {})
        if v:
            L.append(f"\nStage 2 VIF (policy `{cfg.gcide_zero_policy}`, n = {v['n']}): " +
                     ', '.join(f"{k} {x:.2f}" for k, x in v['vif'].items()) + "\n")
    L.append("\n## 3. Models and environment\n")
    L.append(f"Python {env['python'].split()[0]}; torch {env['packages'].get('torch')}; transformers "
             f"{env['packages'].get('transformers')}; tokenizers {env['packages'].get('tokenizers')}; "
             f"CUDA {env.get('cuda_version')}; GPUs: {[g['name'] for g in env.get('gpus', [])] or 'none'}; "
             f"dtype {cfg.dtype}; device {cfg.resolved_device}; device_map {cfg.device_map}; "
             f"quantization {cfg.quantization}.\n")
    rows = []
    for alias, st in model_status.items():
        info = {}
        p = os.path.join(dirs['models'], f"{safe_name(alias)}.json")
        if os.path.exists(p):
            info = read_json(p)
        rows.append({'model': alias, 'repo_id': st.get('repo_id'), 'revision': info.get('resolved_revision'),
                     'variant': info.get('variant'), 'layers': info.get('num_hidden_layers'),
                     'd': info.get('hidden_size'), 'status': st.get('status'),
                     'logit lens': (info.get('lens_verification') or {}).get('status'),
                     'last-layer |Δlogp| (nats)': (info.get('lens_verification') or {}).get(
                         'max_abs_diff_target_logprob_nats_last_layer'),
                     'padding check': (info.get('padding_check') or {}).get('status')})
    L.append(_md_table(pd.DataFrame(rows)))
    if model_errors:
        L.append("\n**Model failures** (full tracebacks in `logs/model_load_errors.json`):\n")
        for a, e in model_errors.items():
            L.append(f"- `{a}`: {e['error']}\n")
    L.append("\n## 4. Prompts, templates and target alignment\n")
    for c in conds.values():
        L.append(f"- `{c.name}` — protocol **{c.protocol}**, role {c.role}, status {c.status}; wording: "
                 f"{c.wording_source}; template sha256 `{c.template_sha256[:16]}…`\n")
    L.append("\nBASE models: the complete text is tokenised in one call (special tokens added by the tokenizer "
             "once; duplicated BOS is a hard error). INSTRUCT models: the tokenizer's own chat template "
             "(`apply_chat_template`, single user message, `add_generation_prompt=True`, no system message added "
             "here — templates may insert their own default); the rendered string re-tokenised with "
             "`add_special_tokens=False` must equal `apply_chat_template(tokenize=True)`, otherwise the model fails. "
             f"Templates that read `date_string` receive the fixed value `{cfg.chat_date_string}` so inputs do not "
             "depend on the run date. Each target's token span is located from character offsets; merges across "
             "the word boundary, gaps, or decode mismatches are alignment failures (excluded and counted). Target "
             "subtoken *t* is scored from position *t−1*; representations are read AT target positions. Rendered "
             "examples and per-model diagnostics are in `prompts/`.\n")
    al = []
    for alias, st in model_status.items():
        for cname, cs in st.get('conditions', {}).items():
            al.append({'model': alias, 'condition': cname, 'targets': cs.get('n_targets'),
                       'alignment failure rate': cs.get('alignment_failure_rate'), 'status': cs.get('status')})
    L.append(_md_table(pd.DataFrame(al)))
    L.append("\n## 5. Measure and analysis plan\n")
    L.append(f"{LENS_METHOD}. Aggregates: " + '; '.join(f"**{k}** = {v}" for k, v in AGGREGATE_DEFINITIONS.items())
             + f". Primary aggregate: `{cfg.primary_aggregate}`.\n\n")
    L.append("The analysis plan (criteria, regions, inference) is in `config.json → analysis_plan` "
             f"(sha256 `{sha256_json(plan)[:16]}…`). {plan['status']}\n")
    if mode == 'audit_only':
        atomic_write_text(os.path.join(dirs['root'], 'REPORT.md'), _md_spacing(''.join(L)))
        return
    T = res.get('tables', {})
    L.append("\n## 6. Results\n")
    g = T.get('stage1_gate_summary', pd.DataFrame())
    L.append("### 6.1 Stage 1 calibration (published items) — confirmatory for the primary condition/aggregate\n")
    L.append("PASS at a layer = preregistered pattern in Appendices A, B **and** C (signs, FDR significance, the "
             "Exp. 2 null, and no collapse of meanings and senses into one effect).\n\n")
    L.append(_md_table(g))
    s2 = T.get('stage2_wordnet_derived', pd.DataFrame())
    if len(s2):
        L.append("\n### 6.2 Stage 2 generalisation (derived pool) — primary spec, selected depths\n")
        sel = s2[(s2['spec'] == 'primary') & (s2['zero_policy'] == cfg.gcide_zero_policy) &
                 (s2['condition'] == cfg.primary_condition) & (s2['aggregate'] == cfg.primary_aggregate) &
                 s2['term'].isin(['gcide_etymologies', 'wn_n_all', 'log_subtlex'])].copy()
        pick = []
        for (m, t), gg in sel.groupby(['model', 'term']):
            for target in (0.25, 0.5, 0.75, 1.0):
                pick.append(gg.iloc[(gg['depth'] - target).abs().argsort()[:1]])
        if pick:
            tab = pd.concat(pick).drop_duplicates(['model', 'term', 'layer_column'])
            L.append(_md_table(tab, ['model', 'term', 'depth', 'n', 'beta_std', 'ci_std_low', 'ci_std_high',
                                     'p_hc3', 'p_fdr_layers', 'vif', 'stage1_gate'] if 'stage1_gate' in tab else
                               ['model', 'term', 'depth', 'n', 'beta_std', 'ci_std_low', 'ci_std_high', 'p_hc3',
                                'p_fdr_layers', 'vif'], max_rows=80))
    dt = T.get('depth_profile_tests', pd.DataFrame())
    if len(dt):
        L.append("\n### 6.3 Depth tests (preregistered statistics)\n")
        d0 = dt[(dt['condition'] == cfg.primary_condition) & (dt['statistic'].isin(
            ['deep_minus_shallow', 'crossing_contrast', 'delta_peak_depth', 'peak_depth']))]
        L.append(_md_table(d0, ['model', 'analysis', 'term', 'statistic', 'comparison', 'estimate', 'boot_ci_low',
                                'boot_ci_high', 'boot_p', 'prediction'], max_rows=80))
    pi = T.get('prompt_interaction_depth_tests', pd.DataFrame())
    if len(pi):
        L.append("\n### 6.4 Layer × condition (exploratory)\n")
        p0 = pi[pi['statistic'].str.startswith('deep − shallow')]
        L.append(_md_table(p0, ['model', 'condition', 'analysis', 'term', 'estimate', 'boot_ci_low',
                                'boot_ci_high', 'boot_p', 'prediction'], max_rows=60))
    pr = T.get('probe_results', pd.DataFrame())
    if len(pr):
        L.append("\n### 6.5 Probes (secondary; decodability ≠ use)\n")
        prv = pr.dropna(subset=['value'])
        best = prv.loc[prv.groupby(['model', 'target', 'pooling'])['value'].idxmax()] if len(prv) else prv
        L.append(_md_table(best, ['model', 'target', 'pooling', 'metric', 'depth', 'value', 'perm_null_q95',
                                  'p_perm', 'p_fdr_layers', 'baseline_value', 'n'], max_rows=60))
        L.append("_Max over layers is shown for orientation only; the per-layer table with FDR is the result._\n")
    mr = T.get('matched_robustness', pd.DataFrame())
    if len(mr):
        L.append("\n### 6.6 Matched robustness (Stage 2 pool)\n")
        L.append('Matching: ' + json.dumps(_clean_json(res.get('matching', {})), default=_json_default)[:1500] + '\n')
    L.append("\n## 7. NOT_RUN / UNAVAILABLE\n")
    for k, v in res.get('not_run', {}).items():
        L.append(f"- **{k}**: {v}\n")
    if res.get('skipped_caches'):
        L.append(f"- skipped caches: {res['skipped_caches']}\n")
    L.append("\n## 8. Confirmatory vs exploratory\n")
    L.append('Confirmatory: ' + '; '.join(plan['confirmatory']) + '.\n\nExploratory / secondary: ' +
             '; '.join(plan['exploratory']) + '.\n')
    L.append("\n## 9. Limitations\n")
    for lim in [
        'Surprisal and probe scores are model-derived measures, not reaction times; the stand-in mapping is an '
        'assumption under test.',
        'The logit lens is known to be unreliable in early layers of some families (Belrose et al., 2023, tuned '
        'lens); early-layer profiles may reflect readout mismatch rather than absent information.',
        'Isolated-word surprisal is dominated by word probability; a senses "facilitation" can also arise from '
        'contextual diversity (words with more senses occur in more contexts), which this design cannot separate.',
        'Summed (sequence) surprisal grows with the number of subtokens, and rarer words split into more '
        'subtokens, so part of any frequency effect on the sum can be tokenisation; see the n_subtokens '
        'robustness specs and the per-subtoken mean aggregate.',
        'Stage 1 cells were matched on CELEX frequency; SUBTLEX-based frequency is used here.',
        'gcide_etymologies is a derived measure (Webster 1913); zero values are handled by a configured policy '
        'that the source leaves unresolved.',
        'Inflected forms and emotional extremes were not screened unless a resource was supplied (the CSV has '
        'no field for either); plurals such as those in the gcide-zero list remain in the pool otherwise.',
        'The relatedness instruction is authored here; context, dominance and usage-distance analyses need '
        'vetted resources that were not supplied.',
        'Per-layer verdicts are many tests; FDR controls the significance criteria, but "any layer passes" is '
        'still lenient — the count of passing layers and the final-layer verdict are reported alongside.']:
        L.append(f"- {lim}\n")
    L.append("\n## 10. Files\n")
    for k in ('tables', 'figures', 'item_scores', 'prompts', 'cache', 'logs'):
        L.append(f"- `{k}/`\n")
    for fpath in res.get('figures', []):
        L.append(f"  - figure: `{os.path.relpath(fpath, dirs['root'])}`\n")
    atomic_write_text(os.path.join(dirs['root'], 'REPORT.md'), _md_spacing(''.join(L)))


def _md_spacing(text: str) -> str:
    """Blank lines around headings, block quotes and tables so every renderer parses them."""
    out = []
    prev = ''
    for line in text.split('\n'):
        starts_block = line.startswith(('#', '> ')) or (line.startswith('|') and not prev.startswith('|')) or \
            (line.startswith('- ') and not prev.startswith(('- ', '  ')))
        if starts_block and prev.strip():
            out.append('')
        if prev.startswith(('#', '> ')) and line.strip():
            out.append('')
        if prev.startswith('|') and line.strip() and not line.startswith('|'):
            out.append('')
        out.append(line)
        prev = line
    return '\n'.join(out)


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════
SCIENTIFIC_KEYS = ('data_sha256', 'seed', 'dtype', 'gcide_zero_policy', 'primary_aggregate', 'primary_condition',
                   'max_items', 'n_bootstrap', 'n_permutations', 'n_folds', 'probe_lambda_scale', 'alpha',
                   'list_size', 'list_repeats', 'context_sha256', 'task_prompts_sha256', 'chat_date_string',
                   'quantization', 'attn_implementation', 'match_caliper_sd', 'screening_sha256',
                   'save_representations', 'script_version')


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='Semantic ambiguity (meanings vs senses) and word frequency across '
                                             'LLM layers — layerwise logit-lens surprisal.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    B = argparse.BooleanOptionalAction
    g = ap.add_argument_group('data and output')
    g.add_argument('--data_csv', default=DEFAULT_DATA_CSV)
    g.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR)
    g.add_argument('--resume', action='store_true', help='reuse caches whose manifests match; add models')
    g.add_argument('--overwrite', action='store_true', help='recompute and replace existing outputs')
    g.add_argument('--max_items', type=int, default=None, help='SUBSET of unique words (debug/smoke only)')
    g.add_argument('--gcide_zero_policy', choices=GCIDE_ZERO_POLICIES, default='drop',
                   help='primary handling of gcide_etymologies == 0 (UNRESOLVED in the source; all policies are '
                        'reported as sensitivity analyses)')
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
                   help="'auto' shards across visible GPUs (needs accelerate); 'none' puts the model on --device")
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
    g.add_argument('--primary_aggregate', choices=['sum', 'mean', 'sum_eow'], default='sum')
    g.add_argument('--primary_condition', default='ld_isolated')
    g.add_argument('--task_prompts_json', default=None,
                   help='JSON {name: {template, protocol, role, wording_source, status}} adding task frames')
    g.add_argument('--context_file', default=None,
                   help='CSV word,context_id,text({TARGET}),bias,dominance,source,validation_status')
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
        cfg.n_bootstrap = min(cfg.n_bootstrap, 50)
        cfg.n_permutations = min(cfg.n_permutations, 20)
    cuda = False
    bf16_ok = False
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
    if cfg.device_map == 'auto' and cfg.resolved_device.startswith('cuda'):
        try:
            import accelerate  # noqa: F401
        except ImportError:
            logger.warning("accelerate not installed: --device_map auto unavailable, using a single device")
            cfg.device_map = 'none'
    cfg.context_sha256 = sha256_file(cfg.context_file) if cfg.context_file else None
    cfg.context_info = {}
    cfg.task_prompts_sha256 = sha256_file(cfg.task_prompts_json) if cfg.task_prompts_json else None
    scr = [cfg.screen_inflected_file, cfg.screen_valence_file]
    cfg.screening_sha256 = sha256_json([sha256_file(p) if p else None for p in scr] + [cfg.valence_extreme_sd])
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
    d['analysis_plan'] = plan
    d['analysis_plan_sha256'] = sha256_json(plan)
    d['conditions'] = {n: dict(asdict(c), template_sha256=c.template_sha256) for n, c in conds.items()}
    d['scientific_keys'] = {k: d.get(k) for k in SCIENTIFIC_KEYS}
    return d


def smoke_report(model_status: dict, dirs: dict) -> tuple[bool, list]:
    rows, ok_all = [], True
    for alias, st in model_status.items():
        info_p = os.path.join(dirs['models'], f"{safe_name(alias)}.json")
        info = read_json(info_p) if os.path.exists(info_p) else {}
        lv = info.get('lens_verification') or {}
        checks = [('tokenizer rendering + prompt examples written',
                   os.path.exists(os.path.join(dirs['prompts'], f"{safe_name(alias)}_tokenizer_diagnostics.json"))),
                  ('model loading', st.get('status') in ('OK', 'PARTIAL') and bool(info)),
                  ('target span alignment', all(c.get('alignment_failure_rate', 1) == 0
                                                for c in st.get('conditions', {}).values())),
                  ('hidden-state count / width', lv.get('n_hidden_state_tensors') == lv.get('expected_hidden_state_tensors')
                   and lv.get('hidden_size_observed') == info.get('hidden_size')),
                  ('logit lens reproduces model logits at last layer', lv.get('status') == 'OK'),
                  ('padding invariance', (info.get('padding_check') or {}).get('status') == 'PASS')]
        finite = False
        for cname in st.get('conditions', {}):
            ce = CacheEntry(dirs['root'], alias, cname)
            if ce.exists():
                _, arr, _ = ce.load()
                s = aggregate_surprisal(arr)['sum']
                finite |= bool(np.isfinite(s[:, -1]).any() and np.isfinite(s[:, 1]).any())
        checks.append(('one small layerwise score is finite', finite))
        for name, ok in checks:
            rows.append((alias, name, 'PASS' if ok else 'FAIL'))
            ok_all &= bool(ok)
    return ok_all, rows


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
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
    os.makedirs(cfg.output_dir, exist_ok=True)
    dirs = output_dirs(cfg.output_dir)
    setup_logging(dirs['logs'])
    logger.info(f"{SCRIPT_NAME} v{SCRIPT_VERSION} → {os.path.abspath(cfg.output_dir)}")
    seed_everything(cfg.seed)
    try:
        df, vrep = load_and_validate(cfg.data_csv)
    except CriticalError as e:
        logger.error(f"CRITICAL data validation failure: {e}")
        return 2
    cfg.data_sha256 = vrep['sha256']
    if exists and not cfg.overwrite:
        old = read_json(os.path.join(cfg.output_dir, 'config.json')).get('scientific_keys', {})
        new = {k: _clean_json(getattr(cfg, k, None)) for k in SCIENTIFIC_KEYS}
        diff = sorted(k for k in SCIENTIFIC_KEYS if _clean_json(old.get(k)) != new.get(k))
        if not (cfg.resume or cfg.analysis_only or cfg.audit_only):
            logger.error("output directory already holds a run; use --resume (same configuration) or --overwrite")
            return 1
        if diff:
            logger.error(f"--resume refused: configuration differs from the existing run in {diff}; use another "
                         f"--output_dir or --overwrite")
            return 1
    screening = load_screening_resources(cfg)
    samples = build_samples(df, cfg, screening)
    audit = data_audit(df, vrep, samples, cfg)
    atomic_write_json(os.path.join(dirs['root'], 'data_audit.json'), audit)
    write_csv(audit_to_flat(audit), os.path.join(dirs['root'], 'data_audit.csv'))
    env = environment_snapshot()
    atomic_write_json(os.path.join(dirs['root'], 'environment.json'), env)
    conds = build_conditions(cfg)
    context_df = None
    if cfg.context_file:
        context_df, cfg.context_info = load_context_file(cfg.context_file)
    plan = analysis_plan(cfg)
    atomic_write_json(os.path.join(dirs['root'], 'config.json'), config_record(cfg, plan, conds))
    for f in funnel_lines(samples):
        logger.info(f)
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
    active = active_conditions(cfg, conds)
    if not cfg.analysis_only:
        if not torch_available():
            logger.error("torch and transformers are required to score models")
            return 1
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
        prev = read_json(sp)
        prev.update(model_status)
        model_status = prev
    atomic_write_json(sp, model_status)
    if cfg.smoke_test:
        ok, rows = smoke_report(model_status, dirs)
        print('\nSMOKE TEST')
        for a, n, r in rows:
            print(f"  [{r}] {a}: {n}")
        print(f"SMOKE TEST {'PASSED' if ok else 'FAILED'}  (outputs: {os.path.abspath(cfg.output_dir)})")
        write_report(cfg, dirs, audit, env, conds, model_status, model_errors,
                     {'not_run': {'statistical_analyses': 'NOT_RUN in smoke mode'}}, plan, 'smoke_test')
        return 0 if ok and not model_errors else 4
    res = run_analyses(cfg, conds, samples, context_df, dirs)
    mode = 'analysis_only' if cfg.analysis_only else ('subset' if samples.subset else 'full')
    write_report(cfg, dirs, audit, env, conds, model_status, model_errors, res, plan, mode)
    atomic_write_json(os.path.join(dirs['root'], 'manifest.json'), {
        'mode': mode, 'script_version': SCRIPT_VERSION, 'data_sha256': cfg.data_sha256,
        'analysis_plan_sha256': sha256_json(plan), 'models': model_status, 'model_errors': list(model_errors),
        'n_scoresets_analysed': res.get('n_scoresets'), 'skipped_caches': res.get('skipped_caches'),
        'tables': sorted(k for k, v in res.get('tables', {}).items() if v is not None and len(v)),
        'figures': [os.path.relpath(p, dirs['root']) for p in res.get('figures', [])],
        'not_run': res.get('not_run')})
    logger.info(f"done → {os.path.abspath(cfg.output_dir)}/REPORT.md")
    return 3 if model_errors else 0


def funnel_lines(samples: Samples) -> list[str]:
    return [f"  [{r['stage']}] {r['step']}: {r['rows']} rows / {r['unique_words']} words" for r in samples.funnel]


# ════════════════════════════════════════════════════════════════════════════
# SELF-TESTS  (python semantic_ambiguity_layerwise.py --self_test)
# ════════════════════════════════════════════════════════════════════════════
def _test_cfg(**kw):
    a = build_parser().parse_args([])
    a.dtype, a.resolved_device, a.device_map = 'fp32', 'cpu', 'none'
    a.context_sha256, a.context_info, a.task_prompts_sha256 = None, {}, None
    a.screening_sha256, a.script_version = None, SCRIPT_VERSION
    a.n_bootstrap, a.n_permutations, a.batch_size = 40, 19, 4
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
    for app, n in (('A', 8), ('B', 8), ('C', 8)):
        for i in range(n):
            w = f"pub{app.lower()}" + ''.join(chr(97 + (i * 7 + j) % 26) for j in range(3))
            amb = 'ambiguous' if i % 2 == 0 else 'unambiguous'
            sen = '' if app == 'A' else ('many' if (i // 2) % 2 == 0 else 'few')
            sub = int(rng.integers(20, 5000))
            rows.append({'word': w, 'provenance': 'published', 'source_set': f'Rodd2002-{app}', 'pool_condition': '',
                         'appendix': app, 'experiment': {'A': 1, 'B': 2, 'C': 3}[app],
                         'modality': 'auditory' if app == 'C' else 'visual', 'ambiguity': amb, 'senses': sen,
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
    except ImportError:
        sm_api = None
    try:
        import sklearn  # noqa: F401
        have_sk = True
    except ImportError:
        have_sk = False
    tmp_root = tempfile.mkdtemp(prefix='semamb_selftest_')
    shared = {}

    def tiny(kind='llama', chat=False, **kw):
        variant = 'instruct' if chat else 'base'
        spec = ModelSpec(f'tiny-{kind}-TEST', f'local/{kind}', 'test', variant, test_model=kind)
        sm = ScoringModel(spec, _test_cfg(**kw))
        sm.verify_readout([sm.encode_unit(u) for u in build_units(
            build_conditions(_test_cfg())['ld_isolated'], ['bank', 'zygote', 'paper'], _test_cfg())])
        return sm

    class T(unittest.TestCase):
        # ── data: required tests 1–6 ────────────────────────────────────────
        def test_01_schema_validated(self):
            p = os.path.join(tmp_root, 's.csv')
            _synthetic_csv(p)
            df, rep = load_and_validate(p)
            self.assertFalse(rep['columns_missing'])
            self.assertTrue(all(v['pass'] for k, v in rep['checks'].items()
                                if k in ('row_id_unique', 'published_senses_values', 'provenance_values')))
            _synthetic_csv(p, lambda d: d.drop(columns=['wn_n_all']))
            with self.assertRaises(CriticalError):
                load_and_validate(p)
            _synthetic_csv(p, lambda d: d.assign(provenance=d['provenance'].replace('published', 'pub')))
            with self.assertRaises(CriticalError):
                load_and_validate(p)
            if csv:
                df, rep = load_and_validate(csv)
                self.assertEqual(set(EXPECTED_COLUMNS) - set(df.columns), set())

        def test_02_stages_disjoint_and_follow_provenance(self):
            if not csv:
                self.skipTest('ambiguity_items_combined.csv not found')
            df, _ = load_and_validate(csv)
            cfg = _test_cfg()
            s = build_samples(df, cfg, load_screening_resources(cfg))
            self.assertTrue((s.stage1['provenance'] == 'published').all())
            self.assertTrue((s.stage2_base['provenance'] == 'wordnet-derived').all())
            self.assertEqual(len(s.stage1), int((df['provenance'] == 'published').sum()))
            self.assertFalse(set(s.stage1['row_id']) & set(s.stage2_base['row_id']))
            self.assertFalse(set(s.stage1['word']) & set(s.stage2_base['word']))

        def test_03_repeated_words_keep_appendix_and_never_leak_across_folds(self):
            if csv:
                df, _ = load_and_validate(csv)
                pub = df[df['provenance'] == 'published']
                self.assertGreater(len(pub), pub['word'].nunique())
                multi = pub.groupby('word')['appendix'].nunique()
                w = multi.index[multi > 1][0]
                self.assertEqual(len(pub[pub['word'] == w]), multi[w])
            groups = np.array(['a', 'a', 'b', 'c', 'c', 'c', 'd', 'e', 'e', 'f', 'g', 'h'])
            for labels in (None, np.array([1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1])):
                folds = make_group_folds(groups, 3, 0, labels)
                for g in set(groups):
                    self.assertEqual(len(set(folds[groups == g])), 1)
            with self.assertRaises(ValueError):
                make_group_folds(np.array(['a', 'a']), 2, 0, np.array([0, 1]))

        def test_04_stage2_count_recomputed(self):
            p = os.path.join(tmp_root, 's4.csv')
            sdf = _synthetic_csv(p)
            df, _ = load_and_validate(p)
            cfg = _test_cfg()
            s = build_samples(df, cfg, load_screening_resources(cfg))
            d = sdf[(sdf['provenance'] == 'wordnet-derived')]
            manual = d[d['gcide_etymologies'].notna() & d['wn_n_all'].notna() & d['length'].between(4, 6) &
                       d['log_subtlex'].between(2.0, 4.0)]
            self.assertEqual(len(s.stage2_base), len(manual))
            if csv:
                df, _ = load_and_validate(csv)
                s = build_samples(df, cfg, load_screening_resources(cfg))
                d = df[df['provenance'] == 'wordnet-derived']
                manual = d[d['gcide_etymologies'].notna() & d['length'].between(4, 6) &
                           d['log_subtlex'].between(2.0, 4.0)]
                self.assertEqual(len(s.stage2_base), len(manual))

        def test_05_log_subtlex_not_relogged(self):
            p = os.path.join(tmp_root, 's5.csv')
            _synthetic_csv(p)
            df, rep = load_and_validate(p)
            self.assertTrue(rep['checks']['log_subtlex_is_log10_subtlex_plus_1']['pass'])
            cfg = _test_cfg()
            s = build_samples(df, cfg, load_screening_resources(cfg))
            d = design_stage2(s.stage2['drop'], 'primary', 'drop')
            exp = s.stage2['drop'].dropna(subset=STAGE2_SPECS['primary'][0])['log_subtlex'].values
            np.testing.assert_array_equal(d.raw['log_subtlex'].values, exp)
            _synthetic_csv(p, lambda x: x.assign(log_subtlex=np.log10(x['log_subtlex'] + 1)))
            with self.assertRaises(CriticalError):
                load_and_validate(p)

        def test_06_gcide_zero_policy_explicit(self):
            d = pd.DataFrame({'gcide_etymologies': [0, 1, 2, np.nan, 0]})
            a, ia = apply_gcide_policy(d, 'drop')
            self.assertEqual(list(a['gcide_etymologies'].fillna(-1)), [1, 2, -1])
            b, _ = apply_gcide_policy(d, 'recode_to_one')
            self.assertEqual(list(b['gcide_etymologies'].fillna(-1)), [1, 1, 2, -1, 1])
            c, _ = apply_gcide_policy(d, 'retain_as_zero')
            self.assertEqual(list(c['gcide_etymologies'].fillna(-1)), [0, 1, 2, -1, 0])
            self.assertEqual(ia['n_zero'], 2)
            with self.assertRaises(ValueError):
                apply_gcide_policy(d, 'impute')

        # ── tokenisation / model: required tests 7–11 ───────────────────────
        def test_07_base_full_text_and_instruct_chat_template(self):
            if not have_torch:
                self.skipTest('torch/transformers not installed')
            cond = build_conditions(_test_cfg())['ld_isolated']
            u = build_units(cond, ['banking'], _test_cfg())[0]
            sm = tiny('llama')
            e = sm.encode_unit(u)
            self.assertEqual(e.input_ids, sm.tokenizer(u.text)['input_ids'])
            self.assertEqual(e.input_ids[0], sm.tokenizer.bos_token_id)
            self.assertEqual(e.input_ids.count(sm.tokenizer.bos_token_id), 1)
            q = tiny('qwen2')
            self.assertIsNone(q.tokenizer.bos_token_id)
            self.assertEqual(q.encode_unit(u).input_ids, q.tokenizer(u.text)['input_ids'])
            c = tiny('llama_chat', chat=True)
            ec = c.encode_unit(u)
            ref = _as_id_list(c.tokenizer.apply_chat_template([{'role': 'user', 'content': u.text}], tokenize=True,
                                                              add_generation_prompt=True, **c.template_kwargs))
            self.assertEqual(ec.input_ids, ref)
            self.assertIn('<|user|>', ec.full_text)
            self.assertIn('26 Jul 2024', ec.full_text)          # fixed date_string, not the run date
            self.assertEqual(ec.input_ids.count(c.tokenizer.bos_token_id), 1)
            self.assertTrue(ec.targets[0]['ok'])
            bad = ModelSpec('nochat', 'local/llama', 'test', 'instruct', test_model='llama')
            with self.assertRaises(ChatTemplateError):
                ScoringModel(bad, _test_cfg(), load_weights=False)

        def test_08_target_and_prediction_positions(self):
            if not have_torch:
                self.skipTest('torch/transformers not installed')
            torch = _torch()
            cond = build_conditions(_test_cfg())['ld_isolated']
            words = ['bank', 'zygote', 'quixotic', 'a', 'paper']
            for kind, chat in (('llama', False), ('qwen2', False), ('llama_chat', True)):
                sm = tiny(kind, chat)
                enc = [sm.encode_unit(u) for u in build_units(cond, words, _test_cfg())]
                nsub = [e.targets[0]['n_sub'] for e in enc]
                self.assertTrue(all(e.targets[0]['ok'] for e in enc), [e.targets[0]['reason'] for e in enc])
                self.assertIn(1, nsub)
                self.assertTrue(any(n >= 2 for n in nsub))
                for e, w in zip(enc, words):
                    t = e.targets[0]
                    self.assertEqual(sm.tokenizer.decode(e.input_ids[t['t0']:t['t1'] + 1]).strip(), w)
                sc = sm.score_encoded(enc, want_reps=False, batch_size=3)
                agg = aggregate_surprisal(pack_scores(enc, sc))
                with torch.inference_mode():
                    for i, e in enumerate(enc):
                        ids = torch.tensor([e.input_ids])
                        lp = torch.log_softmax(sm.model(input_ids=ids).logits.float()[0], -1)
                        t = e.targets[0]
                        manual = -sum(float(lp[p - 1, e.input_ids[p]]) for p in range(t['t0'], t['t1'] + 1)) / LN2
                        self.assertAlmostEqual(agg['sum'][i, -1], manual, places=3)
                        first = -float(lp[t['t0'] - 1, e.input_ids[t['t0']]]) / LN2
                        self.assertAlmostEqual(agg['first'][i, -1], first, places=3)
            text = 'Items:\nbank\n'
            offs = [(0, 5), (5, 6), (6, 7), (7, 11), (11, 12)]
            dec = lambda ids: 'bank'
            self.assertTrue(align_target(offs, [0, 1, 2, 3, 4], text, (7, 11), 'bank', lambda x: 'bank')['ok'])
            self.assertEqual(align_target([(0, 5), (5, 8), (8, 11), (11, 12)], [0, 1, 2, 3], text, (7, 11),
                                          'bank', dec)['reason'], 'prefix_merged_into_target_token')
            nl = align_target([(0, 5), (5, 6), (6, 11), (11, 12)], [0, 1, 2, 3], text, (7, 11), 'bank', dec)
            self.assertTrue(nl['ok'] and nl['leading_ws_absorbed'])     # '\nbank' token: whitespace only
            self.assertEqual(align_target([(0, 5), (5, 6), (6, 7), (7, 12)], [0, 1, 2, 3], text, (7, 11), 'bank',
                                          dec)['reason'], 'suffix_merged_into_target_token')
            for kind, chat in (('llama', False), ('qwen2', False), ('llama_chat', True)):
                sm = tiny(kind, chat)
                units = build_units(cond, words, _test_cfg())
                a = [sm.encode_unit(u).targets[0] for u in units]
                self.assertEqual(sm.offset_method, 'offset_mapping')
                sm._force_decode_offsets = True
                b = [sm.encode_unit(u).targets[0] for u in units]
                self.assertEqual(sm.offset_method, 'decode_fallback')
                self.assertEqual([(x['t0'], x['t1'], x['ok']) for x in a], [(x['t0'], x['t1'], x['ok']) for x in b])
            ws = align_target([(0, 5), (5, 6), (6, 11)], [0, 1, 2], 'Items: bank', (7, 11), 'bank', lambda x: ' bank')
            self.assertTrue(ws['ok'] and ws['leading_ws_absorbed'])

        def test_09_lens_uses_final_norm_not_raw_states(self):
            if not have_torch:
                self.skipTest('torch/transformers not installed')
            torch = _torch()
            sm = tiny('llama')
            self.assertEqual(sm.lens_check['status'], 'OK')
            self.assertTrue(sm.last_is_post_norm)
            e = sm.encode_unit(build_units(build_conditions(_test_cfg())['ld_isolated'], ['bank'], _test_cfg())[0])
            ids, am = sm._pad([e])
            with torch.inference_mode():
                hs = sm._hidden_states(ids, am)
                h = hs[2][0, :5]
                lens = sm.lens_logprobs(h, 2)
                manual = torch.log_softmax(sm.head(sm.final_norm(h)).float(), -1)
                raw = torch.log_softmax(sm.head(h).float(), -1)
                self.assertLess(float((lens - manual).abs().max()), 1e-5)
                self.assertGreater(float((lens - raw).abs().max()), 1e-3)
                last = sm.lens_logprobs(hs[-1][0, :5], sm.num_layers)
                ref = torch.log_softmax(sm.model(input_ids=ids, attention_mask=am).logits[0, :5].float(), -1)
                self.assertLess(float((last - ref).abs().max()), 1e-4)

        def test_10_padding_never_shifts_targets(self):
            if not have_torch:
                self.skipTest('torch/transformers not installed')
            for kind, chat in (('llama', False), ('llama_chat', True)):
                sm = tiny(kind, chat)
                cond = build_conditions(_test_cfg())['ld_isolated']
                enc = [sm.encode_unit(u) for u in build_units(cond, ['a', 'quixotically', 'bank', 'zygote'],
                                                              _test_cfg())]
                self.assertGreater(len({len(e.input_ids) for e in enc}), 1)
                st = sm.check_padding(enc)
                self.assertEqual(st['status'], 'PASS', st)
                ids, am = sm._pad(enc)
                for b, e in enumerate(enc):
                    t = e.targets[0]
                    self.assertTrue(bool(am[b, t['t0'] - 1:t['t1'] + 1].all()))
                    self.assertEqual(ids[b, t['t0']:t['t1'] + 1].tolist(), t['target_ids'])

        def test_11_hidden_state_count_and_width(self):
            if not have_torch:
                self.skipTest('torch/transformers not installed')
            torch = _torch()
            for kind in ('llama', 'qwen2'):
                sm = tiny(kind)
                e = sm.encode_unit(build_units(build_conditions(_test_cfg())['ld_isolated'], ['bank'], _test_cfg())[0])
                with torch.inference_mode():
                    hs = sm._hidden_states(*sm._pad([e]))
                self.assertEqual(len(hs), sm.config.num_hidden_layers + 1)
                self.assertEqual(hs[-1].shape[-1], sm.config.hidden_size)
                self.assertEqual(sm.lens_check['n_hidden_state_tensors'], sm.num_layers + 1)

        # ── cache, NOT_RUN and tables: required tests 12–14 ────────────────
        def test_12_cache_mismatch_refused_and_end_to_end(self):
            if not (have_torch and csv):
                self.skipTest('needs torch/transformers and the CSV')
            out = os.path.join(tmp_root, 'run')
            cfg = _test_cfg(output_dir=out, max_items=60, run_list_protocol=True, list_size=5, list_repeats=2)
            dirs = output_dirs(out)
            df, _ = load_and_validate(csv)
            samples = build_samples(df, cfg, load_screening_resources(cfg))
            conds = build_conditions(cfg)
            active = active_conditions(cfg, conds)
            spec = TEST_MODEL_REGISTRY[0]
            st = process_model(spec, cfg, conds, active, samples, None, dirs)
            self.assertTrue(all(c['status'].startswith('SCORED') for c in st['conditions'].values()), st)
            st2 = process_model(spec, cfg, conds, active, samples, None, dirs)
            self.assertTrue(all(c['status'].startswith('CACHED') for c in st2['conditions'].values()), st2)
            ce = CacheEntry(out, spec.alias, 'ld_isolated')
            man = ce.manifest()
            man['key']['dtype'] = 'bf16'
            atomic_write_json(ce.manifest_path, man)
            st3 = process_model(spec, cfg, conds, active, samples, None, dirs)
            self.assertIn('stale cache', st3['conditions']['ld_isolated']['status'])
            self.assertIn('dtype', ce.mismatches(man['key'] | {'dtype': 'fp32'}))
            cfg.overwrite = True
            st4 = process_model(spec, cfg, conds, active, samples, None, dirs)
            self.assertTrue(st4['conditions']['ld_isolated']['status'].startswith('SCORED'))
            cfg.overwrite = False
            process_model(TEST_MODEL_REGISTRY[2], cfg, conds, active, samples, None, dirs)
            shared.update(cfg=cfg, dirs=dirs, samples=samples, conds=conds)
            item = pd.read_csv(os.path.join(dirs['item_scores'], safe_name(spec.alias), 'ld_isolated_item_layer_scores.csv'))
            self.assertEqual(set(item['layer_kind']), {'embedding_output', 'block_output'})
            self.assertEqual(item.groupby('word').size().iloc[0], 5)       # 4 blocks + embedding

        def test_13_missing_resources_are_not_run(self):
            cfg = _test_cfg()
            nr = not_run_registry(cfg, {}, {})
            for k in ('context_vs_isolated', 'dominant_vs_subordinate', 'usage_representation_distance'):
                self.assertTrue(nr[k].startswith('NOT_RUN'))
            p = os.path.join(tmp_root, 'ctx.csv')
            pd.DataFrame({'word': ['bank'], 'context_id': ['c1'], 'text': ['the {TARGET} by the river']}).to_csv(p, index=False)
            t, info = load_context_file(p)
            self.assertFalse(info['has_dominance_labels'])
            self.assertIn('exploratory', info['status'])
            cfg2 = _test_cfg(context_file=p, context_info=info)
            self.assertTrue(not_run_registry(cfg2, {}, {})['dominant_vs_subordinate'].startswith('NOT_RUN'))
            pd.DataFrame({'word': ['bank'], 'context_id': ['c1'], 'text': ['an ambiguous {TARGET}']}).to_csv(p, index=False)
            with self.assertRaises(CriticalError):
                load_context_file(p)
            res = load_screening_resources(cfg)
            self.assertEqual(res['inflected']['status'], 'NOT_AVAILABLE')
            self.assertEqual(res['emotional_extremes']['status'], 'NOT_AVAILABLE')

        def test_14_analysis_tables_carry_n_and_multiple_testing(self):
            if 'cfg' not in shared:
                self.skipTest('depends on test_12 (torch + CSV)')
            cfg, dirs, samples, conds = shared['cfg'], shared['dirs'], shared['samples'], shared['conds']
            res = run_analyses(cfg, conds, samples, None, dirs)
            T = res['tables']
            for name in ('stage1_published_appendix_A', 'stage1_published_appendix_B', 'stage1_published_appendix_C',
                         'stage2_wordnet_derived'):
                self.assertTrue(len(T[name]), name)
                for col in ('n', 'p_hc3', 'p_fdr_layers', 'p_fdr_global_term', 'beta_std', 'ci_low', 'ci_high'):
                    self.assertIn(col, T[name].columns, (name, col))
            self.assertIn('vif', T['stage2_wordnet_derived'])
            self.assertTrue(len(T['multiple_testing']))
            self.assertTrue(len(T['stage1_calibration_verdicts']))
            self.assertTrue(len(T['list_protocol_order_diagnostics']))
            self.assertTrue((T['stage2_wordnet_derived']['n'] > 0).all())

        # ── statistics ─────────────────────────────────────────────────────
        def test_ols_hc3_against_statsmodels(self):
            if sm_api is None:
                self.skipTest('statsmodels not installed')
            r = np.random.default_rng(1)
            X = np.column_stack([np.ones(200), r.normal(size=(200, 3))])
            Y = X @ r.normal(size=(4, 2)) + r.normal(size=(200, 2)) * (1 + np.abs(X[:, [1]]))
            f = ols_multi(X, Y)
            for l in range(2):
                ref = sm_api.OLS(Y[:, l], X).fit()
                np.testing.assert_allclose(f['beta'][:, l], ref.params, rtol=1e-8)
                np.testing.assert_allclose(f['se'][:, l], ref.bse, rtol=1e-8)
                np.testing.assert_allclose(f['se_hc3'][:, l], ref.get_robustcov_results('HC3').bse, rtol=1e-6)

        def test_bh_and_vif(self):
            p = np.array([0.01, 0.04, 0.03, np.nan, 0.5, 0.001])
            if sm_api is not None:
                from statsmodels.stats.multitest import multipletests
                ok = np.isfinite(p)
                np.testing.assert_allclose(bh_fdr(p)[ok], multipletests(p[ok], method='fdr_bh')[1])
            self.assertTrue(np.isnan(bh_fdr(p)[3]))
            r = np.random.default_rng(2)
            Z = r.normal(size=(500, 3))
            v = vif_table(Z, ['a', 'b', 'c'])
            self.assertTrue(all(0.98 < x < 1.05 for x in v.values()))
            Z[:, 2] = Z[:, 0] + 0.05 * r.normal(size=500)
            self.assertGreater(vif_table(Z, ['a', 'b', 'c'])['c'], 50)

        def test_ridge_probe_equals_sklearn_and_auroc(self):
            r = np.random.default_rng(3)
            X = r.normal(size=(120, 30)) * r.uniform(0.5, 3, 30)
            y = X[:, 0] - X[:, 1] + r.normal(size=120)
            folds = np.arange(120) % 4
            pred = ridge_oof(X, y, folds, 0.5)[:, 0]
            if have_sk:
                from sklearn.linear_model import Ridge
                from sklearn.metrics import roc_auc_score
                for f in range(4):
                    tr, te = folds != f, folds == f
                    mu, sd = X[tr].mean(0), X[tr].std(0)
                    Xs = (X[tr] - mu) / sd
                    lam = 0.5 * np.trace(Xs @ Xs.T) / tr.sum()
                    ref = Ridge(alpha=lam).fit(Xs, y[tr]).predict((X[te] - mu) / sd)
                    np.testing.assert_allclose(pred[te], ref, rtol=1e-6, atol=1e-8)
                lab = (y > 0).astype(int)
                self.assertAlmostEqual(auroc_columns(pred, lab)[0], roc_auc_score(lab, pred), places=10)
            self.assertGreater(r2_columns(pred, y)[0], 0.3)

        def test_stage1_verdict_logic(self):
            def tbl(bA, bS, pA, pS, app):
                rows = []
                names = {'stage1_A': ('ambiguity', 'wn_n_all'), 'stage1_B': ('ambiguity', 'senses'),
                         'stage1_C': ('ambiguity', 'senses')}[app]
                for term, b, p in ((names[0], bA, pA), (names[1], bS, pS)):
                    rows.append({'model': 'm', 'condition': 'c', 'aggregate': 'sum', 'layer_column': 1, 'depth': 1.0,
                                 'analysis': app, 'spec': 'regression' if app == 'stage1_A' else 'factorial',
                                 'term': term, 'beta': b, 'beta_std': b, 'p_hc3': p, 'p_fdr_layers': p})
                return rows
            good = pd.DataFrame(tbl(1, -1, .001, .001, 'stage1_A') + tbl(.1, -1, .5, .001, 'stage1_B') +
                                tbl(1, -1, .001, .001, 'stage1_C'))
            v = stage1_verdicts(good, 0.05)
            self.assertTrue(v['stage1_full_pass'].iloc[0])
            null_violated = pd.DataFrame(tbl(1, -1, .001, .001, 'stage1_A') + tbl(1, -1, .001, .001, 'stage1_B') +
                                         tbl(1, -1, .001, .001, 'stage1_C'))
            self.assertFalse(stage1_verdicts(null_violated, 0.05)['stage1_full_pass'].iloc[0])
            collapsed = pd.DataFrame(tbl(1, 1, .001, .001, 'stage1_A') + tbl(.1, -1, .5, .001, 'stage1_B') +
                                     tbl(1, 1, .001, .001, 'stage1_C'))
            v = stage1_verdicts(collapsed, 0.05)
            self.assertTrue(v['stage1_C_collapsed'].iloc[0])
            self.assertFalse(v['stage1_full_pass'].iloc[0])

        def test_depth_statistics(self):
            depth = np.arange(1, 13) / 12
            freq = np.exp(-depth * 4)
            amb = depth ** 2
            st = depth_stats(np.vstack([freq, amb]), depth)
            self.assertLess(st['deep_minus_shallow'][0], 0)
            self.assertGreater(st['deep_minus_shallow'][1], 0)
            self.assertLess(st['peak_depth'][0], st['peak_depth'][1])
            self.assertEqual(region_mask(depth, 'shallow').sum() + region_mask(depth, 'middle').sum()
                             + region_mask(depth, 'deep').sum(), 12)

        def test_prompts_and_leak_guard(self):
            conds = build_conditions(_test_cfg())
            self.assertEqual(conds['ld_isolated'].template.replace(ITEM_SLOT, '[item list]'),
                             SOURCE_LD_TEMPLATE.replace(ITEM_SLOT, '[item list]'))
            for c in conds.values():
                text, spans = render_items(c.template, ['bank', 'paper'])
                self.assertEqual([text[a:b] for a, b in spans], ['bank', 'paper'])
                self.assertIsNone(TEMPLATE_LEAK_RE.search(text))
            with self.assertRaises(CriticalError):
                check_template('x', 'Rate how ambiguous this is.\n' + ITEM_SLOT)
            with self.assertRaises(CriticalError):
                check_template('x', ITEM_SLOT + ' trailing')

        def test_matching_balances_covariates(self):
            r = np.random.default_rng(4)
            n = 400
            rows = pd.DataFrame({'word': [f'w{i}' for i in range(n)], 'row_id': [f'id{i}' for i in range(n)],
                                 'gcide_etymologies': r.choice([1, 1, 1, 2, 3], n).astype(float),
                                 'wn_n_all': r.integers(1, 15, n).astype(float),
                                 'log_subtlex': r.uniform(2, 4, n), 'length': r.integers(4, 7, n).astype(float),
                                 'concreteness': r.uniform(1.5, 5, n), 'ortho_n': r.integers(0, 8, n).astype(float)})
            m = build_matches(rows, 0.2, 10)
            for name, v in m.items():
                if v['info']['status'] == 'RUN':
                    self.assertTrue((v['balance']['smd_after'].abs() < 0.2).all(), (name, v['balance']))
                    self.assertFalse(set(v['pairs']['treated_word']) & set(v['pairs']['control_word']))

    suite = unittest.TestLoader().loadTestsFromTestCase(T)
    res = unittest.TextTestRunner(verbosity=2 if verbose else 1, stream=sys.stdout).run(suite)
    n_skip = len(res.skipped)
    print(f"\nSELF-TEST SUMMARY: {res.testsRun - len(res.failures) - len(res.errors) - n_skip} PASS, "
          f"{len(res.failures) + len(res.errors)} FAIL, {n_skip} SKIPPED"
          + (f" ({'; '.join(f'{t.id().split(chr(46))[-1]}: {r}' for t, r in res.skipped)})" if n_skip else ''))
    shutil.rmtree(tmp_root, ignore_errors=True)
    return res.wasSuccessful()


if __name__ == '__main__':
    sys.exit(main())
