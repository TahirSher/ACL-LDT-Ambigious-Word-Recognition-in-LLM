#!/usr/bin/env python3
"""
Layerwise encoding of word frequency and lexical-semantic (ambiguity) properties
in base vs. instruct LLMs
===============================================================================

Implements `layerwise_frequency_lexical_semantic_spec_corrected.md` end to end:

    data validation -> word-level splits -> content-matched prompts ->
    per-layer hidden-state extraction (cached) -> linear probes ->
    paired bootstrap / permutation statistics -> base-vs-instruct comparison ->
    figures, long-form CSVs, machine-readable config, reproducibility notes.

SCOPE (spec §1, §12)
  The stimuli are REAL WORDS ONLY. There is no WORD/NONWORD task, no lexical
  decision accuracy, no human RT. A successful probe shows that a property is
  LINEARLY RECOVERABLE from a representation, not that the model uses it.

RESEARCH QUESTIONS
  Q1  Frequency group (secondary view): held-out balanced accuracy / AUROC of a
      regularised logistic probe for high- vs low-frequency words; threshold =
      TRAINING-set median of log_subtlex, ties (== threshold) -> LOW.
  Q2  Continuous frequency (primary): held-out R^2 / MAE / RMSE of a ridge probe
      predicting log_subtlex (= log10(SUBTLEX + 1); verified, never re-logged).
  Q3  Lexical-semantic variables, each separately:
        wn_n_all          WordNet synset count (all POS)          ~ sense count (polysemy)
        gcide_etymologies GCIDE etymology count (dictionary-based) ~ unrelated meanings (homonymy)
      Pre-specified transform for both counts: log1p (right-skewed counts;
      skewness before/after is reported). Missing values stay missing.
  Q4  Base vs instruct: paired (same test words, same bootstrap draws) deltas per
      layer, layer-averaged and at the final layer; linear CKA between the two
      checkpoints' representations at every layer.

DESIGN DECISIONS THAT GO BEYOND THE SPEC (each is a control, not a new task)
  * Content-matched prompts. The base-model text is IDENTICAL to the instruct
    user turn; the only difference is the official chat wrapper. (The spec's
    example gives the instruct model an extra sentence the base model never
    sees, which would confound "instruction tuning" with "prompt content".)
  * Causal-attention fact used for design: in a decoder-only model the hidden
    state at the word's final token is a function of the tokens UP TO that
    token only. Text after the word therefore cannot change `word_last`
    features; it only matters for the separate `prompt_last` position.
  * Frequency is NEVER named in any prompt (no "common/rare/frequent/familiar"),
    nor are senses, ambiguity or etymology. Naming the probed property would
    turn the representation into a task-conditioned response to the property
    (target leakage, spec §4). This is enforced by an assertion.
  * Covariate-controlled ("partial") probing. Byte-level BPE gives frequent
    words a single token and rare words several, and WordNet sense counts rise
    with frequency (Zipf, 1945). A raw probe can therefore "decode semantics"
    by decoding frequency, or "decode frequency" by decoding token count. Each
    target is additionally probed after OLS-residualisation (fitted on TRAINING
    words only) on:
        log_subtlex       | length, length^2, n_subtokens, single-token flag
        wn_n_all          | surface + log_subtlex(+^2) + log1p(gcide_etymologies)
        gcide_etymologies | surface + log_subtlex(+^2) + log1p(wn_n_all)
    The mutual control of the two semantic variables operationalises the
    meanings-vs-senses dissociation of Rodd, Gaskell & Marslen-Wilson (2002).
  * Non-representational baselines: training mean; OLS on surface covariates;
    OLS on surface + frequency (semantic targets). Layer 0 (input embedding of
    the word-final token) is the static, context-free reference.
  * Layer selection without test leakage: in addition to the full trajectory and
    the final layer, each probe reports the layer chosen by TRAINING-ONLY
    cross-validation and its test score (spec §6: never select on test).
  * Tokenisation-stratified test scores (single- vs multi-token words), because
    multi-token words are assembled into word representations in early/middle
    layers (Kaplan et al., 2024; Feucht et al., 2024).
  * Optional untrained control (--random-init-control; Hewitt & Liang, 2019):
    the same architecture and tokenizer with random weights. Sub-token identity
    alone (e.g. "-er" derivations, which lack GCIDE etymologies) can make
    properties partly recoverable; trained-minus-random contrasts isolate what
    pre-training adds. Strongly recommended for at least one model per family.
  * Exploratory (clearly labelled) ambiguity contrast on the 309 human-normed
    Rodd et al. (2002) items: multiple-meanings and many-senses decoding with
    repeated nested CV, raw and with frequency/length/token-count linearly
    erased from the features.

ESTIMATION
  Features are z-scored with TRAINING statistics. Because the L2-penalised
  solution (with unpenalised intercept) lies in the span of the centred training
  rows (representer theorem), every probe is solved EXACTLY in the n_train-dim
  dual space: ridge via one eigendecomposition of the training Gram matrix with
  closed-form exact leave-one-out CV over the penalty grid (Golub, Heath & Wahba,
  1979); logistic regression on the equivalent row-space coordinates with
  5-fold stratified CV on log-loss. Penalty grids are scale-free (relative to the
  mean squared row norm); hitting a grid edge is flagged in the output.

STATISTICS
  95% percentile CIs from a paired bootstrap over test words (B = 2000); the
  SAME resample indices are used for every model, layer and condition with the
  same test set, so differences are paired. Per-layer test of INFORMATION:
  permutation of test targets with the trained probe fixed (P = 2000) using
  Pearson r (regression) or AUROC (classification), Holm-corrected across
  layers. With predictions fixed, permutation only changes sum(y * pred), so it
  is a test of association and NOT of R^2 > 0; "better than the training mean /
  chance" is read from the bootstrap CI. Figures mark a layer only when both
  hold. Base-vs-instruct and within-model contrasts: two-sided paired
  bootstrap p, Holm-corrected across layers. Note: bootstrap over test words
  captures test-sampling variance only; use --split-seeds 0,1,2,3,4 to add
  split/training variability (reported as mean +/- sd across seeds).

PROMPT FRAMES (identical text for base and instruct; {W} = stimulus)
  minimal  [PRIMARY; spec §4 wording]          positions: word_last
      You are processing a single English word.
      Word: {W}
  define   [SECONDARY; semantic-retrieval frame] positions: word_last, prompt_last
      Task: retrieve the meaning of one English word.
      The word is presented on its own, without any sentence context.

      Word: {W}
      Meaning:
  `word_last` = hidden state of the LAST sub-token of the stimulus (primary).
  `prompt_last` = last input position (base: ':' of "Meaning:"; instruct: last
  token of the assistant header), i.e. the state from which the model would
  begin to state the meaning. Positions are always reported separately.

MODELS (default_models(); same list as ldt_fresh.py)
  base:     SmolLM2-360M, SmolLM3-3B-Base, Qwen2.5-1.5B, Qwen2.5-3B, Qwen3-4B-Base,
            Qwen3-8B-Base, Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B
  instruct: SmolLM2-360M-, Qwen2.5-1.5B-, Qwen2.5-3B-, Llama-3.2-1B-, Llama-3.2-3B-,
            Llama-3.1-8B-Instruct (official chat template)
  Six base/instruct pairs get the base-vs-instruct comparison; SmolLM3-3B-Base,
  Qwen3-4B-Base and Qwen3-8B-Base have no instruct partner in this list and are
  analysed on their own (layerwise probes, baselines, within-model contrasts).

LAYER CONVENTION
  layer 0 = embedding output (hidden_states[0]); layer l (1..L) = output of
  transformer block l. In HF decoder-only models hidden_states[L] is taken
  after the final norm; earlier entries are raw residual-stream states.

USAGE
  python layerwise_frequency_semantics_probe.py                       # all 15 default models
  python layerwise_frequency_semantics_probe.py --models Qwen2.5-1.5B,Qwen2.5-1.5B-Instruct
  python layerwise_frequency_semantics_probe.py --custom-pair NAME:BASE_ID:INSTRUCT_ID
  python layerwise_frequency_semantics_probe.py --split-seeds 0,1,2,3,4
  python layerwise_frequency_semantics_probe.py --extract-only        # GPU stage only
  python layerwise_frequency_semantics_probe.py --analyze-only        # CPU stage from cache
  python layerwise_frequency_semantics_probe.py --self-test           # unit tests of the maths

REQUIREMENTS
  python>=3.10, numpy, pandas, scipy, scikit-learn, matplotlib, torch, transformers
  (Meta Llama checkpoints are gated: run `huggingface-cli login` first.)

KEY REFERENCES
  Brysbaert & New (2009) SUBTLEX-US. Rodd, Gaskell & Marslen-Wilson (2002) JML.
  Zipf (1945) meaning-frequency law. Hewitt & Liang (2019) control tasks.
  Belinkov (2022) probing classifiers. Kornblith et al. (2019) CKA. Song et al.
  (2012) unbiased HSIC. Golub, Heath & Wahba (1979) ridge CV. Campbell & Thompson
  (2008) out-of-sample R^2. Holm (1979). Efron & Tibshirani (1993) bootstrap.
  Timkey & van Schijndel (2021) rogue dimensions. Sun et al. (2024) massive
  activations. Kaplan et al. (2024) inner lexicon. Feucht et al. (2024) token
  erasure. Belrose et al. (2023) LEACE (linear concept erasure).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger('lexprobe')

DATA_URL = ('https://raw.githubusercontent.com/TahirSher/ACL-LDT-Ambigious-Word-Recognition-in-LLM/'
            'main/ambiguity_items_combined.csv')

# ════════════════════════════════════════════════════════════════════════════
# PROMPT FRAMES  (frozen before any result was seen; never optimised)
# ════════════════════════════════════════════════════════════════════════════
PROMPT_FRAMES = {
    'minimal': {
        'text': "You are processing a single English word.\nWord: {W}",
        'positions': ('word_last',),
        'role': 'primary',
    },
    'define': {
        'text': ("Task: retrieve the meaning of one English word.\n"
                 "The word is presented on its own, without any sentence context.\n\n"
                 "Word: {W}\nMeaning:"),
        'positions': ('word_last', 'prompt_last'),
        'role': 'secondary',
    },
}
PRIMARY_CONDITION = 'minimal|word_last'
# Words that would disclose a probed property (target leakage, spec §4/§12).
_LEAKY = re.compile(r'frequen|common|rare|familiar|usual|ambig|polysem|homonym|sense|'
                    r'etymolog|how many|number of|count', re.I)
for _n, _f in PROMPT_FRAMES.items():
    if _f['text'].count('{W}') != 1 or re.findall(r'\{(?!W\})', _f['text']):
        raise RuntimeError(f"frame {_n!r} must contain exactly one {{W}} placeholder")
    if _LEAKY.search(_f['text']):
        raise RuntimeError(f"frame {_n!r} names a probed property: {_LEAKY.search(_f['text']).group()!r}")


def frame_prompt_id(frame: str) -> str:
    return f"{frame}-{hashlib.sha1(PROMPT_FRAMES[frame]['text'].encode()).hexdigest()[:8]}"


# ════════════════════════════════════════════════════════════════════════════
# MODELS
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class ModelConfig:
    name: str
    model_id: str
    batch_size: int = 32
    family: str = ''
    variant: str = 'base'           # 'base' | 'instruct'
    base_of: str | None = None      # for instruct variants: name of the base counterpart
    chat: bool = False              # wrap every input in the tokenizer's chat template
    revision: str | None = None     # pin a hub commit for exact reproducibility
    random_init: bool = False       # control: same architecture/tokenizer, untrained weights

    def __post_init__(self):
        if not self.family:
            self.family = self.name.lower().split('-')[0].rstrip('0123456789.')
        if self.variant == 'instruct' and not self.chat:
            raise ValueError(f"{self.name}: instruct variants must use their chat template (spec §4)")

    @property
    def pair(self) -> str:
        """Comparison group = name of the base checkpoint."""
        return self.base_of or self.name


def default_models() -> list[ModelConfig]:
    base = [('SmolLM2-360M', 'HuggingFaceTB/SmolLM2-360M', 'smollm'),
            ('SmolLM3-3B-Base', 'HuggingFaceTB/SmolLM3-3B-Base', 'smollm'),
            ('Qwen2.5-1.5B', 'Qwen/Qwen2.5-1.5B', 'qwen'),
            ('Qwen2.5-3B', 'Qwen/Qwen2.5-3B', 'qwen'),
            ('Qwen3-4B-Base', 'Qwen/Qwen3-4B-Base', 'qwen'),
            ('Qwen3-8B-Base', 'Qwen/Qwen3-8B-Base', 'qwen'),
            ('Llama-3.2-1B', 'meta-llama/Llama-3.2-1B', 'llama'),
            ('Llama-3.2-3B', 'meta-llama/Llama-3.2-3B', 'llama'),
            ('Llama-3.1-8B', 'meta-llama/Llama-3.1-8B', 'llama')]
    instruct = [('SmolLM2-360M-Instruct', 'HuggingFaceTB/SmolLM2-360M-Instruct', 'smollm', 'SmolLM2-360M'),
                ('Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct', 'qwen', 'Qwen2.5-1.5B'),
                ('Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-3B-Instruct', 'qwen', 'Qwen2.5-3B'),
                ('Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.2-1B-Instruct', 'llama', 'Llama-3.2-1B'),
                ('Llama-3.2-3B-Instruct', 'meta-llama/Llama-3.2-3B-Instruct', 'llama', 'Llama-3.2-3B'),
                ('Llama-3.1-8B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct',    'llama',  'Llama-3.1-8B')]
    return ([ModelConfig(n, m, family=f) for n, m, f in base] +
            [ModelConfig(n, m, family=f, variant='instruct', base_of=b, chat=True)
             for n, m, f, b in instruct])


def group_models(models: list[ModelConfig]) -> dict[str, list[ModelConfig]]:
    """Group checkpoints by base model. Groups with base + instruct get the
    base-vs-instruct comparison (same architecture and tokenizer, so differences
    are attributable to post-training rather than tokenisation, spec §8); base-
    only groups (here SmolLM3-3B-Base, Qwen3-4B-Base, Qwen3-8B-Base) get the full
    layerwise analysis without that comparison."""
    names = [m.name for m in models]
    if len(names) != len(set(names)):
        raise ValueError('model names must be unique')
    groups: dict[str, list[ModelConfig]] = {}
    order = {'base': 0, 'instruct': 1, 'random_init': 2}
    for m in sorted(models, key=lambda m: order[m.variant]):
        groups.setdefault(m.pair, []).append(m)
    for g, ms in groups.items():
        for v in ('base', 'instruct', 'random_init'):
            if sum(m.variant == v for m in ms) > 1:
                raise ValueError(f"group {g!r} must contain at most one {v} model")
    return groups


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
TARGETS = {   # internal name -> (column, pre-specified transform)
    'log_subtlex': ('log_subtlex', 'identity'),
    'wn_n_all': ('wn_n_all', 'log1p'),
    'gcide_etymologies': ('gcide_etymologies', 'log1p'),
}
SURFACE = ('length', 'n_subtokens')
CONTROLS = {
    'log_subtlex': SURFACE,
    'wn_n_all': SURFACE + ('log_subtlex', 'gcide_etymologies'),
    'gcide_etymologies': SURFACE + ('log_subtlex', 'wn_n_all'),
}


@dataclass
class Config:
    data: str = 'ambiguity_items_combined.csv'
    out_dir: str = 'results_layerwise_lexical'
    frames: tuple = ('minimal', 'define')
    split_seeds: tuple = (0,)
    test_frac: float = 0.20
    inner_folds: int = 5
    ridge_alphas_rel: tuple = tuple(np.logspace(-4, 3, 36).round(10).tolist())
    logistic_Cs_rel: tuple = tuple(np.logspace(-2, 3, 11).round(10).tolist())
    n_boot: int = 2000
    n_perm: int = 2000
    ci: float = 0.95
    alpha_level: float = 0.05
    min_subset_n: int = 30
    # extraction
    batch_size: int | None = None   # None = ModelConfig.batch_size
    device: str = 'auto'
    precision: str = 'auto'          # auto | float32 | bfloat16
    feature_dtype: str = 'float16'   # on-disk cache dtype (overflow is checked)
    trust_remote_code: bool = False
    instruct_system_message: str | None = None   # None = template's own default
    chat_template_date: str = '26 Jul 2024'      # pins Llama-3.x date line (reproducibility)
    max_words: int | None = None
    # exploratory ambiguity analysis
    run_ambiguity: bool = True
    ambiguity_repeats: int = 3
    ambiguity_conditions: tuple = (PRIMARY_CONDITION,)
    force: bool = False
    seed: int = 12345

    def to_json(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def safe_name(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s)


def sha1_of(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# ════════════════════════════════════════════════════════════════════════════
# DATA: explicit column mapping, validation, de-duplication, splits
# ════════════════════════════════════════════════════════════════════════════
# Spec names first; the resolver refuses to guess if several candidates exist.
COLUMN_CANDIDATES = {
    'word': ('words', 'word', 'Word', 'Words', 'stimulus'),
    'subtlex': ('subtlex', 'SUBTLEX', 'SUBTLWF'),
    'log_subtlex': ('log_subtlex', 'log-subtlex', 'Lg10WF'),
    'gcide_etymologies': ('gcide-etymologies', 'gcide_etymologies'),
    'wn_n_all': ('w_n-n-all', 'wn_n_all', 'wn-n-all', 'w_n_n_all'),
}
OPTIONAL_CANDIDATES = {
    'provenance': ('provenance',), 'source_set': ('source_set',), 'ambiguity': ('ambiguity',),
    'senses': ('senses',), 'ambiguity_class': ('ambiguity_class',),
}
LEXICALITY_COLUMNS = ('lexicality', 'is_word', 'nonword', 'Lexicality', 'type')
NA_TOKENS = ['', 'NA', 'N/A', 'NaN', 'nan', 'null', 'NULL', 'None']


def resolve_columns(columns) -> dict:
    cols, mapping = list(columns), {}
    for internal, cands in {**COLUMN_CANDIDATES, **OPTIONAL_CANDIDATES}.items():
        hits = [c for c in cands if c in cols]
        if len(hits) > 1:
            raise ValueError(f"ambiguous column mapping for {internal!r}: {hits}; resolve explicitly")
        if hits:
            mapping[internal] = hits[0]
        elif internal in COLUMN_CANDIDATES:
            raise ValueError(f"required column {internal!r} not found (tried {cands})")
    return mapping


def transform(x: pd.Series, how: str) -> pd.Series:
    x = pd.to_numeric(x, errors='coerce').astype(float)
    return np.log1p(x) if how == 'log1p' else x


def load_and_validate(cfg: Config) -> tuple[pd.DataFrame, dict]:
    src = cfg.data if (os.path.exists(cfg.data) or cfg.data.startswith('http')) else DATA_URL
    header = pd.read_csv(src, nrows=0).columns
    mapping = resolve_columns(header)
    wcol = mapping['word']
    # Words such as "null", "nan" or "none" must stay words: NA parsing is
    # disabled for the stimulus column only.
    raw = pd.read_csv(src, keep_default_na=False, dtype={wcol: str},
                      na_values={c: NA_TOKENS for c in header if c != wcol})
    rep = {'source': src, 'column_mapping': mapping, 'n_rows_raw': int(len(raw))}
    for c in LEXICALITY_COLUMNS:
        if c in raw.columns:
            raise ValueError(f"column {c!r} suggests a lexicality label; this study is words-only")
    df = pd.DataFrame({k: raw[v] for k, v in mapping.items()})
    df['word'] = df['word'].astype(str).str.strip()
    for k in ('subtlex', 'log_subtlex', 'gcide_etymologies', 'wn_n_all'):
        before = df[k].isna().sum()
        df[k] = pd.to_numeric(df[k], errors='coerce')
        if df[k].isna().sum() > before:
            rep[f'{k}_non_numeric_coerced'] = int(df[k].isna().sum() - before)
    bad = df['word'].eq('') | ~df['word'].str.fullmatch(r"[A-Za-z][A-Za-z'\-]*")
    rep['excluded_invalid_word_strings'] = df.loc[bad, 'word'].tolist()
    df = df.loc[~bad].copy()
    rep['uppercase_words'] = df.loc[df['word'].str.lower() != df['word'], 'word'].tolist()

    # ── de-duplicate by case-folded identity; annotations must agree ───────
    df['word_key'] = df['word'].str.casefold()
    # numeric annotations must agree exactly (NaN included); design labels
    # (e.g. `senses`, defined only for Rodd 2002 Appendix B/C) must agree on
    # their non-missing values, and the non-missing value is kept.
    check = [c for c in ('subtlex', 'log_subtlex', 'gcide_etymologies', 'wn_n_all') if c in df.columns]
    labels = [c for c in ('ambiguity', 'senses') if c in df.columns]
    dup = df[df.duplicated('word_key', keep=False)]
    conflicts = {}
    for key, g in dup.groupby('word_key'):
        bad_cols = ([c for c in check if g[c].nunique(dropna=False) > 1] +
                    [c for c in labels if g[c].nunique(dropna=True) > 1])
        if bad_cols:
            conflicts[key] = bad_cols
    if conflicts:
        raise ValueError(f"duplicate words with conflicting annotations: {conflicts}")
    rep['n_duplicate_rows'] = int(df.duplicated('word_key').sum())
    rep['duplicate_rule'] = ('rows sharing a case-folded word have identical annotations (verified); '
                             'collapsed to one record per word, so a word cannot occur in two splits')
    agg = {c: 'first' for c in df.columns if c not in ('word_key',)}
    for c in ('source_set', 'provenance'):
        if c in df.columns:
            agg[c] = lambda s: ';'.join(sorted(set(map(str, s.dropna()))))
    wt = df.groupby('word_key', sort=False).agg(agg).reset_index(drop=True)
    wt['n_source_rows'] = df.groupby('word_key', sort=False).size().values
    wt['is_published'] = (wt['provenance'].str.contains('published') if 'provenance' in wt
                          else False)
    wt['length'] = wt['word'].str.len().astype(float)
    if cfg.max_words and cfg.max_words < len(wt):
        wt = wt.sample(n=cfg.max_words, random_state=cfg.seed).reset_index(drop=True)
        rep['subsampled_to'] = cfg.max_words

    # ── frequency transform check (spec §6: do not re-log) ─────────────────
    m = wt['subtlex'].notna() & wt['log_subtlex'].notna() & (wt['subtlex'] > 0)
    dev_p1 = float((np.log10(wt.loc[m, 'subtlex'] + 1) - wt.loc[m, 'log_subtlex']).abs().max())
    dev_0 = float((np.log10(wt.loc[m, 'subtlex']) - wt.loc[m, 'log_subtlex']).abs().max())
    rep['log_subtlex_check'] = {
        'max_abs_dev_vs_log10(subtlex+1)': dev_p1, 'max_abs_dev_vs_log10(subtlex)': dev_0,
        'conclusion': ('log_subtlex = log10(subtlex + 1) up to 3-decimal rounding; used as provided'
                       if dev_p1 <= 5e-4 + 1e-9 else 'transformation differs from log10(x+1); inspect')}
    rep['n_unique_words'] = int(len(wt))
    rep['dataset_contains_only_words'] = True
    rep['targets'] = {}
    for t, (col, how) in TARGETS.items():
        x = wt[col]
        xt = transform(x, how).dropna()
        rep['targets'][t] = {
            'column': mapping.get(col, col), 'transform': how, 'n_valid': int(x.notna().sum()),
            'n_missing': int(x.isna().sum()), 'n_zero': int((x == 0).sum()),
            'min': float(x.min()), 'median': float(x.median()), 'max': float(x.max()),
            'skew_raw': float(stats.skew(x.dropna())), 'skew_transformed': float(stats.skew(xt))}
    rep['notes'] = [
        'gcide_etymologies is a count of etymology records in GCIDE; 0 is a real value '
        '(typically derived forms such as "ruler", "speaker" that carry no etymology field) and '
        'reflects dictionary coverage, not "zero meanings". It is a homonymy PROXY.',
        'wn_n_all is the number of WordNet 3.0 synsets across parts of speech: a sense-count '
        'proxy that also reflects lexicographic granularity; it is not a similarity measure.',
        'Missing annotations are kept missing; each analysis uses its own complete cases.']
    return wt, rep


def make_split(wt: pd.DataFrame, seed: int, test_frac: float) -> np.ndarray:
    """Word-level train/test split stratified on log-frequency decile x provenance."""
    rank = wt['log_subtlex'].rank(method='first')
    dec = pd.qcut(rank, 10, labels=False).fillna(4).astype(int).astype(str)
    strata = dec + '|' + np.where(wt['is_published'], 'pub', 'wn')
    if strata.value_counts().min() < 2:
        strata = dec
    idx = np.arange(len(wt))
    _, te = train_test_split(idx, test_size=test_frac, random_state=seed, stratify=strata)
    split = np.full(len(wt), 'train', dtype=object)
    split[te] = 'test'
    return split


def covariate_design(wt: pd.DataFrame, n_sub: np.ndarray, names) -> np.ndarray:
    cols = []
    for nm in names:
        if nm == 'length':
            x = wt['length'].values.astype(float)
            cols += [x, x ** 2]
        elif nm == 'n_subtokens':
            x = n_sub.astype(float)
            cols += [x, (x == 1).astype(float)]
        elif nm == 'log_subtlex':
            x = wt['log_subtlex'].values.astype(float)
            cols += [x, x ** 2]
        elif nm in TARGETS:
            x = transform(wt[TARGETS[nm][0]], TARGETS[nm][1]).values
            cols += [x, x ** 2]
        else:
            raise KeyError(nm)
    return np.column_stack(cols)


def ols_fit_predict(C_tr: np.ndarray, y_tr: np.ndarray, *C_new: np.ndarray):
    """OLS with intercept on train-standardised covariates; returns predictions."""
    mu, sd = C_tr.mean(0), C_tr.std(0)
    sd[sd < 1e-12] = 1.0
    D = lambda C: np.column_stack([np.ones(len(C)), (C - mu) / sd])
    beta, *_ = np.linalg.lstsq(D(C_tr), y_tr, rcond=None)
    return [D(C) @ beta for C in (C_tr,) + C_new]


# ════════════════════════════════════════════════════════════════════════════
# HIDDEN-STATE EXTRACTION
# ════════════════════════════════════════════════════════════════════════════
def _transformers_version() -> tuple:
    import transformers
    return tuple(int(p) for p in re.findall(r'\d+', transformers.__version__)[:3])


class HiddenStateExtractor:
    def __init__(self, spec: ModelConfig, cfg: Config):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.spec, self.cfg, self.torch = spec, cfg, torch
        dev = cfg.device
        if dev == 'auto':
            dev = 'cuda' if torch.cuda.is_available() else (
                'mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()
                else 'cpu')
        self.device = dev
        if cfg.precision == 'float32' or dev != 'cuda':
            dtype = torch.float32
        elif cfg.precision == 'bfloat16' or torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float32         # fp16 risks overflow in residual streams
        self.dtype = dtype
        tok = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision, use_fast=True,
                                            trust_remote_code=cfg.trust_remote_code)
        if not tok.is_fast:
            raise RuntimeError(f"{spec.name}: a fast tokenizer is required for character offsets")
        if spec.chat and not tok.chat_template:
            raise RuntimeError(f"{spec.name}: instruct condition but tokenizer has no chat template")
        if spec.family in ('qwen', 'smollm') and re.search(r'Qwen3|SmolLM3', spec.model_id) and \
                _transformers_version() < (4, 53, 0):
            raise RuntimeError(f"{spec.name} needs transformers >= 4.53 (Qwen3 / SmolLM3 architectures)")
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.tok = tok
        dkey = 'dtype' if _transformers_version() >= (4, 56, 0) else 'torch_dtype'
        if spec.random_init:
            # Untrained control (Hewitt & Liang, 2019): the architecture's own
            # initialisation under a fixed seed; isolates what is recoverable
            # from token identity + architecture alone.
            from transformers import AutoConfig
            torch.manual_seed(cfg.seed)
            mcfg = AutoConfig.from_pretrained(spec.model_id, revision=spec.revision,
                                              trust_remote_code=cfg.trust_remote_code)
            model = AutoModelForCausalLM.from_config(mcfg, trust_remote_code=cfg.trust_remote_code,
                                                     **{dkey: dtype})
        else:
            model = AutoModelForCausalLM.from_pretrained(
                spec.model_id, revision=spec.revision, low_cpu_mem_usage=True,
                trust_remote_code=cfg.trust_remote_code, **{dkey: dtype})
        model.to(dev).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.n_layers = int(model.config.num_hidden_layers)
        self.hidden = int(model.config.hidden_size)
        self.info = {
            'model_id': spec.model_id, 'revision_requested': spec.revision,
            'commit_hash': getattr(model.config, '_commit_hash', None),
            'variant': spec.variant, 'random_init': spec.random_init, 'init_seed': cfg.seed if spec.random_init else None,
            'n_layers': self.n_layers, 'hidden_size': self.hidden,
            'n_parameters': int(sum(p.numel() for p in model.parameters())),
            'dtype': str(dtype).replace('torch.', ''), 'device': dev,
            'tokenizer_class': type(tok).__name__, 'vocab_size': len(tok),
            'bos_token': tok.bos_token, 'adds_bos_in_base_mode': None,
            'chat_template_sha1': (hashlib.sha1(tok.chat_template.encode()).hexdigest()
                                   if isinstance(tok.chat_template, str) else None)}
        logger.info(f"loaded {spec.name}: L={self.n_layers}, d={self.hidden}, "
                    f"{self.info['n_parameters'] / 1e9:.2f}B params, {self.info['dtype']}, {dev}")

    # ── prompt rendering ────────────────────────────────────────────────────
    def render(self, frame: str, word: str) -> tuple[str, tuple[int, int]]:
        tpl = PROMPT_FRAMES[frame]['text']
        pre = tpl.split('{W}')[0]
        body = tpl.replace('{W}', word)
        span = (len(pre), len(pre) + len(word))
        if not self.spec.chat:
            return body, span
        msgs = ([{'role': 'system', 'content': self.cfg.instruct_system_message}]
                if self.cfg.instruct_system_message else [])
        msgs.append({'role': 'user', 'content': body})
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            date_string=self.cfg.chat_template_date)
        if text.count(body) != 1:
            raise RuntimeError(f"{self.spec.name}: chat template altered or duplicated the prompt")
        off = text.find(body)
        return text, (span[0] + off, span[1] + off)

    def tokenize_one(self, frame: str, word: str) -> dict:
        text, (a, b) = self.render(frame, word)
        enc = self.tok(text, add_special_tokens=not self.spec.chat, return_offsets_mapping=True)
        ids, offs = enc['input_ids'], enc['offset_mapping']
        sel = [i for i, (s, e) in enumerate(offs) if e > s and e > a and s < b]
        if not sel or sel != list(range(sel[0], sel[-1] + 1)):
            raise RuntimeError(f"{self.spec.name}: stimulus span not found/contiguous for {word!r}")
        piece = self.tok.decode(ids[sel[0]:sel[-1] + 1])
        clean = (piece.strip() == word and offs[sel[-1]][1] <= b and offs[sel[0]][0] >= a - 1)
        return {'text': text, 'ids': ids, 'word_tokens': sel, 'word_last': sel[-1],
                'prompt_last': len(ids) - 1, 'n_subtokens': len(sel), 'span_clean': bool(clean),
                'word_token_ids': [ids[i] for i in sel],
                'word_token_strs': self.tok.convert_ids_to_tokens([ids[i] for i in sel])}

    # ── forward pass ────────────────────────────────────────────────────────
    def _forward(self, recs: list[dict], positions: tuple) -> dict:
        torch = self.torch
        T = max(len(r['ids']) for r in recs)
        ids = torch.full((len(recs), T), self.pad_id, dtype=torch.long)
        am = torch.zeros((len(recs), T), dtype=torch.long)
        for i, r in enumerate(recs):      # RIGHT padding: real tokens keep positions 0..n-1
            ids[i, :len(r['ids'])] = torch.tensor(r['ids'])
            am[i, :len(r['ids'])] = 1
        ids, am = ids.to(self.device), am.to(self.device)
        with torch.inference_mode():
            out = self.model(input_ids=ids, attention_mask=am, output_hidden_states=True,
                             use_cache=False)
        hs = out.hidden_states
        if len(hs) != self.n_layers + 1:
            raise RuntimeError(f"expected {self.n_layers + 1} hidden states, got {len(hs)}")
        ar = torch.arange(len(recs), device=self.device)
        res = {}
        for pos in positions:
            idx = torch.tensor([r[pos] for r in recs], device=self.device)
            res[pos] = torch.stack([h[ar, idx] for h in hs], 0).float().cpu().numpy()
        return res

    def extract(self, words: list[str], frame: str, fdir: Path) -> dict:
        positions = PROMPT_FRAMES[frame]['positions']
        recs = [self.tokenize_one(frame, w) for w in words]
        store = np.float16 if self.cfg.feature_dtype == 'float16' else np.float32
        mms = {p: np.lib.format.open_memmap(fdir / f'{frame}__{p}.npy.part', mode='w+', dtype=store,
                                            shape=(self.n_layers + 1, len(words), self.hidden))
               for p in positions}
        order = np.argsort([len(r['ids']) for r in recs], kind='stable')
        bs = self.cfg.batch_size or self.spec.batch_size
        t0 = time.time()
        for s in range(0, len(order), bs):
            rows = order[s:s + bs]
            out = self._forward([recs[i] for i in rows], positions)
            for p, arr in out.items():
                if store is np.float16 and np.abs(arr).max() > 6.0e4:
                    raise RuntimeError("activation exceeds float16 range; rerun with "
                                       "--feature-dtype float32")
                mms[p][:, rows, :] = arr.astype(store)
        # batch-invariance check: single-item forward must reproduce batched states
        chk = []
        for i in range(min(3, len(recs))):
            single = self._forward([recs[i]], positions)
            for p in positions:
                for l in (self.n_layers // 2, self.n_layers):
                    a, b = single[p][l, 0], np.asarray(mms[p][l, i], dtype=np.float32)
                    chk.append(float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-8)))
        tol = 2e-2 if self.dtype != self.torch.float32 else 2e-3
        if max(chk) > tol:
            logger.warning(f"{self.spec.name}/{frame}: batched vs single relative diff "
                           f"{max(chk):.2e} > {tol} (padding/numerics)")
        for p, mm in mms.items():
            mm.flush()
            del mm
            os.replace(fdir / f'{frame}__{p}.npy.part', fdir / f'{frame}__{p}.npy')
        logger.info(f"{self.spec.name}/{frame}: {len(words)} words in {time.time() - t0:.1f}s")
        tokrec = pd.DataFrame({
            'word': words, 'rendered_prompt': [r['text'] for r in recs],
            'input_ids': [json.dumps(r['ids']) for r in recs],
            'word_token_positions': [json.dumps(r['word_tokens']) for r in recs],
            'word_token_ids': [json.dumps(r['word_token_ids']) for r in recs],
            'word_token_strs': [json.dumps(r['word_token_strs']) for r in recs],
            'word_last': [r['word_last'] for r in recs], 'prompt_last': [r['prompt_last'] for r in recs],
            'n_subtokens': [r['n_subtokens'] for r in recs],
            'span_clean': [r['span_clean'] for r in recs]})
        return {'tokens': tokrec, 'batch_invariance_max_rel_diff': max(chk),
                'n_span_not_clean': int((~tokrec['span_clean']).sum())}


def extract_model(spec: ModelConfig, wt: pd.DataFrame, cfg: Config, out: Path) -> None:
    fdir = out / 'features' / safe_name(spec.name)
    fdir.mkdir(parents=True, exist_ok=True)
    words = wt['word'].tolist()
    todo = []
    for frame in cfg.frames:
        key = cache_key(spec, frame, words, cfg)
        mp = fdir / f'{frame}__meta.json'
        if not cfg.force and mp.exists() and json.loads(mp.read_text()).get('cache_key') == key and \
                all((fdir / f'{frame}__{p}.npy').exists() for p in PROMPT_FRAMES[frame]['positions']):
            logger.info(f"{spec.name}/{frame}: cached features found")
            continue
        todo.append((frame, key, mp))
    if not todo:
        return
    import torch
    import transformers
    seed_everything(cfg.seed)
    ex = HiddenStateExtractor(spec, cfg)
    for frame, key, mp in todo:
        res = ex.extract(words, frame, fdir)
        res['tokens'].to_csv(fdir / f'{frame}__tokens.csv.gz', index=False)
        meta = {'cache_key': key, 'model': spec.name, 'pair': spec.pair, 'frame': frame,
                'prompt_template': PROMPT_FRAMES[frame]['text'], 'prompt_id': frame_prompt_id(frame),
                'positions': list(PROMPT_FRAMES[frame]['positions']),
                'words_sha1': sha1_of(words), 'n_words': len(words),
                'example_rendered_prompt': res['tokens']['rendered_prompt'].iloc[0],
                'instruct_system_message': cfg.instruct_system_message if spec.chat else None,
                'chat_template_date': cfg.chat_template_date if spec.chat else None,
                'batch_invariance_max_rel_diff': res['batch_invariance_max_rel_diff'],
                'n_span_not_clean': res['n_span_not_clean'],
                'feature_dtype': cfg.feature_dtype, **ex.info,
                'torch': torch.__version__, 'transformers': transformers.__version__}
        mp.write_text(json.dumps(meta, indent=2))
    del ex
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cache_key(spec: ModelConfig, frame: str, words: list, cfg: Config) -> str:
    return sha1_of({'model_id': spec.model_id, 'revision': spec.revision, 'variant': spec.variant,
                    'random_init': spec.random_init, 'init_seed': cfg.seed if spec.random_init else None,
                    'frame_text': PROMPT_FRAMES[frame]['text'], 'words': sha1_of(words),
                    'system': cfg.instruct_system_message if spec.chat else None,
                    'date': cfg.chat_template_date if spec.chat else None,
                    'precision': cfg.precision, 'feature_dtype': cfg.feature_dtype})


def load_features(spec: ModelConfig, frame: str, pos: str, out: Path, words: list) -> tuple:
    fdir = out / 'features' / safe_name(spec.name)
    meta = json.loads((fdir / f'{frame}__meta.json').read_text())
    if meta['words_sha1'] != sha1_of(words):
        raise RuntimeError(f"{spec.name}/{frame}: cached word order does not match the data")
    feats = np.load(fdir / f'{frame}__{pos}.npy', mmap_mode='r')
    toks = pd.read_csv(fdir / f'{frame}__tokens.csv.gz')
    return feats, meta, toks


# ════════════════════════════════════════════════════════════════════════════
# PROBES (exact dual-space ridge / logistic)
# ════════════════════════════════════════════════════════════════════════════
class ProbeSpace:
    """Centred dual representation of a training set (and its test set).

    Given the (feature-standardised) Gram blocks G_tt = Z_tr Z_tr^T and
    G_et = Z_te Z_tr^T, centring by the training mean mu is exact:
        A A^T = G_tt - r 1^T - 1 r^T + s,     B A^T = G_et - 1 r^T - q 1^T + s
    with r_i = mean_j G_tt[i, j], s = mean(G_tt), q_k = mean_j G_et[k, j].
    """

    def __init__(self, G_tt: np.ndarray, G_et: np.ndarray):
        r, s, q = G_tt.mean(1), G_tt.mean(), G_et.mean(1)
        K = G_tt - r[:, None] - r[None, :] + s
        self.K_te = G_et - r[None, :] - q[:, None] + s
        K = 0.5 * (K + K.T)
        lam, U = np.linalg.eigh(K)
        self.lam, self.U = np.clip(lam, 0.0, None), U
        self.n = len(K)
        self.scale = max(self.lam.sum() / self.n, 1e-12)      # mean squared centred row norm
        # Constant features (e.g. layer 0 at `prompt_last`, which is the same
        # token for every word) carry no information: predict a constant.
        self.degenerate = bool(self.lam.max() <= 1e-9 * max(np.abs(G_tt).max(), 1e-300))
        self._R = None

    # ridge with exact LOO selection of the penalty
    def ridge(self, y: np.ndarray, alphas_rel) -> tuple[np.ndarray, dict]:
        alphas_rel = np.asarray(alphas_rel, float)
        lam, U, n = self.lam, self.U, self.n
        ybar = y.mean()
        if self.degenerate:
            return np.full(self.K_te.shape[0], ybar), {
                'hparam': np.nan, 'at_edge': False, 'cv_score': -1.0 / (n - 1),   # LOO R² of the mean
                'tuning': 'constant features: training-mean predictor'}
        yc = y - ybar
        Uy = U.T @ yc
        alphas = alphas_rel * self.scale
        F = lam[:, None] / (lam[:, None] + alphas[None, :])
        fitted = ybar + U @ (F * Uy[:, None])
        H = 1.0 / n + (U ** 2) @ F
        loo = (y[:, None] - fitted) / np.clip(1.0 - H, 1e-12, None)
        mse = (loo ** 2).mean(0)
        k = int(np.argmin(mse))
        c = U @ (Uy / (lam + alphas[k]))
        pred = ybar + self.K_te @ c
        return pred, {'hparam': float(alphas_rel[k]), 'at_edge': k in (0, len(alphas) - 1),
                      'cv_score': float(1.0 - mse[k] / max(np.mean(yc ** 2), 1e-300)),
                      'tuning': 'exact LOO-CV on training words'}

    def row_space(self) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates of train/test rows in the training row space, rescaled so the
        mean squared training row norm is 1 (exact for L2 models)."""
        if self._R is None:
            keep = self.lam > self.lam.max() * 1e-10
            sq = np.sqrt(self.lam[keep])
            c = np.sqrt(self.scale)
            self._R = (self.U[:, keep] * sq / c, (self.K_te @ self.U[:, keep]) / sq / c)
        return self._R

    def logistic(self, y: np.ndarray, Cs_rel, seed: int, folds: int) -> tuple[np.ndarray, dict]:
        if self.degenerate:      # balanced-prior constant: chance-level by construction
            return np.full(self.K_te.shape[0], 0.5), {
                'hparam': np.nan, 'at_edge': False, 'cv_score': -np.log(2.0),
                'tuning': 'constant features: prior predictor'}
        R_tr, R_te = self.row_space()
        Cs = np.asarray(Cs_rel, float) / np.sqrt(self.n)
        nf = int(min(folds, np.bincount(y).min()))
        max_iter, not_conv = 5000, 0

        def new_clf(C):
            return LogisticRegression(C=C, class_weight='balanced', max_iter=max_iter, tol=1e-6,
                                      warm_start=True)
        # Explicit CV over C (ascending = decreasing penalty, warm-started along
        # the path); held-out mean log-loss is the selection criterion.
        ll = np.zeros(len(Cs))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            for tr, va in StratifiedKFold(nf, shuffle=True, random_state=seed).split(R_tr, y):
                clf = new_clf(Cs[0])
                for k, C in enumerate(Cs):
                    clf.set_params(C=C)
                    clf.fit(R_tr[tr], y[tr])
                    not_conv += int(np.max(clf.n_iter_) >= max_iter)
                    pv = np.clip(clf.predict_proba(R_tr[va])[:, 1], 1e-12, 1 - 1e-12)
                    ll[k] += -np.sum(y[va] * np.log(pv) + (1 - y[va]) * np.log(1 - pv))
            k = int(np.argmin(ll))
            clf = new_clf(Cs[k]).fit(R_tr, y)
            not_conv += int(np.max(clf.n_iter_) >= max_iter)
        p = clf.predict_proba(R_te)[:, 1]
        return p, {'hparam': float(Cs_rel[k]), 'at_edge': k in (0, len(Cs) - 1),
                   'cv_score': float(-ll[k] / len(y)), 'n_not_converged': not_conv,
                   'tuning': f'{nf}-fold stratified CV (log-loss) on training words'}


# ════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ════════════════════════════════════════════════════════════════════════════
_RS_CACHE: dict = {}


def resampling(seed: int, key: str, n: int, B: int, P: int):
    k = (seed, key, n, B, P)
    if k not in _RS_CACHE:
        h = int(hashlib.sha256(f'{seed}|{key}|{n}'.encode()).hexdigest()[:12], 16)
        rng = np.random.default_rng(h)
        _RS_CACHE[k] = (rng.integers(0, n, size=(B, n)), np.argsort(rng.random((P, n)), axis=1))
    return _RS_CACHE[k]


def _r2(Y, P):
    sse = ((Y - P) ** 2).sum(-1)
    sst = ((Y - Y.mean(-1, keepdims=True)) ** 2).sum(-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return 1.0 - sse / np.where(sst > 0, sst, np.nan)


def _pearson(Y, P):
    Yc, Pc = Y - Y.mean(-1, keepdims=True), P - P.mean(-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (Yc * Pc).sum(-1) / np.sqrt((Yc ** 2).sum(-1) * (Pc ** 2).sum(-1))


def _auroc(Y, S):
    Y = np.atleast_2d(Y).astype(bool)
    S = np.atleast_2d(S)
    ranks = stats.rankdata(S, axis=-1)
    n1 = Y.sum(-1)
    n0 = Y.shape[-1] - n1
    with np.errstate(divide='ignore', invalid='ignore'):
        a = ((ranks * Y).sum(-1) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return np.where((n1 > 0) & (n0 > 0), a, np.nan)


def _bacc(Y, Yhat):
    Y, Yhat = np.atleast_2d(Y).astype(bool), np.atleast_2d(Yhat).astype(bool)
    with np.errstate(divide='ignore', invalid='ignore'):
        tpr = (Y & Yhat).sum(-1) / Y.sum(-1)
        tnr = (~Y & ~Yhat).sum(-1) / (~Y).sum(-1)
    return 0.5 * (tpr + tnr)


def _f1(Y, Yhat):
    Y, Yhat = np.atleast_2d(Y).astype(bool), np.atleast_2d(Yhat).astype(bool)
    tp = (Y & Yhat).sum(-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return tp / (tp + 0.5 * ((~Y & Yhat).sum(-1) + (Y & ~Yhat).sum(-1)))


def regression_metrics(y, pred, y_train_mean, boot, perm) -> dict:
    """metric -> (score, boot array, perm-null array or None)."""
    Yb, Pb = y[boot], pred[boot]
    sst_tm = lambda Y: ((Y - y_train_mean) ** 2).sum(-1)
    out = {
        'r2': (_r2(y, pred), _r2(Yb, Pb), None),
        'r2_oos_vs_train_mean': (1 - ((y - pred) ** 2).sum() / sst_tm(y),
                                 1 - ((Yb - Pb) ** 2).sum(-1) / sst_tm(Yb), None),
        'mae': (np.abs(y - pred).mean(), np.abs(Yb - Pb).mean(-1), None),
        'rmse': (np.sqrt(((y - pred) ** 2).mean()), np.sqrt(((Yb - Pb) ** 2).mean(-1)), None),
        # Permutation statistic. With the trained probe fixed, permuting test
        # targets changes only the cross term sum(y * pred), so the exact test is
        # one of ASSOCIATION (H0: predictions carry no information about targets),
        # not of R^2 > 0; "beats the training mean" is judged by the R^2 CI.
        'pearson_r': (_pearson(y, pred), _pearson(Yb, Pb), _pearson(y[perm], pred[None, :])),
    }
    return {k: (float(v[0]), v[1], v[2]) for k, v in out.items()}


def classification_metrics(y, prob, boot, perm) -> dict:
    yhat = prob >= 0.5
    Yb, Sb, Hb = y[boot], prob[boot], yhat[boot]
    Yp = y[perm]
    out = {
        'balanced_accuracy': (_bacc(y, yhat)[0], _bacc(Yb, Hb), None),
        'auroc': (_auroc(y, prob)[0], _auroc(Yb, Sb), _auroc(Yp, prob[None, :])),
        'f1_high': (_f1(y, yhat)[0], _f1(Yb, Hb), None),
        'accuracy': ((y == yhat).mean(), (Yb == Hb).mean(-1), None),
        'recall_high': ((yhat[y == 1]).mean(), None, None),
        'recall_low': ((~yhat[y == 0]).mean(), None, None),
        'tn': (float(((y == 0) & ~yhat).sum()), None, None),
        'fp': (float(((y == 0) & yhat).sum()), None, None),
        'fn': (float(((y == 1) & ~yhat).sum()), None, None),
        'tp': (float(((y == 1) & yhat).sum()), None, None),
        'prop_high_in_test': (float(y.mean()), None, None),
    }
    return {k: (float(v[0]), v[1], v[2]) for k, v in out.items()}


def ci_of(b, level):
    if b is None:
        return np.nan, np.nan
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    if not len(b):
        return np.nan, np.nan
    a = (1 - level) / 2
    return float(np.quantile(b, a)), float(np.quantile(b, 1 - a))


ASSOCIATION_METRIC = {'regression': 'pearson_r', 'freq_group': 'auroc'}
CHANCE = {'r2': 0.0, 'balanced_accuracy': 0.5, 'auroc': 0.5}


def perm_p_greater(obs, null):
    if null is None:
        return np.nan
    null = np.asarray(null, float)
    return float((1 + np.sum(null >= obs - 1e-12)) / (len(null) + 1))


def boot_p_two_sided(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    B = len(d)
    if not B:
        return np.nan
    return float(min(1.0, 2 * min((1 + (d <= 0).sum()) / (B + 1), (1 + (d >= 0).sum()) / (B + 1))))


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    adj = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    m = len(ok)
    run = 0.0
    for rank, i in enumerate(ok[np.argsort(p[ok], kind='stable')]):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def _hsic_unbiased(K, L):
    n = K.shape[0]
    K, L = K.copy(), L.copy()
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(L, 0.0)
    t1 = (K * L).sum()
    t2 = K.sum() * L.sum() / ((n - 1) * (n - 2))
    t3 = 2.0 / (n - 2) * (K.sum(0) @ L.sum(0))
    return (t1 + t2 - t3) / (n * (n - 3))


def cka(X: np.ndarray, Y: np.ndarray) -> tuple[float, float]:
    """(debiased linear CKA, standard linear CKA); features z-scored first so
    rogue/massive dimensions do not dominate (Timkey & van Schijndel, 2021)."""
    def z(A):
        A = A - A.mean(0)
        s = A.std(0)
        s[s < 1e-12] = 1.0
        return A / s
    X, Y = z(np.asarray(X, np.float64)), z(np.asarray(Y, np.float64))
    K, L = X @ X.T, Y @ Y.T
    if not np.abs(K).max() > 0 or not np.abs(L).max() > 0:
        return np.nan, np.nan          # a constant representation: CKA undefined
    deb = _hsic_unbiased(K, L) / np.sqrt(max(_hsic_unbiased(K, K) * _hsic_unbiased(L, L), 1e-300))
    std = (K * L).sum() / np.sqrt((K * K).sum() * (L * L).sum())   # columns already centred
    return float(deb), float(std)


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════
def build_analyses(wt: pd.DataFrame, n_sub: np.ndarray, split: np.ndarray) -> tuple[list, float, dict]:
    is_tr, is_te = split == 'train', split == 'test'
    lf = wt['log_subtlex'].values.astype(float)
    thr = float(np.nanmedian(lf[is_tr]))
    tinfo = {'frequency_threshold': thr,
             'threshold_rule': 'training-set median of log_subtlex; high = value > threshold, '
                               'low = value <= threshold (ties assigned to LOW)',
             'n_train_at_threshold': int(np.sum(np.isclose(lf[is_tr], thr))),
             'n_test_at_threshold': int(np.sum(np.isclose(lf[is_te], thr)))}
    Y = {t: transform(wt[c], how).values for t, (c, how) in TARGETS.items()}
    valid = {t: np.isfinite(v) for t, v in Y.items()}
    analyses = []

    def add(aid, task, target, variant, kind, y, req, baselines):
        ok = np.logical_and.reduce([valid[r] for r in req])
        tr, te = np.where(ok & is_tr)[0], np.where(ok & is_te)[0]
        analyses.append(dict(id=aid, task=task, target=target, variant=variant, kind=kind, y=y,
                             tr=tr, te=te, mask_key='+'.join(sorted(req)), baselines=baselines))

    yb = np.where(valid['log_subtlex'], (lf > thr).astype(float), np.nan)
    add('freq_group|log_subtlex|raw', 'freq_group', 'log_subtlex', 'raw', 'clf', yb,
        ('log_subtlex',), {'surface': SURFACE})
    for t in TARGETS:
        base = {'surface': SURFACE}
        if t != 'log_subtlex':
            base['surface+frequency'] = SURFACE + ('log_subtlex',)
        # log_subtlex is required everywhere (drops 2 words) so that probe and
        # surface+frequency baseline are scored on IDENTICAL test words (paired).
        add(f'regression|{t}|raw', 'regression', t, 'raw', 'reg', Y[t], tuple(sorted({t, 'log_subtlex'})), base)
        req = tuple(sorted({t, 'log_subtlex'} | {c for c in CONTROLS[t] if c in TARGETS}))
        add(f'regression|{t}|controlled', 'regression', t, 'controlled', 'reg', Y[t], req,
            {'controls': CONTROLS[t]})
    # residualise controlled targets on covariates (fit on TRAINING words only)
    for a in analyses:
        a['y_tr'], a['y_te'] = a['y'][a['tr']], a['y'][a['te']]
        if a['variant'] == 'controlled':
            C = covariate_design(wt, n_sub, CONTROLS[a['target']])
            ptr, pte = ols_fit_predict(C[a['tr']], a['y_tr'], C[a['te']])
            a['y_tr'], a['y_te'] = a['y_tr'] - ptr, a['y_te'] - pte
            a['controls_r2_train'] = float(_r2(a['y'][a['tr']], ptr))
        if a['kind'] == 'clf':
            a['y_tr'], a['y_te'] = a['y_tr'].astype(int), a['y_te'].astype(int)
    return analyses, thr, tinfo


def analyze_condition(spec: ModelConfig, cond: str, feats, toks: pd.DataFrame, wt: pd.DataFrame,
                      split: np.ndarray, seed: int, cfg: Config, meta: dict) -> tuple:
    """Probe every layer of one (model, frame, position) feature set."""
    frame, pos = cond.split('|')
    n_sub = toks['n_subtokens'].values.astype(int)
    analyses, thr, tinfo = build_analyses(wt, n_sub, split)
    L = feats.shape[0] - 1
    is_tr = split == 'train'
    common = dict(model=spec.name, pair=spec.pair, condition=spec.variant, frame=frame,
                  extraction_position=pos, prompt_id=frame_prompt_id(frame), seed=seed,
                  frequency_threshold=thr, n_layers=L)
    rows, preds, store = [], [], {}
    B, P = cfg.n_boot, cfg.n_perm

    def emit(a, layer, probe, mets, info, subset='all', n_test=None):
        for metric, (score, bt, null) in mets.items():
            lo, hi = ci_of(bt, cfg.ci)
            rows.append({**common, 'layer': layer, 'rel_depth': (layer / L if layer is not None and layer >= 0 else np.nan),
                         'task': a['task'], 'target': a['target'], 'target_variant': a['variant'],
                         'probe': probe, 'subset': subset, 'metric': metric, 'score': score,
                         'ci_low': lo, 'ci_high': hi, 'p_perm': perm_p_greater(score, null),
                         'n_train': len(a['tr']), 'n_validation': info.get('n_validation', np.nan),
                         'n_test': n_test if n_test is not None else len(a['te']),
                         'hparam': info.get('hparam', np.nan), 'hparam_at_grid_edge': info.get('at_edge', False),
                         'cv_score': info.get('cv_score', np.nan), 'tuning': info.get('tuning', ''),
                         'n_not_converged': info.get('n_not_converged', 0)})

    def metrics_for(a, pred, subset_idx=None):
        y = a['y_te'] if subset_idx is None else a['y_te'][subset_idx]
        p = pred if subset_idx is None else pred[subset_idx]
        key = f"{a['mask_key']}|{'all' if subset_idx is None else 'sub'}"
        boot, perm = resampling(seed, key if subset_idx is None else key + str(len(y)), len(y), B, P)
        if a['kind'] == 'reg':
            return regression_metrics(y, p, float(a['y_tr'].mean()), boot, perm)
        return classification_metrics(y, p, boot, perm)

    # ── non-representational baselines ──────────────────────────────────────
    for a in analyses:
        b_preds = {'train_mean': np.full(len(a['te']), a['y_tr'].mean())} if a['kind'] == 'reg' else {}
        for bname, covs in a['baselines'].items():
            if a['variant'] == 'controlled':
                continue
            C = covariate_design(wt, n_sub, covs)
            ok_tr, ok_te = np.isfinite(C[a['tr']]).all(1), np.isfinite(C[a['te']]).all(1)
            if not ok_te.all() or not ok_tr.all():
                logger.warning(f"baseline {bname} for {a['id']}: covariates missing; skipped")
                continue
            if a['kind'] == 'reg':
                _, b_preds[bname] = ols_fit_predict(C[a['tr']], a['y_tr'], C[a['te']])
            else:
                mu, sd = C[a['tr']].mean(0), C[a['tr']].std(0) + 1e-12
                lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=5000)
                lr.fit((C[a['tr']] - mu) / sd, a['y_tr'])
                b_preds[bname] = lr.predict_proba((C[a['te']] - mu) / sd)[:, 1]
        for bname, bp in b_preds.items():
            mets = metrics_for(a, bp)
            emit(a, -1, f'baseline:{bname}', mets, {'tuning': 'OLS / logistic on covariates (train)'})
            pm = 'r2' if a['kind'] == 'reg' else 'balanced_accuracy'
            store[(a['id'], pm, f'baseline:{bname}')] = {'score': mets[pm][0], 'boot': mets[pm][1]}

    # ── layerwise probes ────────────────────────────────────────────────────
    prim = {a['id']: (['r2'] if a['kind'] == 'reg' else ['balanced_accuracy', 'auroc']) for a in analyses}
    for a in analyses:
        for m in prim[a['id']]:
            store[(a['id'], m)] = {'score': np.full(L + 1, np.nan), 'boot': np.full((L + 1, B), np.nan),
                                   'cv': np.full(L + 1, np.nan)}
    single = n_sub == 1
    t0 = time.time()
    for layer in range(L + 1):
        X = np.asarray(feats[layer], dtype=np.float64)
        mu, sd = X[is_tr].mean(0), X[is_tr].std(0)
        sd[sd < 1e-8] = 1.0
        Z = (X - mu) / sd
        G = Z @ Z.T
        spaces = {}
        for a in analyses:
            if a['mask_key'] not in spaces:
                spaces[a['mask_key']] = ProbeSpace(G[np.ix_(a['tr'], a['tr'])], G[np.ix_(a['te'], a['tr'])])
            S = spaces[a['mask_key']]
            if a['kind'] == 'reg':
                pred, info = S.ridge(a['y_tr'], cfg.ridge_alphas_rel)
                info['n_validation'] = 1
            else:
                pred, info = S.logistic(a['y_tr'], cfg.logistic_Cs_rel, seed, cfg.inner_folds)
                info['n_validation'] = int(round(len(a['tr']) / cfg.inner_folds))
            mets = metrics_for(a, pred)
            emit(a, layer, 'linear_probe', mets, info)
            for m in prim[a['id']]:
                st = store[(a['id'], m)]
                st['score'][layer], st['boot'][layer], st['cv'][layer] = mets[m][0], mets[m][1], info['cv_score']
            # tokenisation-stratified point estimates (no CI; descriptive)
            for sname, smask in (('single_token', single[a['te']]), ('multi_token', ~single[a['te']])):
                idx = np.where(smask)[0]
                if len(idx) >= cfg.min_subset_n and (a['kind'] == 'reg' or len(np.unique(a['y_te'][idx])) == 2):
                    pm = prim[a['id']][0]
                    yv, pv = a['y_te'][idx], pred[idx]
                    sc = float(_r2(yv, pv)) if a['kind'] == 'reg' else float(_bacc(yv, pv >= 0.5)[0])
                    emit(a, layer, 'linear_probe', {pm: (sc, None, None)}, info, subset=sname, n_test=len(idx))
            preds.append(pd.DataFrame({'word': wt['word'].values[a['te']], 'analysis': a['id'],
                                       'layer': layer, 'y_true': a['y_te'], 'y_pred': pred}))
        if layer % 8 == 0 or layer == L:
            logger.info(f"  {spec.name} [{cond}] seed={seed} layer {layer}/{L} ({time.time() - t0:.0f}s)")

    # ── holm across layers; CV-selected layer; summaries ───────────────────
    df = pd.DataFrame(rows)
    df['p_holm_layers'] = np.nan
    sel = (df['probe'] == 'linear_probe') & (df['subset'] == 'all') & df['p_perm'].notna()
    for _, g in df[sel].groupby(['task', 'target', 'target_variant', 'metric']):
        df.loc[g.index, 'p_holm_layers'] = holm(g['p_perm'].values)
    # attach the association test (pearson_r / auroc permutation, Holm over
    # layers) to every metric row of the same probe and layer
    keys = ['task', 'target', 'target_variant', 'layer']
    assoc = df[sel & (df['metric'] == df['task'].map(ASSOCIATION_METRIC))][keys + ['p_holm_layers']]
    df = df.merge(assoc.rename(columns={'p_holm_layers': 'p_holm_association'}), on=keys, how='left')
    df.loc[(df['probe'] != 'linear_probe') | (df['subset'] != 'all'), 'p_holm_association'] = np.nan
    extra = []
    for a in analyses:
        for m in prim[a['id']]:
            st = store[(a['id'], m)]
            k = int(np.nanargmax(st['cv']))
            for label, val, bt, lay in (
                    ('cv_selected_layer', st['score'][k], st['boot'][k], k),
                    ('final_layer', st['score'][L], st['boot'][L], L),
                    ('layer_mean_1..L', np.nanmean(st['score'][1:]), np.nanmean(st['boot'][1:], 0), np.nan)):
                lo, hi = ci_of(bt, cfg.ci)
                extra.append({**common, 'layer': lay, 'rel_depth': lay / L if np.isfinite(lay) else np.nan,
                              'task': a['task'], 'target': a['target'], 'target_variant': a['variant'],
                              'probe': f'summary:{label}', 'subset': 'all', 'metric': m, 'score': float(val),
                              'ci_low': lo, 'ci_high': hi, 'n_train': len(a['tr']), 'n_test': len(a['te'])})
            st['cv_selected'] = k
    df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    # share of target variance explained by the covariates on TRAINING words
    ctrl = {a['target']: a['controls_r2_train'] for a in analyses if a['variant'] == 'controlled'}
    df['controls_r2_train'] = np.where(df['target_variant'] == 'controlled', df['target'].map(ctrl), np.nan)
    pdf = pd.concat(preds, ignore_index=True)
    pdf.insert(0, 'model', spec.name)
    pdf.insert(1, 'condition_id', cond)
    pdf.insert(2, 'seed', seed)
    return df, pdf, store, tinfo


def within_model_contrasts(spec, cond, seed, store, L, cfg) -> list:
    """Contextual gain over the static embedding and over covariate baselines."""
    out = []
    for (aid, metric, *rest), st in store.items():
        if rest:
            continue
        k = st['cv_selected']
        comps = [('layer_mean_1..L - embedding', np.nanmean(st['boot'][1:], 0) - st['boot'][0],
                  np.nanmean(st['score'][1:]) - st['score'][0]),
                 ('final_layer - embedding', st['boot'][L] - st['boot'][0], st['score'][L] - st['score'][0]),
                 ('cv_selected_layer - embedding', st['boot'][k] - st['boot'][0], st['score'][k] - st['score'][0])]
        for key, bst in store.items():
            if len(key) == 3 and key[0] == aid and key[1] == metric:
                comps.append((f'cv_selected_layer - {key[2]}', st['boot'][k] - bst['boot'],
                              st['score'][k] - bst['score']))
        for label, d, val in comps:
            lo, hi = ci_of(d, cfg.ci)
            out.append({'comparison': 'within_model', 'pair': spec.pair, 'model': spec.name,
                        'condition_id': cond, 'seed': seed, 'analysis': aid, 'metric': metric,
                        'contrast': label, 'layer': k if 'cv_selected' in label else np.nan,
                        'delta': float(val), 'ci_low': lo, 'ci_high': hi, 'p_boot': boot_p_two_sided(d)})
    return out


def base_vs_instruct(pair, cond, seed, st_b, st_i, L, cfg, label='instruct_minus_base') -> list:
    """Paired contrast st_i - st_b (same test words, same bootstrap draws)."""
    out = []
    for key in st_b:
        if len(key) != 2 or key not in st_i:
            continue
        aid, metric = key
        b, i = st_b[key], st_i[key]
        per_layer = []
        for l in range(L + 1):
            d = i['boot'][l] - b['boot'][l]
            lo, hi = ci_of(d, cfg.ci)
            per_layer.append({'comparison': label, 'pair': pair, 'model': pair,
                              'condition_id': cond, 'seed': seed, 'analysis': aid, 'metric': metric,
                              'contrast': 'per_layer', 'layer': l, 'delta': float(i['score'][l] - b['score'][l]),
                              'ci_low': lo, 'ci_high': hi, 'p_boot': boot_p_two_sided(d)})
        ph = holm([r['p_boot'] for r in per_layer])
        for r, p in zip(per_layer, ph):
            r['p_holm_layers'] = float(p)
        out += per_layer
        kb, ki = b['cv_selected'], i['cv_selected']
        for contrast, d, val in (
                ('layer_mean_1..L', np.nanmean(i['boot'][1:], 0) - np.nanmean(b['boot'][1:], 0),
                 np.nanmean(i['score'][1:]) - np.nanmean(b['score'][1:])),
                ('final_layer', i['boot'][L] - b['boot'][L], i['score'][L] - b['score'][L]),
                (f'cv_selected (L{kb} vs L{ki})', i['boot'][ki] - b['boot'][kb],
                 i['score'][ki] - b['score'][kb])):
            lo, hi = ci_of(d, cfg.ci)
            out.append({'comparison': label, 'pair': pair, 'model': pair,
                        'condition_id': cond, 'seed': seed, 'analysis': aid, 'metric': metric,
                        'contrast': contrast, 'layer': np.nan, 'delta': float(val), 'ci_low': lo,
                        'ci_high': hi, 'p_boot': boot_p_two_sided(d)})
    return out


# ════════════════════════════════════════════════════════════════════════════
# EXPLORATORY: ambiguity contrasts on the human-normed Rodd et al. (2002) items
# ════════════════════════════════════════════════════════════════════════════
def ambiguity_exploratory(spec, cond, feats, toks, wt, cfg) -> pd.DataFrame:
    if 'ambiguity' not in wt or not wt['is_published'].any():
        return pd.DataFrame()
    n_sub = toks['n_subtokens'].values.astype(float)
    has_f = wt['log_subtlex'].notna().values
    pub = wt['is_published'].values & wt['ambiguity'].notna().values & has_f
    labels = {'multiple_meanings (ambiguous vs unambiguous)':
              (pub, (wt['ambiguity'] == 'ambiguous').values)}
    if 'senses' in wt:
        labels['many_vs_few_senses'] = (pub & wt['senses'].notna().values, (wt['senses'] == 'many').values)
    cov_all = np.column_stack([wt['log_subtlex'].values, wt['length'].values, n_sub])
    L = feats.shape[0] - 1
    rows = []
    for lname, (mask, yfull) in labels.items():
        idx = np.where(mask)[0]
        y = yfull[idx].astype(int)
        if len(idx) < 40 or np.bincount(y).min() < 10:
            continue
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=cfg.ambiguity_repeats, random_state=cfg.seed)
        splits = list(cv.split(idx, y))
        boot, _ = resampling(cfg.seed, f'amb|{lname}', len(idx), cfg.n_boot, 1)
        C = cov_all[idx]

        def score_rows(oof, layer, variant):
            s = np.nanmean(oof, 0)
            per_rep = [_auroc(y, np.nanmean(oof[r * 5:(r + 1) * 5], 0))[0] for r in range(cfg.ambiguity_repeats)]
            for metric, val, bt in (('auroc', _auroc(y, s)[0], _auroc(y[boot], s[boot])),
                                    ('balanced_accuracy', _bacc(y, s >= 0.5)[0], _bacc(y[boot], s[boot] >= 0.5))):
                lo, hi = ci_of(bt, cfg.ci)
                rows.append({'model': spec.name, 'pair': spec.pair, 'condition': spec.variant,
                             'condition_id': cond, 'label': lname, 'features': variant, 'layer': layer,
                             'rel_depth': layer / L if layer >= 0 else np.nan, 'metric': metric,
                             'score': float(val), 'ci_low': lo, 'ci_high': hi,
                             'sd_over_repeats': float(np.std(per_rep)) if metric == 'auroc' else np.nan,
                             'n_words': len(idx), 'n_positive': int(y.sum()),
                             'status': 'EXPLORATORY (n small; repeated nested CV; not pre-registered)'})

        oof = np.full((len(splits), len(idx)), np.nan)
        for r, (tr, te) in enumerate(splits):
            mu, sd = C[tr].mean(0), C[tr].std(0) + 1e-12
            lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=5000)
            lr.fit((C[tr] - mu) / sd, y[tr])
            oof[r, te] = lr.predict_proba((C[te] - mu) / sd)[:, 1]
        score_rows(oof, -1, 'baseline:covariates(freq,length,n_subtokens)')
        for layer in range(L + 1):
            X = np.asarray(feats[layer][idx], dtype=np.float64)
            for variant in ('raw', 'covariates_linearly_erased'):
                oof = np.full((len(splits), len(idx)), np.nan)
                for r, (tr, te) in enumerate(splits):
                    mu, sd = X[tr].mean(0), X[tr].std(0)
                    sd[sd < 1e-8] = 1.0
                    Z = (X - mu) / sd
                    if variant != 'raw':          # remove linear covariate information (train-fit)
                        cm, cs = C[tr].mean(0), C[tr].std(0) + 1e-12
                        D = np.column_stack([np.ones(len(C)), (C - cm) / cs])
                        beta, *_ = np.linalg.lstsq(D[tr], Z[tr], rcond=None)
                        Z = Z - D @ beta
                    G = Z @ Z.T
                    S = ProbeSpace(G[np.ix_(tr, tr)], G[np.ix_(te, tr)])
                    oof[r, te], _ = S.logistic(y[tr], cfg.logistic_Cs_rel, cfg.seed + r, 3)
                score_rows(oof, layer, variant)
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# FIGURES
# ════════════════════════════════════════════════════════════════════════════
COLORS = {'base': '#2a78d6', 'instruct': '#eb6834', 'random_init': '#1baf7a'}
STYLES = {'base': '-', 'instruct': '--', 'random_init': ':'}
LABELS = {'base': 'base', 'instruct': 'instruct', 'random_init': 'random-init control'}


def _panel(ax, res, pair, cond, aid_parts, metric, seed, ref, ylab, title, show_baselines=True):
    import matplotlib.pyplot as plt  # noqa: F401
    task, target, variant = aid_parts
    q = res[(res['pair'] == pair) & (res['frame'] + '|' + res['extraction_position'] == cond) &
            (res['seed'] == seed) & (res['task'] == task) & (res['target'] == target) &
            (res['target_variant'] == variant) & (res['metric'] == metric) & (res['subset'] == 'all')]
    n_test = None
    for cond_name in ('base', 'instruct', 'random_init'):
        g = q[(q['condition'] == cond_name) & (q['probe'] == 'linear_probe')].sort_values('layer')
        if g.empty:
            continue
        n_test = int(g['n_test'].iloc[0])
        ax.fill_between(g['layer'], g['ci_low'], g['ci_high'], color=COLORS[cond_name], alpha=0.16, lw=0)
        ax.plot(g['layer'], g['score'], STYLES[cond_name], color=COLORS[cond_name], lw=2,
                label=LABELS[cond_name])
        sig = (g['p_holm_association'] < 0.05) & (g['ci_low'] > ref)
        ax.plot(g.loc[sig, 'layer'], g.loc[sig, 'score'], 'o', ms=3.5, color=COLORS[cond_name])
    if show_baselines:
        b = q[(q['condition'] == 'base') & q['probe'].str.startswith('baseline:') &
              ~q['probe'].eq('baseline:train_mean')]
        for k, (_, r) in enumerate(b.iterrows()):
            ax.axhline(r['score'], color='#6b6a66', lw=1, ls=[':', '-.'][k % 2],
                       label=r['probe'].replace('baseline:', 'covariates: '))
    ax.axhline(ref, color='#b4b3ad', lw=0.8)
    ax.set_title(title + (f'  (n_test={n_test})' if n_test else ''), fontsize=9)
    ax.set_ylabel(ylab, fontsize=8)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[['top', 'right']].set_visible(False)


def make_figures(res: pd.DataFrame, comp: pd.DataFrame, ckadf: pd.DataFrame, out: Path, seed: int):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig_dir = out / 'figures'
    fig_dir.mkdir(exist_ok=True)
    pairs = list(dict.fromkeys(res['pair']))
    conds = list(dict.fromkeys(res['frame'] + '|' + res['extraction_position']))
    panels_main = [
        (('freq_group', 'log_subtlex', 'raw'), 'balanced_accuracy', 0.5, 'balanced accuracy',
         'A  high vs low frequency'),
        (('regression', 'log_subtlex', 'raw'), 'r2', 0.0, 'held-out R²', 'B  log_subtlex (continuous)'),
        (('regression', 'wn_n_all', 'raw'), 'r2', 0.0, 'held-out R²', 'C1  WordNet synset count'),
        (('regression', 'gcide_etymologies', 'raw'), 'r2', 0.0, 'held-out R²', 'C2  GCIDE etymology count'),
    ]
    panels_ctrl = [
        (('regression', 'log_subtlex', 'controlled'), 'r2', 0.0, 'partial R²', 'log_subtlex | surface'),
        (('regression', 'wn_n_all', 'controlled'), 'r2', 0.0, 'partial R²', 'WordNet | surface+freq+etym'),
        (('regression', 'gcide_etymologies', 'controlled'), 'r2', 0.0, 'partial R²', 'etymologies | surface+freq+WN'),
    ]
    for cond in conds:
        for tag, panels in (('main', panels_main), ('controlled', panels_ctrl)):
            fig, axes = plt.subplots(len(pairs), len(panels), figsize=(3.6 * len(panels), 2.6 * len(pairs)),
                                     squeeze=False)
            for r, pair in enumerate(pairs):
                for c, (parts, metric, ref, ylab, title) in enumerate(panels):
                    _panel(axes[r, c], res, pair, cond, parts, metric, seed, ref, ylab,
                           f'{pair}: {title}' if c == 0 else title, show_baselines=(tag == 'main'))
                    if r == len(pairs) - 1:
                        axes[r, c].set_xlabel('layer (0 = embeddings)', fontsize=8)
            h, l = axes[0, 0].get_legend_handles_labels()
            h2, l2 = axes[0, -1].get_legend_handles_labels()
            seen = dict(zip(l + l2, h + h2))
            fig.legend(seen.values(), seen.keys(), loc='upper center', ncol=len(seen), fontsize=8,
                       frameon=False, bbox_to_anchor=(0.5, 1.0))
            fig.suptitle(f'Layerwise recoverability [{cond}] seed={seed}; band = 95% paired-bootstrap CI; '
                         f'dots = Holm-adjusted association p<.05 AND CI above chance', fontsize=9, y=1.03)
            fig.tight_layout()
            for ext in ('png', 'pdf'):
                fig.savefig(fig_dir / f'{tag}__{cond.replace("|", "_")}__seed{seed}.{ext}',
                            dpi=200, bbox_inches='tight')
            plt.close(fig)
        # base-vs-instruct deltas + CKA
        cq = comp[(comp['comparison'] == 'instruct_minus_base') & (comp['contrast'] == 'per_layer') &
                  (comp['condition_id'] == cond) & (comp['seed'] == seed)]
        dpan = [('freq_group|log_subtlex|raw', 'balanced_accuracy', 'Δ balanced acc (freq group)'),
                ('regression|log_subtlex|raw', 'r2', 'Δ R² log_subtlex'),
                ('regression|wn_n_all|raw', 'r2', 'Δ R² WordNet'),
                ('regression|gcide_etymologies|raw', 'r2', 'Δ R² etymologies')]
        dpairs = [p for p in pairs if p in set(cq['pair'])]
        if not dpairs:
            continue
        fig, axes = plt.subplots(len(dpairs), len(dpan) + 1, figsize=(3.4 * (len(dpan) + 1), 2.5 * len(dpairs)),
                                 squeeze=False)
        for r, pair in enumerate(dpairs):
            for c, (aid, metric, title) in enumerate(dpan):
                ax = axes[r, c]
                g = cq[(cq['pair'] == pair) & (cq['analysis'] == aid) & (cq['metric'] == metric)].sort_values('layer')
                if not g.empty:
                    ax.fill_between(g['layer'], g['ci_low'], g['ci_high'], color='#6250d6', alpha=0.18, lw=0)
                    ax.plot(g['layer'], g['delta'], color='#6250d6', lw=2)
                    sig = g['p_holm_layers'] < 0.05
                    ax.plot(g.loc[sig, 'layer'], g.loc[sig, 'delta'], 'o', ms=3, color='#6250d6')
                ax.axhline(0, color='#b4b3ad', lw=0.8)
                ax.set_title(f'{pair}: {title}' if c == 0 else title, fontsize=9)
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.25, lw=0.5)
                ax.spines[['top', 'right']].set_visible(False)
            ax = axes[r, -1]
            g = ckadf[(ckadf['pair'] == pair) & (ckadf['condition_id'] == cond)].sort_values('layer')
            if not g.empty:
                ax.plot(g['layer'], g['cka_debiased'], color='#1baf7a', lw=2)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title('CKA(base, instruct) (debiased)', fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25, lw=0.5)
            ax.spines[['top', 'right']].set_visible(False)
        for c in range(len(dpan) + 1):
            axes[-1, c].set_xlabel('layer', fontsize=8)
        fig.suptitle(f'Instruct − base (paired bootstrap 95% CI; dots = Holm p<.05) [{cond}]', fontsize=9, y=1.01)
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(fig_dir / f'instruct_minus_base__{cond.replace("|", "_")}__seed{seed}.{ext}',
                        dpi=200, bbox_inches='tight')
        plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ════════════════════════════════════════════════════════════════════════════
def run_group(pair: str, specs: list, wt, splits, cfg: Config, out: Path):
    """Analyse one base model and (if present) its instruct counterpart."""
    pdir = out / 'pairs' / safe_name(pair)
    done = pdir / 'DONE'
    if done.exists() and not cfg.force:
        logger.info(f"{pair}: analysis already complete (use --force to recompute)")
        return
    pdir.mkdir(parents=True, exist_ok=True)
    words = wt['word'].tolist()
    conds = [f'{f}|{p}' for f in cfg.frames for p in PROMPT_FRAMES[f]['positions']]
    all_res, all_comp, all_cka, all_amb, thr_info = [], [], [], [], {}
    tok_check = {}
    for cond in conds:
        frame, pos = cond.split('|')
        loaded = {s.variant: (s, *load_features(s, frame, pos, out, words)) for s in specs}
        paired = 'base' in loaded and 'instruct' in loaded
        tok_check[cond] = {s.name: {'n_subtokens_distribution': t['n_subtokens'].value_counts().sort_index().to_dict(),
                                    'n_span_not_clean': int((~t['span_clean']).sum())}
                           for s, _, _, t in loaded.values()}
        if paired:
            (sb, fb, mb, tb), (si, fi, mi, ti) = loaded['base'], loaded['instruct']
            if fb.shape[0] != fi.shape[0]:
                raise RuntimeError(f"{pair}: base and instruct differ in layer count")
            same = float((tb['word_token_ids'] == ti['word_token_ids']).mean())
            tok_check[cond]['fraction_words_identically_tokenised_base_vs_instruct'] = same
            if same < 1.0:
                logger.warning(f"{pair}/{cond}: {1 - same:.1%} of words tokenised differently in base vs instruct")
        for seed in cfg.split_seeds:
            stores = {}
            for s, f, m, t in loaded.values():
                L = f.shape[0] - 1
                logger.info(f"probing {s.name} [{cond}] seed={seed}")
                df, pdf, store, tinfo = analyze_condition(s, cond, f, t, wt, splits[seed], seed, cfg, m)
                thr_info[seed] = tinfo
                all_res.append(df)
                pdf.to_csv(pdir / f'predictions__{safe_name(s.name)}__{cond.replace("|", "_")}__seed{seed}.csv.gz',
                           index=False)
                all_comp += within_model_contrasts(s, cond, seed, store, L, cfg)
                stores[s.variant] = store
            if paired:
                all_comp += base_vs_instruct(pair, cond, seed, stores['base'], stores['instruct'],
                                             fb.shape[0] - 1, cfg)
            for v in ('base', 'instruct'):
                if v in stores and 'random_init' in stores:
                    all_comp += base_vs_instruct(pair, cond, seed, stores['random_init'], stores[v],
                                                 loaded[v][1].shape[0] - 1, cfg, label=f'{v}_minus_random_init')
        if paired:
            L = fb.shape[0] - 1
            for layer in range(L + 1):
                deb, std = cka(fb[layer], fi[layer])
                all_cka.append({'pair': pair, 'condition_id': cond, 'layer': layer, 'rel_depth': layer / L,
                                'cka_debiased': deb, 'cka_linear': std, 'n_words': len(words)})
        if cfg.run_ambiguity and cond in cfg.ambiguity_conditions:
            for s, f, m, t in loaded.values():
                logger.info(f"exploratory ambiguity probes: {s.name} [{cond}]")
                all_amb.append(ambiguity_exploratory(s, cond, f, t, wt, cfg))
    pd.concat(all_res, ignore_index=True).to_csv(pdir / 'results_long.csv', index=False)
    pd.DataFrame(all_comp).to_csv(pdir / 'comparisons.csv', index=False)
    if all_cka:
        pd.DataFrame(all_cka).to_csv(pdir / 'cka.csv', index=False)
    if all_amb:
        pd.concat(all_amb, ignore_index=True).to_csv(pdir / 'ambiguity_exploratory.csv', index=False)
    (pdir / 'tokenization_check.json').write_text(json.dumps(tok_check, indent=2))
    (pdir / 'thresholds.json').write_text(json.dumps(thr_info, indent=2))
    done.write_text(time.strftime('%Y-%m-%d %H:%M:%S'))


def aggregate(out: Path, cfg: Config, pairs: list):
    def cat(name):
        fs = [out / 'pairs' / safe_name(p) / name for p in pairs]
        fs = [f for f in fs if f.exists() and f.stat().st_size > 2]
        return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
    res, comp, ck, amb = cat('results_long.csv'), cat('comparisons.csv'), cat('cka.csv'), cat('ambiguity_exploratory.csv')
    if res.empty:
        logger.warning("no results to aggregate")
        return
    res.to_csv(out / 'results_long.csv', index=False)
    comp.to_csv(out / 'comparisons.csv', index=False)
    ck.to_csv(out / 'cka.csv', index=False)
    if not amb.empty:
        amb.to_csv(out / 'ambiguity_exploratory.csv', index=False)
    # compact summary (primary metrics; summaries + baselines)
    keep = res['probe'].str.startswith('summary:') | (res['probe'].str.startswith('baseline:') &
                                                     res['metric'].isin(['r2', 'balanced_accuracy', 'auroc']))
    summ = res[keep & (res['subset'] == 'all')]
    summ = summ[['model', 'pair', 'condition', 'frame', 'extraction_position', 'seed', 'task', 'target',
                 'target_variant', 'probe', 'layer', 'metric', 'score', 'ci_low', 'ci_high', 'n_train', 'n_test']]
    summ.to_csv(out / 'summary.csv', index=False)
    if len(cfg.split_seeds) > 1:
        rob = (summ.groupby(['model', 'frame', 'extraction_position', 'task', 'target', 'target_variant',
                             'probe', 'metric'])['score'].agg(['mean', 'std', 'count']).reset_index())
        rob.to_csv(out / 'summary_across_split_seeds.csv', index=False)
    make_figures(res, comp, ck, out, cfg.split_seeds[0])


def write_notes(out: Path, cfg: Config, rep: dict, pairs: list):
    lines = ['# Reproducibility notes (auto-generated)', '',
             f'- generated: {time.strftime("%Y-%m-%d %H:%M:%S")}',
             f'- python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}',
             f'- data: {rep["source"]}; {rep["n_rows_raw"]} rows -> {rep["n_unique_words"]} unique words',
             f'- column mapping: `{json.dumps(rep["column_mapping"])}`',
             f'- log_subtlex check: {rep["log_subtlex_check"]["conclusion"]}',
             f'- split seeds: {list(cfg.split_seeds)}; test fraction {cfg.test_frac}',
             f'- model pairs: {pairs}', '', '## Prompt frames (identical text for base and instruct)', '']
    for f in cfg.frames:
        lines += [f'### {f} ({PROMPT_FRAMES[f]["role"]}, id `{frame_prompt_id(f)}`)', '```text',
                  PROMPT_FRAMES[f]['text'], '```',
                  f'positions: {", ".join(PROMPT_FRAMES[f]["positions"])}', '']
    lines += ['Rendered prompts and token ids for every word/model: `features/<model>/<frame>__tokens.csv.gz`;',
              'model checkpoint, commit hash, dtype, layer count, versions: `features/<model>/<frame>__meta.json`.',
              '', '## Files', '',
              '- `results_long.csv` long-form results (spec §9 fields + p_perm, p_holm_layers, hparam, cv_score)',
              '- `comparisons.csv` instruct−base and within-model paired-bootstrap contrasts',
              '- `cka.csv` debiased / standard linear CKA between base and instruct per layer',
              '- `ambiguity_exploratory.csv` EXPLORATORY Rodd (2002) item analyses',
              '- `pairs/<pair>/predictions__*.csv.gz` per-word test predictions per layer',
              '- `figures/` main, controlled, and instruct−base figures', '',
              '## Notes', ''] + [f'- {n}' for n in rep['notes']]
    (out / 'REPRODUCIBILITY.md').write_text('\n'.join(lines) + '\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=Config.data)
    ap.add_argument('--out', default=Config.out_dir)
    ap.add_argument('--models', default=None,
                    help='comma-separated subset of default_models() names (default: all 15)')
    ap.add_argument('--custom-pair', action='append', default=[], metavar='NAME:BASE_ID:INSTRUCT_ID',
                    help='add a base/instruct pair (HF ids or local paths); repeatable')
    ap.add_argument('--custom-model', action='append', default=[], metavar='NAME:MODEL_ID',
                    help='add a base-only model (HF id or local path); repeatable')
    ap.add_argument('--frames', default=','.join(Config.frames))
    ap.add_argument('--split-seeds', default='0')
    ap.add_argument('--n-boot', type=int, default=Config.n_boot)
    ap.add_argument('--n-perm', type=int, default=Config.n_perm)
    ap.add_argument('--batch-size', type=int, default=None, help='override per-model batch size')
    ap.add_argument('--device', default=Config.device)
    ap.add_argument('--precision', default=Config.precision, choices=['auto', 'float32', 'bfloat16'])
    ap.add_argument('--feature-dtype', default=Config.feature_dtype, choices=['float16', 'float32'])
    ap.add_argument('--system-message', default=None,
                    help='explicit system message for instruct models (default: template default)')
    ap.add_argument('--max-words', type=int, default=None, help='debug: random subset of words')
    ap.add_argument('--ambiguity-repeats', type=int, default=Config.ambiguity_repeats)
    ap.add_argument('--skip-ambiguity', action='store_true')
    ap.add_argument('--random-init-control', action='store_true',
                    help='also probe an untrained copy of every base architecture (same tokenizer)')
    ap.add_argument('--extract-only', action='store_true')
    ap.add_argument('--analyze-only', action='store_true')
    ap.add_argument('--trust-remote-code', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()

    frames = tuple(f.strip() for f in args.frames.split(',') if f.strip())
    unknown = set(frames) - set(PROMPT_FRAMES)
    if unknown:
        ap.error(f'unknown frames {unknown}; choose from {list(PROMPT_FRAMES)}')
    cfg = Config(data=args.data, out_dir=args.out, frames=frames,
                 split_seeds=tuple(int(s) for s in args.split_seeds.split(',')),
                 n_boot=args.n_boot, n_perm=args.n_perm, batch_size=args.batch_size, device=args.device,
                 precision=args.precision, feature_dtype=args.feature_dtype,
                 instruct_system_message=args.system_message, max_words=args.max_words,
                 run_ambiguity=not args.skip_ambiguity, ambiguity_repeats=args.ambiguity_repeats,
                 trust_remote_code=args.trust_remote_code, force=args.force)
    if PRIMARY_CONDITION.split('|')[0] not in frames:
        cfg.ambiguity_conditions = (f'{frames[0]}|word_last',)
    seed_everything(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    models = default_models()
    if args.models:
        want = [m.strip() for m in args.models.split(',') if m.strip()]
        known = {m.name: m for m in models}
        missing = [w for w in want if w not in known]
        if missing:
            ap.error(f'unknown models {missing}; available: {list(known)}')
        models = [known[w] for w in want]
    if args.custom_pair:
        custom = []
        for c in args.custom_pair:
            parts = c.split(':', 2)
            if len(parts) != 3:
                ap.error('--custom-pair must be NAME:BASE_ID:INSTRUCT_ID')
            n, b, i = parts
            custom += [ModelConfig(n, b), ModelConfig(f'{n}-Instruct', i, variant='instruct', base_of=n, chat=True)]
        models = (models if args.models else []) + custom
    if args.custom_model:
        extra = [ModelConfig(*c.split(':', 1)) for c in args.custom_model]
        models = (models if (args.models or args.custom_pair) else []) + extra
    if args.random_init_control:
        models = models + [ModelConfig(f'{m.name}-RandomInit', m.model_id, batch_size=m.batch_size,
                                       family=m.family, variant='random_init', base_of=m.name,
                                       revision=m.revision, random_init=True)
                           for m in models if m.variant == 'base']
    specs = group_models(models)

    wt, rep = load_and_validate(cfg)
    splits = {s: make_split(wt, s, cfg.test_frac) for s in cfg.split_seeds}
    words_out = wt.copy()
    for s, sp in splits.items():
        words_out[f'split_seed{s}'] = sp
        thr = float(np.nanmedian(wt['log_subtlex'].values[sp == 'train']))
        words_out[f'freq_group_seed{s}'] = np.where(wt['log_subtlex'].isna(), 'NA',
                                                    np.where(wt['log_subtlex'] > thr, 'high', 'low'))
        rep[f'split_seed{s}'] = {'n_train': int((sp == 'train').sum()), 'n_test': int((sp == 'test').sum()),
                                 'frequency_threshold_train_median': thr,
                                 'group_counts_train': words_out.loc[sp == 'train', f'freq_group_seed{s}'].value_counts().to_dict(),
                                 'group_counts_test': words_out.loc[sp == 'test', f'freq_group_seed{s}'].value_counts().to_dict()}
    words_out.to_csv(out / 'word_table_with_splits.csv', index=False)
    (out / 'data_report.json').write_text(json.dumps(rep, indent=2, default=str))
    (out / 'config.json').write_text(json.dumps({'config': cfg.to_json(),
                                                 'model_groups': {k: [asdict(s) for s in v] for k, v in specs.items()},
                                                 'prompt_frames': PROMPT_FRAMES, 'targets': TARGETS,
                                                 'controls': CONTROLS}, indent=2))
    logger.info(f"data: {rep['n_unique_words']} unique words; log_subtlex: {rep['log_subtlex_check']['conclusion']}")

    for pair, sp in specs.items():
        if not args.analyze_only:
            for s in sp:
                extract_model(s, wt, cfg, out)
        if not args.extract_only:
            run_group(pair, sp, wt, splits, cfg, out)
    if not args.extract_only:
        aggregate(out, cfg, list(specs))
        write_notes(out, cfg, rep, list(specs))
    logger.info(f"done -> {out.resolve()}")


# ════════════════════════════════════════════════════════════════════════════
# SELF-TEST (mathematical correctness of the estimators)
# ════════════════════════════════════════════════════════════════════════════
def self_test():
    from sklearn.linear_model import Ridge
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
    rng = np.random.default_rng(0)
    n, d, m = 60, 150, 25
    Xtr, Xte = rng.normal(size=(n, d)), rng.normal(size=(m, d))
    w = rng.normal(size=d) / np.sqrt(d)
    ytr, yte = Xtr @ w + 0.3 * rng.normal(size=n) + 2.0, Xte @ w + 2.0
    S = ProbeSpace(Xtr @ Xtr.T, Xte @ Xtr.T)
    # 1. centred Gram identity
    A = Xtr - Xtr.mean(0)
    assert np.allclose(S.K_te, (Xte - Xtr.mean(0)) @ A.T)
    # 2. dual ridge == sklearn Ridge at the selected penalty
    pred, info = S.ridge(ytr, [1e-3, 1e-2, 1e-1, 1.0, 10.0])
    alpha = info['hparam'] * S.scale
    ref = Ridge(alpha=alpha).fit(Xtr, ytr).predict(Xte)
    assert np.allclose(pred, ref, atol=1e-8), 'dual ridge != sklearn Ridge'
    # 3. closed-form LOO == brute-force LOO
    lam, U = S.lam, S.U
    F = lam / (lam + alpha)
    yc = ytr - ytr.mean()
    fitted = ytr.mean() + U @ (F * (U.T @ yc))
    H = 1 / n + (U ** 2) @ F
    loo_fast = (ytr - fitted) / (1 - H)
    loo_slow = np.array([ytr[i] - Ridge(alpha=alpha).fit(np.delete(Xtr, i, 0), np.delete(ytr, i))
                         .predict(Xtr[i:i + 1])[0] for i in range(n)])
    assert np.allclose(loo_fast, loo_slow, atol=1e-8), 'LOO shortcut is wrong'
    # 4. row-space logistic == full-space logistic (same objective)
    yb = (ytr > np.median(ytr)).astype(int)
    R_tr, R_te = S.row_space()
    C_rel = 0.5
    c = np.sqrt(S.scale)
    p_row = LogisticRegression(C=C_rel, tol=1e-10, max_iter=100000).fit(R_tr, yb).predict_proba(R_te)[:, 1]
    p_full = LogisticRegression(C=C_rel, tol=1e-10, max_iter=100000).fit(Xtr / c, yb).predict_proba(Xte / c)[:, 1]
    assert np.allclose(p_row, p_full, atol=1e-5), 'row-space logistic not equivalent'
    # 5. vectorised metrics == sklearn
    yy = rng.integers(0, 2, 200)
    ss = rng.random(200) + 0.3 * yy
    assert np.isclose(_auroc(yy, ss)[0], roc_auc_score(yy, ss))
    assert np.isclose(_bacc(yy, ss >= 0.6)[0], balanced_accuracy_score(yy, ss >= 0.6))
    assert np.isclose(_f1(yy, ss >= 0.6)[0], f1_score(yy, ss >= 0.6))
    yr, pr = rng.normal(size=100), rng.normal(size=100)
    from sklearn.metrics import r2_score
    assert np.isclose(_r2(yr, pr), r2_score(yr, pr))
    # 6. Holm
    assert np.allclose(holm([0.01, 0.04, 0.03, 0.2]), [0.04, 0.09, 0.09, 0.2])
    # 7. CKA invariances
    X = rng.normal(size=(80, 30))
    Q, _ = np.linalg.qr(rng.normal(size=(30, 30)))
    assert np.isclose(cka(X, X)[0], 1.0) and np.isclose(cka(X, X)[1], 1.0)
    assert abs(cka(X, rng.normal(size=(80, 30)))[0]) < 0.15
    # 8. permutation null of a useless predictor is ~uniform; of a perfect one is minimal
    boot, perm = resampling(0, 'self', 100, 200, 999)
    met = regression_metrics(yr, yr, 0.0, boot, perm)
    assert perm_p_greater(met['pearson_r'][0], met['pearson_r'][2]) <= 0.002
    met0 = regression_metrics(yr, pr, 0.0, boot, perm)
    assert perm_p_greater(met0['pearson_r'][0], met0['pearson_r'][2]) > 0.01
    # 9. prompt frames: one placeholder, no leakage words
    for f in PROMPT_FRAMES.values():
        assert f['text'].count('{W}') == 1 and not _LEAKY.search(f['text'])
    print('self-test: all 9 checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
