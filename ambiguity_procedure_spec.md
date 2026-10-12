# Semantic Ambiguity: Procedure Spec for LLM Simulation

The paradigm below is paired with **the stimulus set that was actually run in it**, so the
task description and the items match. Every parameter was read from the published Methods
rather than recalled.

The simulation runs in **two stages**: first reproduce Rodd et al.'s published effects on
their own 309 items, then extend to 1,633 WordNet-derived items to test whether the result
survives at scale. Stage 2 is only interpretable if Stage 1 passes — see *The two-stage
design* below.

| Paradigm | Source study | Items | Task verified |
|---|---|---|---|
| 1. Visual lexical decision | Rodd, Gaskell & Marslen-Wilson (2002), Exp. 1–2 | 310 trials over 309 words (`Rodd2002-A/B`) | fully — from the article |
| 1b. Auditory lexical decision | Rodd, Gaskell & Marslen-Wilson (2002), Exp. 3 | 92 words (`Rodd2002-C`) | fully — from the article |

---

## 1. Lexical Decision — Rodd, Gaskell & Marslen-Wilson (2002)

**Source** Rodd, J., Gaskell, G., & Marslen-Wilson, W. (2002). Making sense of semantic
ambiguity: Semantic competition in lexical access. *Journal of Memory and Language*, 46(2),
245–266.

**Items** The 309 words in the pool with `source` starting `Rodd2002`. These are the items
this task was run on.

**Task** **Lexical decision** — visual in Experiments 1 and 2, auditory in Experiment 3.
Participants judged whether each item was a real English word.

**What it measures** The cost or benefit of a word simply having more than one meaning.
This is the design that produced the meanings/senses dissociation: multiple unrelated
*meanings* slow responses, multiple related *senses* speed them up.

### Design, verified against the printed appendices

| Set | Appendix | Experiment | N | Structure |
|---|---|---|---|---|
| `Rodd2002-A` | A | Experiment 1 | 182 | 124 ambiguous vs 58 unambiguous, no sense manipulation |
| `Rodd2002-B` | B | Experiment 2 | 128 | meanings × senses 2 × 2, 32 per cell |
| `Rodd2002-C` | C | Experiment 3 (auditory) | 92 | the same 2 × 2, 23 per cell |

All 309 unique items in the pool were checked word by word against Appendices A, B and C of
the article: membership, set assignment and 2 × 2 cell all match with no discrepancies.
These 309 words are the `provenance = published` rows of `ambiguity_items_combined.csv`,
where they carry 24 lexical-variable columns alongside the design labels — see the data
dictionary at the end of this document.

> Appendix A prints 58 unambiguous words, while Table 2 of the same article reports
> N = 60. The discrepancy is in the original. The 124 ambiguous words match exactly.

### Trial structure — Experiments 1 and 2 (visual)

| Step | Screen | Duration |
|---|---|---|
| 1 | fixation point, screen centre | 500 ms |
| 2 | **stimulus** (word or nonword) | until response — self-paced |
| 3 | next fixation point | appears the moment the response is made |

There is no fixed stimulus duration and no response deadline: the item is replaced by the
next fixation point as soon as the participant responds. Word responses were made with the
dominant hand, nonword responses with the other hand; the instruction was to respond as
quickly as possible without making mistakes.

**Blocking and lists.** Items were pseudorandomly divided into **four lists** with
approximately equal numbers of each stimulus type, then some items were swapped between
lists so that no ambiguous word shared a list with an item that would bias one of its
meanings. Lists were presented in a pseudorandom (Exp. 1) or random (Exp. 2) order, one per
block, with a break after each block; item order within a list was randomised per
participant. Every participant saw every item.

| | Experiment 1 | Experiment 2 |
|---|---|---|
| Participants | 25 (MRC CBU subject panel) | 25 (MRC CBU subject panel) |
| Practice items | 30 | 64 |
| Unanalysed items at block start | 5 | 10 |
| Nonword foils | pseudohomophones | pseudohomophones |
| Exclusions | 2 participants with mean RT > 1,000 ms; responses > 1,200 ms (1.2% of data) | — |
| Mean RT | unambiguous 556, ambiguous-unrelated 577, ambiguous-related 561, nonwords 636 | 587 / 578 / 586 / 567, nonwords 659 |

Pseudohomophone foils were a deliberate choice, following Azuma & Van Orden (1997), who
found an ambiguity effect with pseudohomophone foils but not with word-like nonwords. A
replication of Experiment 1 with word-like nonwords is reported in the article's footnote 4.

### Trial structure — Experiment 3 (auditory)

Same 2 × 2 design, **spoken** items, 77% overlap with Experiment 2, all monosyllabic.

| Step | Event | Timing |
|---|---|---|
| 1 | spoken item onset | 1,000 ms after the response to the preceding item |
| 2 | response window | 3,000 ms from word onset; after that the next item begins |

26 Cambridge University students. Blocking followed Experiment 2. Nonwords were word-like,
not pseudohomophones — this experiment was partly designed to show the effect survives
without them. Groups matched on CELEX log frequency, number of phonemes, uniqueness point,
actual duration in ms (601–610), concreteness and familiarity. Exclusions: 4 participants
with error rates above 10%; responses longer than 1,500 ms (2% of data). Mean RT 986 / 935 /
939 / 924, nonwords 1,031. Overall error rate 5.8%.

### Stimulus selection and matching, as published

- 113 of the 124 ambiguous words in Experiment 1 came from the **Twilley et al. (1994)**
  homograph norms; the other 11 were judged to have similar properties. Most had two
  meanings; words with a third meaning were admitted only when that meaning had a meaning
  probability below 0.1.
- Meanings vs senses were operationalised through the **Wordsmyth dictionary**: separate
  entries = separate meanings, senses within an entry = senses. WordNet sense counts were
  recorded in parallel.
- Experiment 2 cells were matched on CELEX log frequency, length in letters, syllable count,
  concreteness and familiarity; neighbourhood size was not matched explicitly but did not
  differ (p > .3).
- Pretest norms were collected on 7-point scales (Gilhooly & Logie, 1980 format) by at least
  24 raters per variable: concreteness (rated separately for each meaning, each cued by an
  associate such as *bark–dog* / *bark–tree*), familiarity, and relatedness of the two
  meanings. Mean relatedness was 2.64; the inverse of the rating was used in the analyses.
- Dominance followed the Twilley et al. procedure: 38 participants gave word associates,
  which were then classified by meaning; the dominance score is the proportion of valid
  responses matching the dominant meaning (1.0 = fully dominant, 0.5 = balanced).

> If a fixed-duration visual trial is needed instead of the self-paced one above, the
> documented alternative is Kim, Kim & Nam (2024), *Frontiers in Psychology*, 15, 1378125 —
> a Korean visual lexical decision study with ERP: fixation `***` 300 ms → blank 1,500 ms →
> target 1,500 ms → ITI with blink cue 1,500 ms, 4.8 s per trial, 76 targets plus 76
> pseudowords across 152 trials, 28 participants, N400 350–550 ms and P600 600–1000 ms.
> Borrowing those timings is defensible, but say so rather than attributing them to Rodd.

### LLM prompt template

```
You are a participant in a lexical decision experiment.
You will see one letter string at a time.
For each one, decide as quickly and accurately as you can
whether it is a real English word.

Respond in this exact format, one line per item:
<item> | <WORD or NONWORD> | <ease 1-7>

"ease" is how easy the decision felt: 1 = very hard, 7 = very easy.
Do not explain. Do not revise earlier answers.

Items:
[item list]
```

> This task needs nonword foils, which the pool does not contain. The original used
> **pseudohomophones** (nonwords that sound like real words: *brane*, *pleet*) in
> Experiments 1 and 2, matched to the targets in length distribution. Experiment 3 used
> ordinary word-like nonwords. Which type you generate matters: Azuma & Van Orden (1997)
> obtained the ambiguity effect only with pseudohomophones.
>
> The dependent variable in the original is reaction time, which an LLM cannot produce.
> `ease` is a stand-in whose relation to human RT is itself untested.

---

## The two-stage design

`ambiguity_items_combined.csv` is built for a simulation run in two stages. The two halves of
the file do different jobs and are not interchangeable.

| | Stage 1 — calibration | Stage 2 — extension |
|---|---|---|
| Items | `provenance == "published"` | `provenance == "wordnet-derived"` |
| Words | 309 (402 rows) | 1,633 |
| Design | factorial, 2 × 2 cells as published | regression on continuous predictors |
| Ground truth | Rodd et al.'s reported effects | none — this is the generalisation test |
| Question | *does the simulation reproduce the paper?* | *does the effect hold at scale?* |

Stage 2 is only interpretable if Stage 1 passes. Run them in order.

### Stage 1 — does the simulation reproduce Rodd et al. (2002)?

Use the published items in their original cells. Appendix B gives four cells of 32,
Appendix C four cells of 23; Appendix A gives 124 ambiguous against 58 unambiguous with no
sense manipulation.

```python
import pandas as pd
df = pd.read_csv("ambiguity_items_combined.csv")

stage1_factorial = df[(df.provenance == "published") & df.senses.notna()]   # 220 rows, B + C
stage1_regression = df[(df.provenance == "published") & (df.appendix == "A")]  # 182 rows
```

The targets to reproduce, all from the article:

| Source | Effect | Direction and size |
|---|---|---|
| Exp. 1 regression (n = 184) | ambiguity | β = +.18, t = 2.7 — ambiguous **slower** |
| Exp. 1 regression | word senses | β = −.17, t = −2.4 — many senses **faster** |
| Exp. 1 regression | meaning relatedness | β = +.17, t = 2.3 — unrelated meanings slower |
| Exp. 2, visual (n = 128) | senses | many senses 14 ms **faster**, significant |
| Exp. 2, visual | ambiguity | 6 ms slower, **not** significant |
| Exp. 3, auditory (n = 92) | senses | many senses 33 ms faster, significant |
| Exp. 3, auditory | ambiguity | 29 ms slower, significant |

**What counts as passing.** The sign pattern is the result, not the millisecond values: the
two factors must come apart, with the senses effect facilitatory and the ambiguity effect
inhibitory, in the same model. A simulation that produces one main effect of "ambiguity"
collapsing both has failed Stage 1 regardless of how large that effect is. The null
ambiguity effect in Experiment 2 is also informative — reproducing a significant one there
is a mismatch, not a bonus.

Because an LLM has no reaction time, the comparison is between *patterns*, not magnitudes.
Decide the stand-in measure before running (see the cautions below), and pre-register the
sign predictions so the check cannot be read post hoc.

### Stage 2 — extension to the derived pool

The derived items carry no condition labels, so Stage 2 cannot be a factorial replication.
It is a regression, and the two computed variables are what make it possible:

- **`wn_n_all`** — the senses predictor (r = .67 with the published senses factor)
- **`gcide_etymologies`** — the meanings predictor (r = .55 with the published meanings
  factor, −.06 with senses)

Both are available for the derived pool: 1,464 of the 1,633 derived words have
`gcide_etymologies`, and all have `wn_n_all`. Restricting to the published items' central
range — length 4–6, `log_subtlex` 2.0–4.0 — leaves **596 words with both predictors**, with
`gcide_etymologies` spread 1 (359 words), 2 (129), 3 (34), 4+ (20).

```python
stage2 = df[(df.provenance == "wordnet-derived") &
            df.gcide_etymologies.notna() &
            df.length.between(4, 6) &
            df.log_subtlex.between(2.0, 4.0)]            # 596 words

# the model Stage 1 licenses
# outcome ~ gcide_etymologies + wn_n_all + log_subtlex + length + concreteness
```

Collinearity in that pool is workable: `gcide_etymologies` × `wn_n_all` = .33,
`gcide_etymologies` × `log_subtlex` = .22, `wn_n_all` × `log_subtlex` = .45. Enter frequency
alongside both and report VIF.

**What Stage 2 can and cannot show.** It tests whether the dissociation found on 309 hand-picked
items survives on an order of magnitude more words that no one selected for the purpose — the
strongest available check against item-selection artifacts. It cannot be reported as a
replication of Rodd et al., because the items were never run on humans and the predictors are
computed rather than normed. Report it as a generalisation analysis on a derived pool, and
state the provenance split.

### Item screening, both stages

1. **Match the paradigm to the source.** Rodd items were run in lexical decision —
   `Rodd2002-A` and `-B` visually, `-C` auditorily.
2. **Match across conditions** on log frequency, length and concreteness. Sense count
   correlates with frequency at r ≈ .49, which makes frequency the hardest to match.
3. **Exclude inflected forms.** WordNet sense counts are inflated for plurals
   (stocks, heads, bases).
4. **Screen out emotionally extreme items.** Valence drives the EPN and the LPP separately
   and will contaminate the late windows.
5. **Decide what to do with `gcide_etymologies == 0`** — present in GCIDE, no etymology
   recorded — before fitting anything. Drop or recode to 1, and say which.
6. **Never mix the stages in one model.** `provenance` is confounded with length, frequency
   and neighbourhood size (see "The two sets are not interchangeable" below).

---

## Cautions for the simulation

**The dependent variable does not survive the move to an LLM.** Lexical decision is measured
in reaction time, and reaction time is the one thing a language model cannot produce. Any
simulation of this paradigm is testing a stand-in, so decide what the stand-in is and justify
it before running anything.

**An LLM has no reaction time.** Whether you use `ease`, token probability or surprisal, it is
a stand-in and the correspondence is itself the thing under test. Read the output as how the
model represents the stimulus structure, not as a prediction of human performance.

**Order effects behave differently.** The model sees every preceding item in the same context
window, so unless trials are issued as independent calls, priming and adaptation are uncontrolled.

**Raising temperature is not enough to create participant variance.** Run several
"participants" with different instructions and personas, and treat items as a within-subject
factor.

**Never expose condition labels in the prompt.** `condition`, `ambiguity`, `senses` and
`ambiguity_class` are the things being predicted.

---

## Data dictionary — `ambiguity_items_combined.csv`

The verified Rodd items and the WordNet-derived expansion pool in one table, carrying the
same 24 lexical-variable columns so the two can be compared directly.

**2,035 rows, 1,942 unique words, 35 columns.** A Rodd word appears once per appendix it was used in,
which is why the row count exceeds the word count.

| `provenance` | Rows | Words | What it is |
|---|---|---|---|
| `published` | 402 | 309 | Rodd, Gaskell & Marslen-Wilson (2002), verified against Appendices A–C |
| `wordnet-derived` | 1,633 | 1,633 | Derived here from WordNet 3.0. **Never used in any published experiment** |

### Provenance and design — columns 1–9

| Column | Content |
|---|---|
| `word` | the item. Lowercase alphabetic, 3–9 letters |
| `provenance` | `published` or `wordnet-derived` — **check this before reporting anything** |
| `source_set` | `Rodd2002-A` / `-B` / `-C`, or `WordNet 3.0` |
| `pool_condition` | the derived items' original label: `homonym-candidate` (1,399) or `polyseme-candidate` (234). Empty for published items |
| `appendix` | A, B or C. **Empty for derived rows** |
| `experiment` | 1, 2 or 3. **Empty for derived rows** |
| `modality` | `visual` (Exp. 1–2) or `auditory` (Exp. 3). **Empty for derived rows** |
| `ambiguity` | `ambiguous` / `unambiguous` — the **number of meanings**, defined by Wordsmyth dictionary entries. **Empty for derived rows** |
| `senses` | `few` / `many` for Appendices B and C. Empty for Appendix A, which had no sense manipulation, and for every derived row |

Derived words have no design columns because no one ever assigned them to a condition.

### Form and frequency — columns 10–17

| Column | Source | Range | Coverage |
|---|---|---|---|
| `length` | letters | 3–9 | 1,942 / 1,942 |
| `n_syll` | MRC | 1–5 | 1,752 |
| `n_phon` | MRC | 1–10 | 1,643 |
| `ortho_n` | computed | 0–20 | 1,942 |
| `subtlex` | Brysbaert file | 17–267,620 | 1,940 |
| `log_subtlex` | computed | 1.25–5.43 | 1,940 |

`subtlex` is a raw occurrence count, not a per-million rate — use `log_subtlex`, which is
`log10(subtlex + 1)`, in any regression.

`ortho_n` is Coltheart's N computed here over the 37,056 alphabetic words of the Brysbaert
list. It is **not** the CELEX-based N the article used, so it will not reproduce its Table 7.

### Ambiguity variables — the point of the file — columns 18–29

| Column | Range | Meaning |
|---|---|---|
| `wn_n_all` | 1–60 | total WordNet 3.0 synsets, all parts of speech. **The closest match to the article's "WordNet senses"** |
| `wn_n_noun` | 0–33 | noun synsets. Zero means the word is a noun in the Brysbaert norms but has no noun entry in WordNet — worth filtering out |
| `wn_n_verb` | 0–59 | verb synsets |
| `wn_n_adj` | 0–25 | adjective synsets, including satellites |
| `wn_n_adv` | 0–4 | adverb synsets |
| `wn_n_pos` | 1–4 | how many parts of speech the word has any synset in — a word-class ambiguity index |
| `wn_domains_all` | 1–19 | distinct WordNet lexicographer files spanned, all POS |
| `wn_domains_noun` | 0–15 | the same for noun senses only (`noun.artifact`, `noun.person`, …). More domains = senses scattered further apart = more homonym-like |
| `sim_top2` | 0.08–0.91 | Wu–Palmer similarity of the two leading noun senses. **Lower = closer to homonymy.** Empty for the 63 words with fewer than two noun senses |
| `min_sim` | 0.07–0.90 | lowest pairwise Wu–Palmer similarity among the top four noun senses — the furthest-apart pair. **Higher = all senses mutually close**, which indicates polysemy |
| `mean_sim` | 0.08–0.90 | mean pairwise Wu–Palmer similarity among the same four |
| `ambiguity_class` | — | derived label, see below |

`sim_top2` asks whether the two main senses are far apart; `min_sim` asks whether *all* senses
are close. They are complementary, which is why both are here.

`ambiguity_class` takes four values, applied uniformly to published and derived rows alike:

| Value | Criterion |
|---|---|
| `homonym-candidate` | `sim_top2` ≤ 0.25 |
| `polyseme-candidate` | `wn_n_noun` ≥ 3 and `min_sim` ≥ 0.50 |
| `intermediate` | two or more noun senses, neither criterion met |
| `single-noun-sense` | fewer than two noun senses, so no pair can be formed |

For the 1,633 derived rows it reproduces `pool_condition` exactly, with zero mismatches —
expected, since both apply the same criteria.

#### Which of these index senses, and which index meanings

Senses and meanings are separate factors in the Rodd design, and the columns split cleanly
between them. Measured on the 220 rows of Appendices B and C, where the two factors are
orthogonal by construction:

| Column | r with **senses** (few/many) | r with **meanings** (unambiguous/ambiguous) |
|---|---|---|
| `gcide_etymologies` | −.06 | **.55** |
| `gcide_entries` | .20 | .48 |
| `wn_domains_all` | .70 | .03 |
| `wn_n_all` | .67 | .05 |
| `wn_n_verb` | .65 | .02 |
| `wn_n_noun` | .52 | .02 |
| `wn_domains_noun` | .52 | −.02 |
| `wn_n_pos` | .20 | .05 |
| `wn_n_adj` | −.19 | .09 |
| `sim_top2` | −.01 | .00 |
| `mean_sim` | −.10 | .06 |
| `min_sim` | −.26 | .07 |
| `log_subtlex` | .18 | −.05 |
| `length` | −.01 | −.07 |
| `concreteness` | .06 | −.05 |

**Use `wn_n_all` as the senses variable.** It is the column that corresponds to the article's
own WordNet sense count, it is continuous, and it drops straight into a regression.
`wn_domains_all` correlates slightly higher but measures how far senses are *spread* across
semantic fields, not how many there are — a different construct.

`wn_n_adj` runs in the opposite direction and should not be used. `wn_n_pos` is word-class
ambiguity, not sense count.

**`sim_top2`, `min_sim` and `mean_sim` do not measure sense count at all.** They measure the
*distance* between senses — the homonymy/polysemy axis. `sim_top2` correlates .00 with the
senses factor. Treat them as a separate dimension, never as a substitute for a count.

**No WordNet column recovers the meanings factor.** Every WordNet row above sits within
noise of zero on the right. The one computed column that does recover it is
`gcide_etymologies`, and it is the mirror image of `wn_n_all` — .55 on meanings, −.06 on
senses, against .05 and .67. Use `wn_n_all` for senses and `gcide_etymologies` for meanings;
the next two sections explain where the second one comes from and why nothing in WordNet
could do the job.

#### Why WordNet cannot supply a meanings variable

Nine further WordNet measures were built and tested against the meanings factor on the same
220 rows, specifically to try to produce one. None works:

| Candidate | r with meanings | r with senses |
|---|---|---|
| sense clusters, Wu–Palmer ≥ .65 | −.02 | .53 |
| sense clusters, Wu–Palmer ≥ .50 | −.04 | .48 |
| sense clusters, ≥ .50, nouns + verbs | .02 | .72 |
| lexname entropy | .04 | .69 |
| distinct hypernym roots, nouns | −.06 | .23 |
| depth of the lowest common ancestor | −.02 | .00 |
| has both noun and verb senses | −.10 | .41 |
| derivationally unlinked sense pairs | .33 | −.35 |
| derivational components | .25 | −.27 |

Every structural measure WordNet supports turns out to be a *senses* measure wearing a
different name — it loads on the factor that was not being targeted.

The last two look like exceptions and are not. Their sign **reverses** on Appendix A
(r = −.14 and .05), and the cell means show the effect comes from a single cell —
ambiguous / few senses sits at .74 while the other three sit between .24 and .33. The
fraction saturates whenever a word has few lemmas, so it is tracking sparseness, not
homonymy. Shipping it as a meanings column would be shipping an artifact.

The reason is structural: WordNet has no notion of a dictionary headword and records no
etymology, so the one distinction the meanings factor rests on — *are these senses
historically and lexicographically the same word?* — is simply not in the database. A
resource that does record etymology answers it immediately, which is the next section.

#### A continuous meanings variable: `gcide_entries` and `gcide_etymologies`

These two columns are computed from **GCIDE**, the GNU Collaborative International Dictionary
of English — the machine-readable Webster's Revised Unabridged (1913). Unlike WordNet, GCIDE
is a real dictionary: homographs are separate headword blocks, and each block carries its own
`<ety>` etymology field.

| Column | Definition | Range | Coverage |
|---|---|---|---|
| `gcide_entries` | separate headword blocks for the orthographic form | 1–14 | 1,771 / 1,942 words |
| `gcide_etymologies` | **distinct etymologies across those blocks** | 0–7 | same |

`gcide_etymologies` is the meanings variable. It operationalises the criterion directly:
two senses count as different meanings when the dictionary traces them to different origins.

Worked examples — *bank* 3 (ridge / bench / financial), *bass* 3 (fish / bast / music),
*pen* 3, *bark* 2 (dog / boat), *pupil* 2 (student / eye), *fan* 2, against *book* 1,
*lamb* 1, *cage* 1 and *mask* 1. The last two are Rodd et al.'s own unambiguous exemplars.

##### Validation

Measured on the 220 rows of Appendices B and C, where senses and meanings are orthogonal by
design, the measure separates cleanly — and does so in the opposite pattern to `wn_n_all`:

| | r with meanings | r with senses | r with `wn_n_all` | partial on meanings, controlling `wn_n_all` | controlling `log_subtlex` |
|---|---|---|---|---|---|
| `gcide_etymologies` | **+.554** | −.061 | +.093 | **+.553** | **+.558** |
| `gcide_entries` | +.481 | +.202 | +.244 | +.484 | +.489 |
| `wn_n_all` | +.047 | +.666 | — | — | — |

The partials are the point: controlling for sense count or for frequency moves the
correlation by less than .005, so this is not a sense-count or frequency effect in disguise.
Coverage of those 220 rows is complete.

It also replicates across all three independently labelled sets in
`meanings_stimulus_pool.csv`, each of which used a different criterion:

| Labelled set | r | mean, multiple meanings | mean, one meaning |
|---|---|---|---|
| Rodd et al. (Wordsmyth entries) | +.41 | 2.26 | 1.38 |
| Trott & Bergen (MW + OED entries) | +.58 | 2.89 | 1.41 |
| DeLong et al. (norming category) | +.53 | 2.42 | 1.15 |

And the DeLong four levels order monotonically on it: homonym 2.42 > irregular polyseme 1.42
> regular polyseme 1.08 > unambiguous 0.89.

##### Caveats

- **Prefer `gcide_etymologies` over `gcide_entries`.** The entry count picks up sense
  subdivision as well as homonymy (r = +.20 with the senses factor, +.24 with `wn_n_all`),
  which is exactly the contamination the etymology count avoids.
- **A value of 0** means the word is in GCIDE but no entry carries an `<ety>` field — 251
  rows. It is not the same as a blank, which means the word is absent from GCIDE. Either
  drop zeros or recode them to 1, and say which.
- **GCIDE is Webster's 1913.** 171 of the 1,942 words are missing, mostly modern vocabulary,
  and the etymological scholarship is a century old. For a word whose status matters to the
  design, check it against the OED rather than trusting the count.
- This is a **derived measure, not a published norm.** It was validated here against three
  labelled sets, but it has no citation of its own. Report it as computed from GCIDE and give
  the validation, or use it only for screening and take the reportable condition labels from
  `meanings_stimulus_pool.csv`.

#### Where a meanings variable comes from instead: `meanings_stimulus_pool.csv`

Since no column can be computed, the meanings factor has to be imported from studies that
assigned it by hand. Three such sets were retrieved and are shipped as a separate file,
`meanings_stimulus_pool.csv` — **831 rows, 631 unique words, 36 columns.** A word appears once per study
(and once per Rodd appendix) that used it.

| Study | Rows | How meanings were defined | Multiple / one meaning |
|---|---|---|---|
| Rodd, Gaskell & Marslen-Wilson (2002) | 402 | separate **Wordsmyth** dictionary entries | 234 / 168 |
| Trott & Bergen (2021), RAW-C stimulus set | 115 | separate entries in **Merriam-Webster *and* the OED** | 38 / 74 (+3 unclear) |
| DeLong, Trott & Kutas (2022), *Behavior Research Methods* | 314 | four-level norming category H / IP / RP / UA | 78 / 236 |

The Trott & Bergen items carry their own provenance in `original_source`: 56 from
Klepousniotou, Titone & Romero (2008), 25 from Brown (2008), 21 from Klepousniotou & Baum
(2007), 5 from Klepousniotou (2002), 8 new. The DeLong et al. set is the only one with a
**matched unambiguous baseline** built in — 78 homonyms against 78 unambiguous items, with
80 irregular and 78 regular polysemes in between.

##### Columns

| Column | Content |
|---|---|
| `study` | which of the three sets the row comes from |
| `original_source` | the study the item was originally drawn from, or the Rodd appendix |
| `meanings_criterion` | the operational definition that study used — these are **not** the same rule |
| `meanings_condition` | normalised to `multiple-meanings` / `one-meaning` / `unclear` so the three sets can be pooled |
| `raw_label` | the study's own label, verbatim (`ambiguous`, `Homonymy`, `homonym`, `regular polyseme`, …) |
| `entries_mw`, `entries_oed`, `diff_etymology_oed` | per-dictionary judgments, Trott & Bergen rows only |
| `senses_label` | the sense manipulation where one exists — `few`/`many` for Rodd B and C, overlap level for Trott & Bergen |

All 24 lexical-variable columns are the same as in `ambiguity_items_combined.csv` and were
recomputed over all 631 words. Coverage: `wn_n_all` 831/831, `log_subtlex` and `concreteness`
792/831, `imageability` 623/831.

##### The three criteria do not agree

The sets overlap, so they can be checked against each other. They disagree substantially:

| Comparison | Shared words | Agreement |
|---|---|---|
| Rodd vs Trott & Bergen | 29 | 72% |
| Rodd vs DeLong et al. | 48 | 71% |
| Trott & Bergen vs DeLong et al. | 42 | **95%** |

Every disagreement runs the same way: Rodd classes the word as having multiple meanings and
the two modern sets class it as a polyseme — *charm*, *clog*, *degree*, *panel*, *term*,
*nail*, *ruler*, *watch*, *straw*, *marble* among them. Rodd's ambiguous set was built from
the Twilley et al. (1994) homograph norms, where meanings were generated by participants,
then checked against Wordsmyth; the later sets require separate headword entries in two
dictionaries, which is stricter.

> **The meanings factor is criterion-dependent, and the choice changes the item set.** Decide
> which definition the hypothesis needs before pooling. Pooling all three gives the largest
> set but mixes a participant-generation criterion with a dictionary-entry criterion; using
> `study == "DeLong, Trott & Kutas (2022), Behavior Research Methods"` alone gives the
> cleanest design, since it is the only set with a matched unambiguous baseline.
>
> The 14 words that appear in all three sets are the safest items of all — only use them as a
> contrast where all three agree.

##### Why the meanings factor is categorical and not a count

No *study* in this file published a meanings count, and that is a property of the sources,
not a choice made here. All three published group membership only. The continuous
`gcide_etymologies` column described above is a measure computed here, not something any of
these studies reported.

**Rodd et al. (2002).** Appendix A prints two columns, Ambiguous and Unambiguous; Appendices
B and C print four cells. Per-word Wordsmyth entry counts were never published. The only
place the article reports that number is Tables 7 and 9, and there it is a **cell mean** —
Wordsmyth meanings 2.03 / 2.09 / 1.00 / 1.00 — which says the ambiguous group averaged 2.03
entries, not that any particular word has two.

The design is close to binary in any case. Selection required two or more Wordsmyth entries,
and the article states that *"most of the selected words had only two meanings"*; a word with
a third meaning was admitted only when that meaning's probability in the Twilley et al.
(1994) norms fell below 0.1. The unambiguous cells sit at exactly 1.00. With the ambiguous
items massed at 2 and the unambiguous items fixed at 1, a count column would carry almost no
variance beyond the group label itself.

**The other two sets are categorical by construction.** Trott & Bergen is a two-way
Homonymy / Polysemy split from dictionary entries. DeLong et al. is a four-level norming
category. Neither reports a per-word number of meanings.

Practical consequences:

- Code `meanings_condition` as a factor. There is nothing to put on a continuous axis.
- `raw_label` preserves each study's own wording, so a study-specific contrast can still be
  recovered after pooling.
- The DeLong et al. levels are **ordered**, and the ordering holds in the released data:
  mean BERT layer-12 distance between usages runs H 0.515 > IP 0.497 > RP 0.464 > UA 0.455.
  Treating those four levels as ordinal is the closest this file comes to a graded meanings
  variable, and it is the reason to prefer that set for a parametric design.
- For a continuous predictor, use `gcide_etymologies`, or import **Gilbert & Rodd (2022)** —
  182 words with a per-word, participant-generated count of 2 to 8 meanings. The GCIDE column
  covers far more words; the Gilbert & Rodd count is the one with a citation behind it.

> Do not substitute `wn_n_all` for a meanings count. It is a senses measure, as the
> correlation table above shows, and using it as a stand-in reintroduces exactly the
> confound Rodd et al.'s Experiments 2 and 3 were designed to remove.

##### Still not retrievable

The dominance norms remain out of reach from this environment: Gilbert & Rodd (2022),
*Journal of Cognition*, 5(1), 4 norms 182 spoken ambiguous words with 2–8 meanings each, but
its data sit on OSF, which is blocked here. The same applies to eDom (Armstrong, Tokowicz &
Plaut, 2012) and the Twilley et al. (1994) norms. Any of them joins to `word` directly if you
can supply the file.

### Semantic norms — columns 30–35

| Column | Source | Scale | Coverage |
|---|---|---|---|
| `concreteness` | Brysbaert et al. (2014) | 1–5, here 1.33–5.00 | 1,940 / 1,942 |
| `conc_sd` | Brysbaert et al. (2014) | 0–1.83 | 1,940 |
| `imageability` | MRC | 100–700, here 233–652 | 950 |
| `familiarity` | MRC | 100–700, here 169–645 | 968 |
| `meaningfulness` | MRC, Colorado norms | 100–700, here 236–616 | 608 |
| `aoa` | MRC | 100–700, here 144–650 | 487 — **too sparse to match on** |

`concreteness` and `imageability` are on different scales and are not directly comparable.
They correlate at about r = .84 but come apart on emotion words — *love* is 311 on
concreteness but 569 on imageability. Coverage below half the file means filtering on any MRC
column shrinks the usable set sharply.

### The two sets are not interchangeable

The derived pool was built by searching WordNet for ambiguity structure, not by matching
Rodd's items, and it differs on every control variable:

| | Published (402) | Derived (1,633) |
|---|---|---|
| `length`, median | 4 | 6 |
| `log_subtlex`, median | 2.80 | 2.31 |
| `ortho_n`, median | 4 | 1 |
| `concreteness`, median | 4.37 | 4.10 |
| `sim_top2`, median | 0.27 | 0.18 |

Derived items are **longer, rarer and far more orthographically isolated**. Dropping them
into a design alongside Rodd items without matching would confound ambiguity with length,
frequency and neighbourhood size at once.

Filtering to the published items' central range — length 4–6, `log_subtlex` 2.0–4.0 — leaves
**646 derived words** (548 homonym-candidates, 98 polyseme-candidates). Adding concreteness
3.5–5.0 leaves **489**. That is the realistic size of the usable expansion pool.

```python
import pandas as pd
df = pd.read_csv("ambiguity_items_combined.csv")

rodd = df[df.provenance == "published"]          # reportable as replication
new  = df[(df.provenance == "wordnet-derived") &
          df.length.between(4, 6) &
          df.log_subtlex.between(2.0, 4.0)]      # 646 matchable candidates
```

> **The `ambiguity` column is the meanings factor and it exists only for published rows.**
> No WordNet variable recovers it (see the validation section above), so a derived word
> cannot be assigned to the ambiguous/unambiguous contrast without new norming — dictionary
> entry counts, or meaning-generation data of the Twilley et al. (1994) kind.
>
> If published and derived items are mixed in one experiment, say so in the stimulus
> appendix and report the `provenance` split. Derived items are candidates, not stimuli with
> a history.

---

## Validation of the computed variables

Checked against the published descriptives for the 309 Rodd items,
`wn_n_all` recovers the article's sense manipulation almost exactly, running slightly high
because WordNet 3.0 has more senses than the 1998 release the authors used:

| Cell | `wn_n_all` here | WordNet senses in the article |
|---|---|---|
| Exp. 2 ambiguous / few senses | 5.44 | 4.88 |
| Exp. 2 ambiguous / many senses | 12.78 | 11.84 |
| Exp. 2 unambiguous / few senses | 5.06 | 5.00 |
| Exp. 2 unambiguous / many senses | 12.66 | 11.19 |
| Exp. 3 ambiguous / few senses | 5.74 | 5.00 |
| Exp. 3 ambiguous / many senses | 12.78 | 11.43 |
| Exp. 3 unambiguous / few senses | 4.57 | 4.43 |
| Exp. 3 unambiguous / many senses | 12.13 | 10.17 |

Length, syllables, concreteness and `log_subtlex` are flat across the four cells of both
appendices, confirming the matching the article reports.

> **The meanings factor is not recoverable from WordNet.** In Appendices B and C the
> ambiguous and unambiguous cells are indistinguishable on every WordNet variable —
> `wn_n_all` 5.44 vs 5.06, `sim_top2` .31 vs .29, `wn_domains_noun` 1.84 vs 2.22. This is by
> design: meanings were defined by **Wordsmyth dictionary entries**, a resource WordNet does
> not mirror, and senses were matched across the ambiguity levels on purpose. Use the
> `ambiguity` column for the meanings factor and `wn_n_all` for the senses factor; do not
> try to derive the first from WordNet.
>
> Appendix A is the exception — its ambiguous words score higher on every WordNet measure
> (`wn_n_all` 8.47 vs 3.97), but there meanings and senses are confounded, which is exactly
> why Experiment 2 was run.

> **Dominance is still missing.** Rodd et al. derived it from 38 participants' word
> associates following Twilley et al. (1994); neither that norm set nor the eDom norms
> (Armstrong, Tokowicz & Plaut, 2012) could be retrieved here. `sim_top2` and WordNet sense
> order are approximations of sense structure, not dominance measures.
