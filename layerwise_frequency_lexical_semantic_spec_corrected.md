# Layerwise Encoding of Word Frequency and Lexical-Semantic Properties in LLMs
## Corrected Experimental Specification for Base and Instruct Models

**Critical scope correction:** The stimulus dataset contains **real words only**. It does not contain nonwords. Therefore, this study must not be framed as WORD-versus-NONWORD classification or as a conventional lexical-decision experiment. The core study examines how information about **word frequency** and related lexical/semantic properties is represented across transformer layers, and how these patterns compare between base and instruct models.

Human reaction time is not part of the core objective.

## 1. Research objective

Measure how word-frequency information and selected lexical/semantic properties are recoverable from the internal representations of language models, layer by layer. Compare the trajectories in base and instruct versions of the selected model families.

The study should answer three main questions:

1. **Frequency-group information:** At each layer, how well can representations distinguish high-frequency from low-frequency words?
2. **Continuous frequency encoding:** How accurately can the frequency value of a word be predicted from each layer's representation?
3. **Related lexical/semantic information:** How does the layerwise recoverability of the supplied WordNet and dictionary-derived variables change across the network?

The first question uses a high-/low-frequency grouping for an interpretable classification analysis. The second is the primary frequency analysis because it preserves the continuous frequency information. The third investigates related variables separately rather than treating them as interchangeable measures of meaning.

A successful probe shows that a property is recoverable from a representation. It does not, by itself, prove that the model causally uses that property.

## 2. Dataset and column definitions

Use the supplied word dataset, with one record per word where possible. The known columns are:

| Column | Intended role |
|---|---|
| `words` | The word stimulus presented to the model |
| `subtlex` | Raw SUBTLEX frequency value; retain for data checks and descriptive reporting |
| `log_subtlex` | Log-transformed word frequency; primary continuous frequency target |
| `gcide-etymologies` | Dictionary-derived etymological count; interpret as a proxy based on the dataset's operational definition |
| `w_n-n-all` | WordNet synset/sense count; interpret according to the dataset's operational definition |

Column spellings may differ in the actual file. Resolve them explicitly in a single configuration/mapping step. Preserve original columns and document normalized internal names. Do not silently assume that hyphenated names and underscore variants are identical.

### Data validation

- Confirm that each row contains a valid word string and that the dataset contains words only.
- Report the number of rows, unique words, duplicate entries, missing values, and exclusions.
- Check that frequency and lexical/semantic annotations are aligned to the correct word.
- Do not replace missing lexical/semantic values with zero. Missing means unavailable; zero may have a substantive interpretation.
- Check case normalization and tokenization. Record words that split into multiple subtokens.
- If a word occurs in duplicate rows, keep all copies in the same split or resolve duplicates according to a documented rule.
- Split by word identity so duplicate instances of the same word cannot leak across training, validation, and test sets.
- Do not fabricate nonwords or introduce a WORD/NONWORD label.

`gcide-etymologies` is a dictionary-derived count and should not automatically be called a count of distinct meanings. `w_n-n-all` is a WordNet synset/sense-count variable, not a direct measure of semantic similarity. Describe each variable using its actual construction and acknowledge its limitations.

## 3. Defining high- and low-frequency words

Use `log_subtlex` as the primary frequency measure.

For the high-/low-frequency classification analysis:

1. Inspect the distribution and missingness of `log_subtlex`.
2. Define the split threshold using **training data only**. Unless the project has a justified, pre-specified external threshold, use the training-set median as the default.
3. Label words above the threshold as high frequency and words below the threshold as low frequency. Specify how words exactly at the threshold are handled; use a deterministic rule and report it.
4. Apply the same training-derived threshold to validation and test words.
5. Do not calculate a separate median independently within the test set, because that changes the target definition and uses test-distribution information.
6. Report group counts and the threshold in the experiment metadata.

If a scientifically justified frequency threshold or established high-/low-frequency norm is already defined in the project, use it instead and document its source before evaluating the test set. Do not choose a threshold after inspecting model performance.

The high-/low-frequency grouping is a secondary, easier-to-interpret view of the same frequency variable; it does not replace the continuous-frequency analysis.

## 4. Models and input construction

Evaluate selected **base** and **instruct** versions as separate conditions. Keep the word list, data splits, representation extraction convention, probe family, and evaluation protocol matched wherever possible.

### Prompt templates

Use model-appropriate input construction. For example:

**Base-model condition**
```text
Word: {STIMULUS}
```

**Instruct-model condition**
```text
You are processing a single English word.
Word: {STIMULUS}
```

These are examples, not mandatory universal templates. For instruct models, use the model's official chat template. Keep the content minimal and do not disclose the word's frequency or annotation values. Avoid asking the model to label a word as high or low frequency in the input, since that would expose the target task and change what is being measured.

If the existing experiment has a fixed, already validated prompt, preserve it consistently rather than changing it without reason. Record the exact prompt, chat template, token IDs, model checkpoint, tokenizer, precision, software versions, and layer count.

### Representation extraction

The primary feature should be the hidden state at the **final token of the word stimulus**, extracted separately from each transformer layer. Record the token span because some words may be split into multiple subtokens.

Do not silently mix the word-final-token representation with the final prompt/answer-position representation. If the project's existing implementation uses the final prompt token, define that as a separate extraction condition and report it distinctly. Use the same extraction convention across layers within a model condition.

For each word and layer, save one vector and metadata identifying the word, frequency value, lexical/semantic variables, model, condition, layer, prompt, and extraction position.

## 5. Primary analysis A: high- versus low-frequency decoding

This analysis asks whether a layer's representation contains information that distinguishes relatively frequent from relatively infrequent words.

### Procedure

1. Derive high-/low-frequency labels from `log_subtlex` using the training-only threshold defined in Section 3.
2. For each model condition and transformer layer, train a separate regularized logistic-regression classifier to predict high versus low frequency from that layer's word representations.
3. Fit feature scaling and all preprocessing on the training data only.
4. Tune regularization using training-only cross-validation or a validation set. Never tune using test performance.
5. Use the same word-level split assignments at every layer and, where feasible, across base and instruct conditions.
6. Evaluate once on the held-out test words.

### Metrics

Report for every layer:

- Balanced accuracy (primary classification metric)
- AUROC
- F1 score
- Confusion matrix or class-wise recall
- Ordinary accuracy, with class balance shown

If the classes are imbalanced, emphasize balanced accuracy rather than ordinary accuracy alone. Keep the decision threshold and metric definitions consistent across layers.

### Interpretation

A strong result means high-/low-frequency group membership is linearly recoverable from the layer representation under the chosen grouping rule. It does not prove that the model has an explicit frequency category or that frequency causally determines the representation.

## 6. Primary analysis B: continuous word-frequency prediction

This is the **primary frequency analysis**, because it preserves the continuous information in the supplied frequency measure.

### Procedure

For each layer and model condition:

1. Use word representations as input features.
2. Predict `log_subtlex` using a regularized linear regression probe (for example, ridge regression).
3. Fit any scaling and transformations on training data only.
4. Select hyperparameters using training-only cross-validation or validation data.
5. Evaluate on held-out words using the same split assignments across layers.

Use the supplied `log_subtlex` values as provided unless inspection of the data documentation shows that the transformation differs from what the name implies. Do not apply an extra logarithm automatically.

### Metrics

Report held-out:

- \(R^2\)
- Mean absolute error (MAE)
- Root mean squared error (RMSE), if useful

Include a simple training-mean or equivalent baseline so the meaning of \(R^2\) and error values is clear. A negative test \(R^2\) is possible and should be reported honestly; it means the probe performs worse than predicting the test targets using the relevant baseline, not that the experiment failed.

Report both the full layerwise trajectory and the final-layer result. Do not select the best layer on test results and then present that best-layer score as if the layer had been pre-specified.

## 7. Primary analysis C: related lexical/semantic variables

Run these analyses on the words with valid annotations. Treat each target separately:

- `w_n-n-all`: WordNet synset/sense-count variable.
- `gcide-etymologies`: dictionary-derived etymological-count variable.

For each target and layer, train a regularized linear regression probe and report held-out \(R^2\) and MAE (optionally RMSE). Inspect the target distribution before choosing transformations. Any transformation must be justified, specified in advance, and fitted without test leakage. Keep missing values as missing and report the number of usable words for each target.

Do not merge the two variables into one generic “semantic score” without a validated reason. Do not describe them as direct measures of semantic similarity. Their values may reflect dictionary coverage, annotation procedures, and properties of the underlying lexical resources as well as lexical-semantic structure.

## 8. Base-versus-instruct comparison

The base/instruct comparison is integrated into each analysis. For fair comparison, keep constant where possible:

- word list and word-level split assignments;
- frequency threshold and group-label definition;
- target columns and missing-data rules;
- representation extraction convention;
- probe type and hyperparameter-selection protocol;
- evaluation metrics and uncertainty procedure.

Use the official chat template for each instruct model and document differences in tokenization or input formatting. If the base and instruct checkpoints differ in architecture, tokenizer, or training setup beyond instruction tuning, do not attribute all observed differences solely to instruction tuning.

## 9. Main figure and result tables

Use one compact multi-panel figure to show the progression of information across layers.

**Panel A — Frequency-group decoding:** held-out balanced accuracy (and optionally AUROC) for high- versus low-frequency words by layer. Plot base and instruct variants separately or use clearly distinguishable curves.

**Panel B — Continuous frequency encoding:** held-out \(R^2\) for predicting `log_subtlex` by layer. Include uncertainty bands or error bars where feasible.

**Panel C — Lexical/semantic annotations:** held-out \(R^2\) trajectories for `w_n-n-all` and `gcide-etymologies`, with separate labels and clear target definitions.

Do not place metrics with incompatible scales on the same axis unless there is a clear and justified normalization. Use consistent layer numbering and model labels. Show sample sizes for analyses that have different numbers of usable annotations.

Recommended long-form results table fields:

`model`, `condition`, `layer`, `task`, `target`, `metric`, `score`, `ci_low`, `ci_high`, `n_train`, `n_validation`, `n_test`, `seed`, `prompt_id`, `extraction_position`, `frequency_threshold`.

## 10. Statistics and reproducibility

- Pre-specify the primary metrics and comparisons before examining test results.
- Keep the held-out test set untouched until final evaluation.
- Use paired bootstrap resampling over test words or another justified uncertainty procedure.
- Correct for multiple comparisons when making inferential claims about individual layers.
- Report sample counts, group balance, missingness, exclusions, random seeds, and the high-/low-frequency threshold.
- Save per-word, per-layer features or a documented feature cache, along with probe predictions.
- Store all run settings in a machine-readable configuration.
- Key cached features by model checkpoint, tokenizer, prompt/chat template, word identity, extraction position, and layer.
- Keep data splits identical across layers so the trajectory reflects layer differences rather than changes in the evaluation sample.

## 11. Optional secondary causal analysis

A causal intervention is not required for the core study. If pursued, it should target a clearly defined frequency-related representation and test a model outcome that is actually available in the experimental setup. Define the intervention using training data only, include sham and magnitude-matched controls, and evaluate on held-out words. Do not describe an intervention as changing a WORD/NONWORD decision because the dataset contains no nonwords and that is not the current study objective.

If the project does not have a principled, pre-specified model outcome for this intervention, omit it rather than adding a weak causal claim.

## 12. Explicitly out of scope

Do **not** implement the following in this project unless the research question is explicitly changed:

- WORD-versus-NONWORD classification;
- fabricated nonword stimuli or labels;
- conventional lexical-decision accuracy;
- human reaction-time regression or claims about human RT;
- using inference/generation latency as a substitute for human RT;
- frequency values or semantic annotations included in the model prompt;
- selecting the best layer on the test set;
- combining frequency, WordNet counts, and etymological counts into one unvalidated score;
- unrelated benchmark suites or broad prompting experiments that do not answer the stated questions.

## 13. Expected deliverables

1. Validated word-only dataset and documented column mapping.
2. Exact prompts/chat templates and tokenization records for every model condition.
3. Per-word, per-layer representations or a reproducible feature cache.
4. Layerwise high-/low-frequency classification results.
5. Layerwise continuous `log_subtlex` regression results.
6. Separate layerwise regression results for `w_n-n-all` and `gcide-etymologies`.
7. Main multi-panel figure with uncertainty estimates where feasible.
8. Long-form results CSV, configuration file, and concise reproducibility notes.

**Final framing:** This is a layerwise study of **frequency information and related lexical/semantic properties in word representations**. The high-/low-frequency classifier is one view of frequency encoding; continuous frequency prediction is the primary quantitative analysis. There is no WORD/NONWORD task in the supplied dataset.
