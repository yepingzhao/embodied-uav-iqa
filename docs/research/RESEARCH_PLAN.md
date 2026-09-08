# UAV-Embodied-IQA Research Plan

**Updated:** 2026-09-07  
**Status:** protocol design and data-integrity repair; research claims remain unvalidated.  
**Problem:** predict the task-dependent cognitive utility of degraded, multi-UAV visual
observations without a clean reference at inference time.  
**Method thesis:** a compact spatial/frequency model conditioned on the question and
semantic subtask may predict this utility better than task-agnostic quality estimators,
provided that supervision, view handling, and evaluation splits are valid.

This is the single maintained research plan. It consolidates the previous three paper
notes, UAV literature review, and multi-view baseline report, but does not inherit their
unverified novelty statements or external performance numbers as project evidence.
Implementation status below was checked against the repository and local artifacts on
the date above. Proposed repairs and experiments are not implemented by this document.

## 1. Research scope and literature decisions

The immediate target is **VLM-derived cognitive utility**, not human aesthetic quality,
flight safety, navigation success, or physical execution success. A sample consists of a
set of distorted UAV views, a question, a semantic subtask, and an aggregated score from
specified VLM annotators. The learned estimator must not receive clean references,
reference answers, distortion identities, or annotator outputs at deployment time.

| Prior work | What this project should retain | Boundary on the inference |
| --- | --- | --- |
| AirCopBench [1] | Multi-UAV observations, semantic questions, simulated/real sources, and the 14-subtask taxonomy | Its perception/decision questions are not measured physical flight outcomes; upstream VQA splits need not be content-disjoint IQA splits |
| Embodied-IQA [2] | Distinguish cognition, decision, and execution; retain raw multi-model annotations | Text agreement is a proxy whose validity must be checked, not proof of downstream task success |
| EPD / MA-EIQA [3] | Task-oriented quality and a multi-scale/spatial-attention baseline | PANet/CBAM and task dependence are prior concepts; their reuse alone is not algorithmic novelty |
| CrossScore [4] | Compare against explicit multi-view baselines and document reference access | NVS cross-reference scoring has a different input contract; it is not a drop-in no-reference UAV baseline |
| Viewport/light-field methods from the archived review | Consider pooling alternatives only if masked averaging is insufficient | Spherical viewports and regular angular grids do not match arbitrary UAV camera poses |

The contribution to pursue is a **validated task-oriented UAV evaluation protocol and a
compact, justified predictor**, not an unsupported claim to be the first multi-view IQA
dataset. A fresh literature/citation check is required before submission; this consolidation
did not perform a new external literature search. Omit inherited leaderboard numbers,
competition rankings, and universal claims that all existing IQA methods fail.

### Task taxonomy

Use the executable mapping in [domain/annotations.py](../../src/uav_iqa/domain/annotations.py):

| Dimension | Codes | Semantic subtasks |
| --- | --- | --- |
| Scene Understanding | 1.1–1.3 | scene_description, scene_comparison, observing_posture |
| Object Understanding | 2.1–2.4 | object_recognition, object_counting, object_grounding, object_matching |
| Perception Assessment | 3.1–3.3 | quality_assessment, usability_assessment, causal_assessment |
| Collaborative Decision | 4.1–4.4 | when_to_collaborate, what_to_collaborate, who_to_collaborate, why_to_collaborate |

Do not relabel these as tracking/delivery/inspection/SAR domains. Cross-task generalization
requires a separately designed protocol; filtering all splits to the same task subset is
not leave-one-task-out evaluation.

## 2. Current evidence and implementation boundaries

### Local data snapshot

Counts below come from scanning the `*_VQA_*.json` arrays in each local split. They count
question–distortion records, not independent scenes. Paths are local, gitignored artifacts.

| Artifact | Train records | Test records | Observed condition |
| --- | ---: | ---: | --- |
| `data/processed/` | 457,452 | 36,900 | All 494,352 records have missing `cognitive_score` |
| `data/annotated/` | 457,452 | 36,900 | No missing/non-finite scores; 106,343 scores exceed 1; overall range 0.097123–1.428571 |
| Original image paths in either copy | 2,816 unique | 1,869 unique | All 1,869 test paths also occur in training; this is path overlap, not yet a scene/content-hash audit |

`data/processed/distorted/` exists; `data/annotated/distorted/` does not. The current
dataset resolves image paths against the same root used for labels. Merely changing
`data_root` from processed to annotated therefore does not establish a usable dataset.

`outputs/benchmark/benchmark_results.json` and a partial benchmark file exist. They are
legacy artifacts pending sample-ID, label-version, split, input-view, and failure audits,
not a validated main result table. No trained-model checkpoint was found under the current
`outputs/` tree during this inspection; this does not rule out artifacts stored elsewhere.

### Code facts and required consequences

| Evidence | Current behavior | Required action before formal runs |
| --- | --- | --- |
| [dataset.py](../../src/uav_iqa/data/dataset.py), [samples.py](../../src/uav_iqa/data/samples.py) | Missing scores become zero; failed training image loads become zero tensors; unknown subtasks default to task 0 | Reject/quarantine invalid samples with counts; distinguish missing values from genuine zero scores |
| [datamodule.py](../../src/uav_iqa/data/datamodule.py) | Validation and testing both use `test`; one subtask filter applies to all splits | Add independent validation and immutable split manifests before checkpoint selection |
| [uav_iqa_net.py](../../src/uav_iqa/models/uav_iqa_net.py) | PANet fuses scales within each view; view features are averaged without a validity mask | Implement masked view aggregation and test padding invariance; do not describe PANet as cross-view fusion |
| [dataset.py](../../src/uav_iqa/data/dataset.py) | Collation pads view counts; per-view random flips/color jitter run during training | Ensure view masking; disable label-changing augmentation unless task/label consistency is demonstrated |
| [text.py](../../src/uav_iqa/evaluation/text.py), [heads.py](../../src/uav_iqa/models/heads.py) | CIDEr contributes on an approximately 0–10 scale; regressor output is sigmoid-bounded | Freeze an explicit label-range contract and retain raw labels; do not silently clip scores above 1 |
| [samples.py](../../src/uav_iqa/data/samples.py), [evaluator.py](../../src/uav_iqa/baselines/evaluator.py) | Baseline sample projection selects the first sorted UAV view; some scoring failures become zero | Add matched multi-view baselines and explicit failure accounting |
| [losses.py](../../src/uav_iqa/training/losses.py) | Cross-task regularization rewards differences between task predictions | Check whether it manufactures task separation; include a zero-weight diagnostic before retaining it |
| [uav_iqa_net.py](../../src/uav_iqa/models/uav_iqa_net.py) | Pretrained-weight failure falls back to random initialization | Require verified, identical weight provenance for comparable runs; fail a formal run on unintended fallback |
| [validate_synth_real_correlation.py](../../scripts/validate_synth_real_correlation.py) | `--output-dir` is declared twice | Repair the CLI and validate its annotation pairing before using it as evidence |

The current model also enables a character-CNN question encoder by default, even though
the eight YAML files do not explicitly set `use_text_encoder`. Removing task conditioning
does **not** remove question information. Resolve and save all defaults in each run.

## 3. Claim map and decision rules

Only two positive research claims are planned; both are hypotheses.

| Claim | Minimum convincing evidence | Blocks |
| --- | --- | --- |
| C1 — Task-conditioned visual quality prediction improves cognitive-utility ranking on unseen UAV content | Beat a matched task-agnostic model and the strongest selected NR baseline on content-disjoint data; survive question-only/task-prior and view-access controls; the label must pass independent validity checks | B1–B4 |
| C2 — UAV-specific distortion training provides benefit beyond generic corruption training | Compare full and generic-only training under matched content coverage and optimization budget on the same UAV-distortion test set; quantify trade-offs on generic distortions and available held-out real-source data | B5, supported by B1 |

**Anti-claims to rule out:** apparent gains are caused by repeated scene content, missing
labels, padding artifacts, task/question priors, extra views, extra parameters, longer
training, or learning an annotator's textual idiosyncrasies rather than visual utility.

Pre-register task-macro SRCC as the decisive ranking metric. As a proposed practical
threshold, require a mean improvement of at least **0.02 absolute SRCC**, with a paired
95% cluster-bootstrap interval for the difference above zero, across three training seeds.
Freeze or revise this threshold using pilot data only, before looking at the final test
set. It is a decision criterion, not an expected result. Report all task results and seed
variability even when the aggregate passes; do not treat three seeds as a large sample.

If conditioning does not help, retain the simpler predictor and narrow C1. If UAV-specific
training does not help, drop C2 rather than reselecting favorable tasks or severity bins.
If independent label checks fail, stop utility claims and repair the measurement target.

## 4. Data and evaluation protocol to freeze at M0

1. **Sample contract.** Record source dataset, scene/sequence/capture ID, question ID/text,
   semantic task, original/distorted paths for every view, view validity, distortion type,
   intensity, RNG seed, annotator model/revision, prompt, decoding settings, raw responses,
   raw metric components, aggregation version, and sample ID. The model sees only the
   permitted deployment inputs; metadata remains available for audits and analysis.
2. **Label/image join.** Join annotated labels to image manifests using unique sample IDs
   and verify view identities. Make label and image roots explicit or create a validated
   unified training artifact. Reject duplicate IDs, unknown tasks, missing/non-finite
   labels, unreadable views, and silent fallback scores. Report retained/excluded counts
   by source, task, distortion, and annotator coverage.
3. **Content-disjoint splitting.** Group by source and scene/sequence, with shared-image
   connected components as a minimum leakage constraint. Put all questions, captures from
   a protected sequence, and distortion variants in one split. Start with a proposed
   70/15/15 train/validation/test group split, stratified where feasible. Audit path hashes,
   exact content hashes, and near-duplicate frames; do not force this ratio if the number
   of independent groups is insufficient. Record both group and record counts. Freeze
   manifests/hashes; fit transforms on training data and choose models only on validation.
4. **Label definition.** Distinguish reference–distorted response agreement from
   ground-truth-answer correctness and GT-normalized ratios; do not mix them as one target.
   For the existing fixed-weight agreement score `(BLEU + ROUGE-L + 0.1 * CIDEr) / 2.1`,
   the nominal upper bound is `10/7`, not 1. A candidate repair is to divide this raw score
   by `10/7`, equivalent to dividing the numerator by 3, if component bounds are verified.
   Freeze this decision and metric version after B1; preserve raw scores and regenerate
   aggregates consistently. Do not apply this transform to other target definitions or
   infer a scale from test-set extrema. Test identical, unrelated, empty, and paraphrased
   responses, including short/structured answers where n-gram metrics are fragile.
5. **Annotator coverage.** Use a fixed declared ensemble or a documented missing-model
   policy; do not average an arbitrary available subset without auditing its bias. Keep
   individual scores for disagreement and leave-one-annotator-out analysis. Questions whose
   answers are unchanged despite visible corruption need independent correctness checks.
6. **Matched inputs.** Use the same valid view sets, image transforms, question access,
   score orientation, and test IDs for comparable systems. NR baseline headline results
   must include per-view score pooling. First-view-only results are input-restriction
   diagnostics. FR methods receive clean references and belong in a separate diagnostic
   panel, not the headline deployable comparison or an assumed upper bound.
7. **Metrics and uncertainty.** Report task-macro SRCC first, then pooled SRCC, PLCC, RMSE,
   and Kendall tau, per task/source/distortion/view-count. Predeclare score direction
   (e.g. negate lower-is-better metrics); never take absolute correlations post hoc. Fit
   any calibration on train/validation only. Bootstrap independent scene/sequence groups,
   paired across methods (initially 2,000 resamples), retaining within-group dependence.
   Report undefined constant-target strata and their counts instead of replacing them
   with zero; distinguish seed SD from scene-sampling uncertainty.
8. **Integrity tests.** Require prediction invariance to padded batch companions and view
   permutation within numerical tolerance, finite losses/gradients, deterministic eval,
   and metric tests on known orderings. Overfit a small valid subset with augmentation
   disabled. Audit whether ranking comparisons mix unrelated scenes/questions and whether
   a grouping-aware sampler is needed; distortion type alone is not a matched ranking unit.

## 5. Five core experiment blocks

All core training comparisons use the frozen M0 split, normalized target version, three
seeds (42, 100, 200), identical model-selection rules, and recorded compute. No final run
is launch-ready until its prerequisites pass. Limit the main baseline set to three families:
conventional NR-IQA, a compact task-agnostic learned predictor, and task-conditioned variants.

### B1. Establish whether the supervision measures visual utility — MUST

- **Claim:** measurement prerequisite for C1/C2; this block also tests the necessity of
  the multi-VLM annotation ensemble.
- **Data/task:** a pilot panel of approximately 280 independent capture–question units,
  initially about 20 per semantic task where available, stratified by source and severity.
  Keep panel groups separate from the final test set and cluster repeated variants.
- **Comparisons:** individual annotators versus the fixed ensemble, lexical agreement
  versus answer correctness, and held-out-annotator scores versus the remaining ensemble.
  Include blinded clean/degraded answer comparisons; a consistently wrong answer is not
  evidence of high utility merely because its wording is unchanged.
- **Independent reference:** exact/rule-based checks where questions permit them; otherwise
  two independent human assessors with adjudication, evaluating task answer correctness
  rather than aesthetics. Record eligibility, disagreement, and missing coverage.
- **Metrics/setup:** score coverage/range, inter-annotator agreement, pairwise ranking
  agreement, and correlations with independent task evidence; fixed prompts/decoding.
- **Success gate:** zero silent failures in retained data and positive independent
  association with uncertainty reported; ensemble complexity must improve reliability or
  reduce annotator-specific failures relative to the strongest single-annotator choice
  selected on the pilot. Freeze quantitative reliability gates before the main test.
- **Failure interpretation:** repair metrics/annotators or limit the target to response
  consistency; do not claim task utility. Human annotation requires a separately approved
  collection workflow/budget; none is initiated by this plan.
- **Paper artifact:** label/provenance table and annotator-agreement figure.

### B2. Main content-disjoint utility prediction — MUST

- **Claim:** C1, against comparable input access and a strong baseline.
- **Data/task:** all eligible tasks, all retained distortion families, frozen M0 splits.
- **Systems:** `full_model`; a shared-backbone masked-mean regressor with text/task
  conditioning disabled; strong conventional NR candidates such as TOPIQ-NR, MANIQA,
  and MUSIQ scored per view then averaged. Select one trainable NR baseline on validation
  and fine-tune it on the same target; record candidate search and comparable tuning budget.
  Include mean-label/task-prior and question-only controls, not only image-based methods.
- **Setup:** start from the current MobileNetV4 configuration: 256 px, AdamW learning rate
  0.0012, weight decay 0.0001, 5 warmup epochs, 50 total epochs, and two frozen backbone
  stages. Batch size 256, 16 workers, and mixed precision are existing defaults, not verified
  hardware recommendations; profile and freeze a feasible effective batch size for all
  matched variants. Validate pretrained weights and label-compatible augmentation.
- **Metrics/gate:** C1's predeclared task-macro SRCC comparison, with per-task/source
  results and uncertainty; a visual model must outperform nonvisual priors to support
  the visual-utility interpretation.
- **Failure interpretation:** investigate data/label confounds before architectural growth;
  publish a bounded benchmark finding if a simple NR baseline is already sufficient.
- **Paper artifact:** main comparison table with separate reference-access columns.

### B3. Isolate conditioning, frequency, attention, and text — MUST

- **Claim:** C1 mechanism attribution; frequency gains are not assumed in advance.
- **Data/task:** identical frozen split and seeds as B2; all valid views for every variant.
- **Systems:** full model versus `no_task_conditioning`, `no_frequency_encoder`,
  `no_spatial_attention`, and `use_text_encoder=false` with task conditioning retained.
  Use B2's compact no-text/no-task control to expose combined semantic-prior effects.
  Question-only and shuffled-question probes distinguish image evidence from wording.
- **Setup:** retain the same training schedule and validated losses. Run one-seed
  `lambda_rank=0` and `lambda_cross_task=0` diagnostics before freezing the loss recipe;
  expand to three seeds if either loss materially affects the conclusion. A loss that
  forces unsupported separation must be removed and all dependent runs repeated.
- **Metrics/gate:** paired changes in task-macro SRCC, task-specific effects, parameter
  count, and latency. Keep a component only if its incremental benefit justifies its cost;
  report interactions rather than assuming independent additive contributions.
- **Failure interpretation:** remove unnecessary blocks or narrow the mechanism claim.
- **Paper artifact:** compact ablation table; loss diagnostics can go in the appendix.

### B4. Defend input fairness and model simplicity — MUST

- **Claim:** C1 is not explained solely by extra views or excessive model complexity.
- **Data/task:** the same test set plus prespecified view-count strata and missing-view
  perturbations. Single-view perturbations retain the full-group label only as a robustness
  diagnostic; a genuine single-view utility claim would require corresponding labels.
- **Systems:** first-view IQA versus per-view-score mean pooling; compact masked-feature
  mean pooling versus the full predictor. Use real view masks throughout. Learned attention
  pooling is optional and only follows a demonstrated masked-mean limitation.
- **Setup/metrics:** reuse B2/B3 checkpoints where appropriate; report total/trainable
  parameters, checkpoint size, peak memory, and synchronized latency at batch size 1 for
  1/3/6 valid views, with hardware, precision, preprocessing boundary, warmup, and repeats
  specified. Compare accuracy–cost trade-offs; do not infer onboard speed from desktop GPU
  throughput or repeat unmeasured INT8 size claims.
- **Success gate:** headline improvement survives equal-view access; select the simplest
  model within the preregistered accuracy tolerance of the best model.
- **Failure interpretation:** extra views explain the gain, or a smaller model is enough;
  revise the claimed contribution accordingly.
- **Paper artifact:** input-contract table and accuracy/latency plot.

### B5. UAV distortion specificity and bounded transfer — MUST for C2

- **Claim:** C2, not a blanket simulation-to-real or cross-task generalization claim.
- **Data/task:** identical held-out content, with UAV-specific and generic distortion
  panels; retain source labels to report real-origin images separately. Real-origin
  images with injected corruption are not equivalent to naturally degraded flight data.
- **Systems:** full-distortion versus `generic_only` training, matched for independent
  source content, effective optimization steps, and sampling policy; `uav_only` is optional.
  Evaluate both on the same complete test panels. Current distortion filters also affect
  evaluation, so their YAML files alone do not implement this matched-test protocol.
- **Setup:** three seeds; report the six implemented UAV distortion types individually.
  The synthesis pipeline samples an intensity per group/type; use existing intensity
  bins descriptively. A controlled severity curve needs newly generated, matched levels
  and reannotation, not an assumption that the current data already contain such sweeps.
- **Metrics/gate:** paired task-macro/per-distortion SRCC differences under C2's threshold,
  generic-panel trade-offs, and independent utility evidence from B1. A natural-degradation
  real-source holdout is a gated extension only after provenance and labels are established.
- **Failure interpretation:** generic augmentation suffices, or synthetic gains do not
  transfer; narrow C2 and report the limitation.
- **Paper artifact:** distortion-family table; verified natural-degradation transfer is
  an additional table, never inferred from dataset names alone.

## 6. Run order, tracker, and resource gates

This embedded tracker is the only maintained plan tracker. `TODO` means not validated
under the new protocol, even if a legacy artifact or configuration exists.

| Milestone | Run group / deliverable | Status | Stop/go gate | Planning effort / compute |
| --- | --- | --- | --- | --- |
| M0 — Integrity | DATA: label/image joins, grouped splits, score contract, masks, failure tests | BLOCKED: known code/data defects above | Zero silent failures, disjoint groups, independent validation, versioned target | Provisional 2–5 engineering days; CPU I/O audit plus short GPU smoke tests |
| M1 — Measurement and baselines | LABEL: B1; SANITY: tiny overfit; BASE: B2 baselines | TODO, depends on M0 | Independent supervision check and competitive, input-matched baselines | Provisional 2–4 working days excluding annotator scheduling; VLM pilot budget measured separately |
| M2 — Main result | MAIN: full model and two learned baseline systems, 3 seeds each | TODO, depends on M1 | No checkpoint selection on test; stable, reproducible predictions | 9 training runs; elapsed time determined by measured pilot throughput |
| M3 — Decisions | ABLATE: four component variants; LOSS: zero-weight probes; SIMPLE: B4 | TODO, depends on M2 | Retain only supported components and freeze final method | 12 component runs, 2 short loss probes, reused inference; expand loss comparisons only if necessary |
| M4 — Distortion evidence and reporting | UAV: generic-only matched-budget control; transfer/failure panels | TODO, depends on M3 | C2 passes or is explicitly withdrawn; all claims trace to artifacts | 3 training runs plus inference; natural-degradation evaluation is conditional |

### Budget envelope, not an allocation

- **Core training:** 24 full-length runs = 8 systems × 3 seeds: full model, compact
  no-task/no-text baseline, selected fine-tuned NR baseline, three component-removal YAMLs,
  no-text variant, and generic-only training. Pilot/loss diagnostics and candidate baseline
  search are additional. Reuse valid checkpoints across blocks; do not count them twice.
- **GPU-hours:** no defensible numeric total exists before profiling. If one measured
  full run costs `T` GPU-hours, the equal-cost approximation is `24 * T`; replace it with
  `3 * sum(T_system)` after per-system pilots. Add measured annotation/inference costs and
  a provisional 20% retry margin. No GPUs are booked and no jobs are launched by this plan.
- **Annotation:** reuse valid raw responses first; start B1 with the bounded pilot panel
  rather than regenerating the entire corpus. Human ratings, independent units, number of
  degraded conditions, and token/image budgets must be priced separately before expansion.
- **Storage:** retain immutable raw data/responses, split hashes, score-version metadata,
  resolved configs, predictions keyed by sample ID, and selected checkpoints. Estimate
  expansion before generating severity sweeps; use ID-based joins instead of image copies.
- **Wall time:** M0/M1 estimates are planning assumptions, not commitments. Training elapsed
  time is total measured GPU-hours divided by usable concurrent GPUs, plus queue/I/O time;
  available hardware and external annotation capacity are not assumed.

### Current configuration map

See the [configuration catalog](../../configs/README.md). These are implementation starting
points, not proof that the new protocol is supported end to end.

| Configuration | Role / limitation |
| --- | --- |
| [full_model.yaml](../../configs/experiments/full_model.yaml) | Main starting point; explicitly resolve the default-enabled text encoder |
| [no_task_conditioning.yaml](../../configs/experiments/no_task_conditioning.yaml) | Remove FiLM/task embedding; question encoder remains enabled |
| [no_frequency_encoder.yaml](../../configs/experiments/no_frequency_encoder.yaml) | Frequency deletion study |
| [no_spatial_attention.yaml](../../configs/experiments/no_spatial_attention.yaml) | Spatial-attention deletion study |
| [generic_only.yaml](../../configs/experiments/generic_only.yaml) | Training-domain control; must override the shared evaluation filter in a repaired protocol |
| [uav_only.yaml](../../configs/experiments/uav_only.yaml) | Optional complementary training-domain control |
| [backbone_mobilevit_s.yaml](../../configs/experiments/backbone_mobilevit_s.yaml) | Optional backbone robustness/efficiency check |
| [backbone_efficientvit_b0.yaml](../../configs/experiments/backbone_efficientvit_b0.yaml) | Optional backbone robustness/efficiency check |

The no-text and loss-weight switches exist in the training constructor; the compact
control, question-only baseline, independent splits, masked pooling, strict data validation,
and train-only distortion filtering still require explicit implementation/configuration.
Do not present them as already supported commands. A parse-only check of the current
reference config is safe and does not certify data readiness:

```bash
python scripts/train.py fit --config configs/experiments/full_model.yaml --print_config
```

Current YAMLs explicitly instantiate W&B logging; use the documented optional dependency
or an explicit CSV-only logger configuration. Isolate each seed's checkpoint and logger
paths; changing `trainer.default_root_dir` alone does not change logger paths.

## 7. Risk register and deliberate exclusions

| Risk | Response / consequence |
| --- | --- |
| Model predicts question difficulty or annotator style rather than visual utility | Nonvisual controls, within-question degradation comparisons, independent correctness panel, held-out-annotator analysis |
| High record count hides few independent scenes | Publish group counts and cluster uncertainty; reduce claim scope if groups are insufficient |
| Geometric/color augmentation changes answers or target quality | Disable until label-preservation tests pass; do not flip left/right questions without updating semantics |
| Full model benefits from more views or unequal pretraining | Match view access and weight provenance; separate single-view and FR diagnostics |
| Distortion mixture changes optimization budget | Match source coverage and optimizer steps; report residual distribution differences |
| Task regularizer creates artificial separation | Zero-weight loss probes and independent per-task calibration checks |
| Simulator/real domain is confounded with task or source | Audit source–task contingency tables; avoid transfer claims unsupported by matched coverage |
| Backbone fallback or benchmark failures look like valid scores | Fail formal runs on unintended initialization/fallback and log explicit coverage |

**Main paper:** B1 label validity, B2 main comparison, B3 decisive deletions, B4 fairness and
cost, and B5 only if C2 is supported. **Appendix:** additional backbones, UAV-only control,
FR diagnostics, detailed annotator and failure strata. **Defer:** learned cross-view
attention, exhaustive 15-method retraining, dense severity sweeps, and leave-one-task-out
studies until the compact core result is trustworthy.

**Out of scope for the current claim:** VLA action supervision, reinforcement learning,
real-flight control/retransmission policies, guaranteed safety, universal task transfer,
and proven synthetic-to-natural distortion equivalence. An unseen task embedding is not
a valid zero-shot task-generalization mechanism. Such work requires new protocol and
implementation decisions, not restoration of the removed legacy task configs.

## 8. Completion checklist

- [ ] Labels, images, annotators, and split manifests have traceable versions and hashes.
- [ ] Missing labels/images, invalid tasks, and failed inference cannot become valid zeros.
- [ ] Scene/sequence separation and an independent validation split pass automated checks.
- [ ] Label range matches the regressor; raw scores and metric components are preserved.
- [ ] Padding invariance and augmentation/label consistency pass.
- [ ] Independent evidence supports the cognitive-utility interpretation or scope is narrowed.
- [ ] Strong NR and compact learned baselines share the headline input contract.
- [ ] Three-seed comparisons include paired group uncertainty and all task results.
- [ ] Task, text, frequency, attention, and loss effects are separated adequately.
- [ ] Model complexity and latency claims are measured on specified hardware.
- [ ] Every retained claim points to frozen predictions, metrics, and a reproducible run.
- [ ] Unsupported C2, transfer, first-of-kind, and physical-execution claims are removed.

## 9. Selected references and provenance

1. [AirCopBench: A Benchmark for Multi-drone Collaborative Embodied Perception and Reasoning](https://arxiv.org/abs/2511.11025).
   Task/dataset context; [local paper source](../arXiv-AirCopBench/main.tex).
2. [Image Quality Assessment for Embodied AI](https://arxiv.org/abs/2505.16815).
   Cognitive/decision/execution distinction and multi-model supervision;
   [local paper source](../arXiv-Embodied-IQA/neurips_2025.tex).
3. [Embodied Image Quality Assessment for Robotic Intelligence](https://arxiv.org/abs/2412.18774).
   Task-oriented EPD and MA-EIQA architectural context. No local full paper was checked
   during this consolidation; recheck the source before quantitative citation.
4. [CrossScore: Towards Multi-View Image Evaluation and Scoring](https://arxiv.org/abs/2404.14409).
   Multi-view/reference-contract context; [implementation](https://github.com/ActiveVisionLab/CrossScore).
   Source details and adaptation feasibility need verification before adding it as a baseline.

The prior five Markdown documents were preserved unchanged in a repository-external
recovery archive during consolidation. Their imported Obsidian attachments and speculative
claims are not copied into this plan. Downloaded LaTeX/bibliography/figure directories remain
unchanged. This plan is an evidence-bounded synthesis, not a new literature audit or a report
of completed experiments.
