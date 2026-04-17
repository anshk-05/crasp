# CRASP v2 Explained

## What This Rebuild Is

`CRASP v2` is a clean-slate rebuild of the pruning pipeline intended to replace the older experimental CRASP code without throwing away the useful benchmark and evaluation work that already exists.

The main idea is:

- keep the current baseline and evaluation stack as reference
- build a fresh implementation for the actual CRASP method
- make the new pipeline easier to run on RunPod
- make each phase resumable and inspectable
- explicitly preserve clinical reasoning and safety while pruning

This rebuild targets one reliable first result on `meta-llama/Llama-3.1-8B` before expanding into the full paper-quality experiment matrix.

---

## What Was Added

### 1. A new clean package: `crasp_v2/`

This is the new implementation surface.

Top-level structure:

- [`crasp_v2/data/`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/data)
- [`crasp_v2/saliency/`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/saliency)
- [`crasp_v2/pruning/`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/pruning)
- [`crasp_v2/eval/`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/eval)
- [`crasp_v2/pipeline/`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/pipeline)
- [`crasp_v2/config/`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/config)

This keeps the old `src/` code intact while giving you a clean place to implement the actual CRASP method.

### 2. Fixed safety calibration generation

The old bug was real:

- Med-HALT `options` were stored as stringified dicts
- the old safety calibration generator treated them like plain strings
- that produced broken prompts like `A) {`, `B) '`, `C) 0`

This is now fixed in two places:

- the new parser in [`crasp_v2/data/medhalt.py`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/data/medhalt.py)
- the legacy generator patch in [`scripts/generate_safety_dataset.py`](/Users/ansh/Research/CRASP/crasp/crasp/scripts/generate_safety_dataset.py)

The current safety calibration file was also regenerated:

- [`data/calibration/safety_calibration.jsonl`](/Users/ansh/Research/CRASP/crasp/crasp/data/calibration/safety_calibration.jsonl)
- [`data/calibration/safety_calibration.validation.json`](/Users/ansh/Research/CRASP/crasp/crasp/data/calibration/safety_calibration.validation.json)

Current validation result:

- `total_records = 128`
- `invalid_option_prompts = 0`
- `valid = true`

### 3. Calibration dataset builders

The new calibration layer supports:

- CoT reasoning calibration
- plain MedQA calibration control
- safety calibration from Med-HALT
- mixed SFT training data

Main file:

- [`crasp_v2/data/calibration.py`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/data/calibration.py)

### 4. Head-level saliency collection

CRASP v2 does not use Wanda or SparseGPT pruning internals for pruning decisions.

Instead, it:

- discovers attention modules
- hooks into each attention block’s `o_proj` input
- reshapes activations by head
- measures per-head activation energy and coverage
- keeps reasoning and safety saliency separate before combining them

Main files:

- [`crasp_v2/saliency/modeling.py`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/saliency/modeling.py)
- [`crasp_v2/saliency/stats.py`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/saliency/stats.py)

### 5. Structured head masking and pruning order

The pruning unit is an attention head.

The current implementation is designed around:

- never-activated heads pruned first
- remaining heads ranked by lowest combined saliency
- equal default weighting between reasoning and safety saliency
- head masking applied at runtime through forward pre-hooks

Main files:

- [`crasp_v2/pruning/masks.py`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/pruning/masks.py)
- [`crasp_v2/pruning/runtime.py`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/pruning/runtime.py)

This is intentionally conservative for the first version:

- it gives you a real structured pruning mechanism
- it avoids inheriting wrong abstractions from unstructured pruning code
- it keeps debugging easier on RunPod

### 6. End-to-end phase CLIs

The following CLIs were added:

- `python -m crasp_v2.pipeline.build_calibration`
- `python -m crasp_v2.pipeline.run_sft`
- `python -m crasp_v2.pipeline.capture_saliency`
- `python -m crasp_v2.pipeline.run_prune`
- `python -m crasp_v2.pipeline.run_recovery`
- `python -m crasp_v2.pipeline.run_eval`
- `python -m crasp_v2.pipeline.run_full`

Files:

- [`crasp_v2/pipeline/build_calibration.py`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/pipeline/build_calibration.py)
- [`crasp_v2/pipeline/run_sft.py`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/pipeline/run_sft.py)
- [`crasp_v2/pipeline/capture_saliency.py`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/pipeline/capture_saliency.py)
- [`crasp_v2/pipeline/run_prune.py`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/pipeline/run_prune.py)
- [`crasp_v2/pipeline/run_recovery.py`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/pipeline/run_recovery.py)
- [`crasp_v2/pipeline/run_eval.py`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/pipeline/run_eval.py)
- [`crasp_v2/pipeline/run_full.py`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/pipeline/run_full.py)

### 7. Resumable phase artifacts

Each major stage writes metadata describing:

- what phase it is
- what parent stage it came from
- where the adapter is
- where the pruning mask is
- where the metrics are

Main file:

- [`crasp_v2/pipeline/artifacts.py`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/pipeline/artifacts.py)

This means you can resume from:

- `post_sft`
- `post_prune`
- `post_recovery`

without manually reconstructing model state.

### 8. RunPod config and docs

Added:

- [`crasp_v2/config/default_runpod.yaml`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/config/default_runpod.yaml)
- [`docs/CRASP_V2_RUNPOD.md`](/Users/ansh/Research/CRASP/crasp/crasp/docs/CRASP_V2_RUNPOD.md)
- [`docs/CRASP_V2_ARTIFACTS.md`](/Users/ansh/Research/CRASP/crasp/crasp/docs/CRASP_V2_ARTIFACTS.md)

These describe:

- default first-run settings
- expected filesystem layout
- command order
- output artifacts
- resume strategy

### 9. Tests

A lightweight pure-Python test suite was added under:

- [`tests/`](/Users/ansh/Research/CRASP/crasp/crasp/tests)

Covered areas:

- Med-HALT safety option parsing
- detection of the old character-split bug
- answer extraction
- retention calculation
- saliency aggregation
- pruning ranking and mask generation
- phase artifact serialization

---

## What Was Reused vs Replaced

### Reused conceptually

- evaluator and metrics from `src/`
- baseline result format
- MedQA prompt-generation approach
- hook orchestration ideas from Wanda and SparseGPT loaders

### Not reused directly

- Wanda pruning logic
- SparseGPT pruning logic
- the old safety calibration formatting
- weight-level pruning assumptions

This is important: `CRASP v2` is not “Wanda with medical prompts.” It is a separate head-level structured pruning pipeline.

---

## How the New Pipeline Works

## Phase 0: Mixed SFT

Inputs:

- CoT calibration prompts
- safety prompts

Action:

- apply LoRA fine-tuning on mixed medical reasoning + safety data

Output:

- `post_sft` stage directory
- adapter weights
- evaluation metrics

Purpose:

- shape the model’s internal activations before saliency is measured

## Phase A: Saliency Capture

Inputs:

- `post_sft` model
- CoT calibration set
- safety calibration set

Action:

- run forward passes
- collect per-head activation statistics
- compute:
  - mean activation
  - coverage
  - never-activated flag
- keep reasoning and safety saliency separate
- combine them into a head-level score

Output:

- `saliency_report.json`

Purpose:

- identify which heads matter for medical reasoning
- identify which heads matter for safety / hallucination resistance

## Phase B: Iterative Pruning

Inputs:

- saliency report
- raw baseline metrics

Action:

- rank heads for pruning
- prune never-activated heads first
- prune lowest combined-score heads next
- evaluate after pruning steps
- stop if retention constraints fail too often

Default constraints:

- clinical retention >= `0.80`
- safety retention >= `0.80`
- patience = `2`

Output:

- `mask.json`
- pruning iteration history
- `post_prune` stage directory

## Phase R: Recovery

Inputs:

- pruned stage
- mixed SFT data

Action:

- run LoRA recovery fine-tuning

Output:

- `post_recovery` stage directory
- recovery metrics

Purpose:

- recover performance lost during pruning

## Evaluation

Evaluated checkpoints:

- raw
- post-SFT
- post-prune
- post-recovery

Metrics:

- clinical accuracy on MedQA
- safety score on Med-HALT
- retention always relative to raw pre-SFT baseline

---

## What Is Already Verified

These were verified in the local environment:

- new package compiles
- tests pass
- safety calibration now validates cleanly
- the old safety formatting bug is fixed

Commands run:

```bash
python3 -m unittest discover -s /Users/ansh/Research/CRASP/crasp/crasp/tests
python3 -m compileall /Users/ansh/Research/CRASP/crasp/crasp/crasp_v2 /Users/ansh/Research/CRASP/crasp/crasp/tests /Users/ansh/Research/CRASP/crasp/crasp/scripts/generate_safety_dataset.py
```

---

## What Is Not Yet Verified Locally

This environment does **not** have the actual ML stack installed:

- `torch`
- `transformers`
- `peft`
- `datasets`
- `yaml`

So the following are implemented but not executed here:

- actual Llama model loading
- LoRA training
- head-saliency collection on the real model
- iterative pruning on real GPU
- RunPod end-to-end run

That means the code is ready to take to RunPod, but the real experiment still has to be run there.

---

## Expected First Results

For the first serious RunPod run, the expected outcome is **not** “perfect final paper results immediately.”

The realistic expected result is:

1. calibration builds correctly
2. mixed SFT checkpoint is created
3. saliency report contains non-empty per-head statistics for both reasoning and safety
4. pruning mask removes a non-trivial number of heads
5. post-prune metrics are measurable and comparable to raw baseline
6. recovery improves either clinical retention, safety retention, or both

What would count as a strong first success:

- safety calibration remains valid throughout
- saliency report clearly distinguishes active vs inactive heads
- pruning does not catastrophically collapse MedQA
- pruning does not catastrophically collapse Med-HALT
- recovery produces a measurable lift over immediate post-prune performance

What would count as an especially encouraging result:

- `20%` head pruning with acceptable clinical retention
- safety retention staying above threshold or recovering above threshold
- CRASP v2 competitive with or better than your saved Wanda/SparseGPT baseline summaries

---

## Expected Risks

There are still important research and implementation risks:

### 1. Med-HALT metric scale may still be awkward

Your existing baseline results have very small absolute safety scores. That may still be a real issue in:

- dataset composition
- evaluator semantics
- answer extraction behavior

So if safety numbers still look suspicious after the first RunPod run, that is a real investigation point, not just noise.

### 2. Head masking is the first implementation, not the final optimized one

The current pruning mechanism uses structured head masking in forward execution. That is correct for first results, but it is not yet a compact exported architecture.

### 3. Phase 0 and Phase R are sensitive

If SFT is too aggressive:

- clinical behavior may drift
- safety may degrade before pruning

If recovery is too weak:

- you may under-estimate CRASP’s real performance ceiling

### 4. Saliency signal may need refinement

If activation magnitude alone is not discriminative enough, the next refinement could involve:

- normalization changes
- coverage weighting changes
- different reasoning/safety weight ratios
- additional gradient-aware terms

But the current version is the right starting point for a first clean result.

---

## What You Should Do Next

1. Install the full ML stack on RunPod.
2. Run:

```bash
python -m crasp_v2.pipeline.build_calibration
```

3. Confirm:

- `safety_calibration.validation.json` says `valid: true`

4. Run a smoke pass with reduced sample sizes and fewer epochs.
5. Then run the full pipeline with:

```bash
python -m crasp_v2.pipeline.run_full
```

6. Compare the final CRASP metrics against:

- [`results/wanda/`](/Users/ansh/Research/CRASP/crasp/crasp/results/wanda)
- [`results/sparsegpt/`](/Users/ansh/Research/CRASP/crasp/crasp/results/sparsegpt)
- [`results/baselines/`](/Users/ansh/Research/CRASP/crasp/crasp/results/baselines)

---

## Bottom Line

`CRASP v2` now gives you:

- a clean implementation surface
- a fixed safety calibration path
- a real head-level pruning workflow
- resumable phase artifacts
- RunPod-ready commands and config
- tests covering the most important correctness failures

The rebuild is now in a state where the next meaningful step is not more architecture work here. It is taking this pipeline onto RunPod, running the smoke pass, and getting your first reliable CRASP v2 result.
