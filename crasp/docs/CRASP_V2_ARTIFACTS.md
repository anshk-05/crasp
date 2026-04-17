# CRASP v2 Artifact Schema

## Calibration Artifacts
- `cot_calibration.jsonl`
- `plain_calibration.jsonl`
- `safety_calibration.jsonl`
- `safety_calibration.validation.json`
- `mixed_sft.jsonl`
- `calibration_manifest.json`

## Phase Directories
Each checkpoint stage directory contains:
- `phase_artifact.json`
- optional `adapter/`
- optional `mask.json`
- optional `metrics.json`

## `phase_artifact.json`
Serialized form of [`PhaseArtifact`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/pipeline/artifacts.py):

```json
{
  "phase": "post_prune",
  "created_at": "2026-04-03T22:00:00+00:00",
  "model_name": "meta-llama/Llama-3.1-8B",
  "base_model": "meta-llama/Llama-3.1-8B",
  "parent": "/workspace/crasp-vol/checkpoints/post_sft",
  "adapter_path": null,
  "mask_path": "/workspace/crasp-vol/checkpoints/post_prune/mask.json",
  "metrics_path": "/workspace/crasp-vol/checkpoints/post_prune/metrics.json",
  "extra": {
    "pruned_heads": 24,
    "total_heads": 256
  }
}
```

## Saliency Report
`saliency_report.json` contains:
- `reasoning`
- `safety`
- `combined`
- `total_heads_per_layer`
- `reasoning_path`
- `safety_path`

Each layer in `combined` stores:
- `reasoning_mean_activation`
- `reasoning_coverage`
- `safety_mean_activation`
- `safety_coverage`
- `reasoning_score`
- `safety_score`
- `combined_score`
- `never_activated`

## Pruning Iteration Report
`pruning_iterations.json` contains:
- list of iteration rows
- target sparsity
- cumulative pruned head count
- evaluation metrics for that iteration

## Metrics
`metrics.json` is baseline-compatible and should include:
- `clinical_accuracy`
- `safety_score`
- `safety_breakdown`
- `retention`
- `model_name`
- `sparsity` when applicable

## Comparison Output
`full_run_summary.json` stores:
- stage paths
- baseline comparison rows
- rendered markdown table
