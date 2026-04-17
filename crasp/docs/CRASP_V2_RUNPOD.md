# CRASP v2 RunPod Runbook

## Goal
Run one reliable end-to-end CRASP v2 experiment on `meta-llama/Llama-3.1-8B` using a high-memory RunPod GPU such as `H100` or `H200`.

## Recommended Pod
- GPU: `H100 80GB` or `H200`
- Disk: enough for model cache plus checkpoints, ideally `>200GB`
- Python: `3.10+`
- CUDA-compatible PyTorch stack

## Setup
1. Clone the repo into the pod workspace.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure the volume root exists:

```bash
mkdir -p /workspace/crasp-vol/{models,data,checkpoints,results}
```

4. Copy or mount any existing baseline result JSON files into:

```text
/workspace/crasp-vol/results
```

## Config
Default config lives at:

[`crasp_v2/config/default_runpod.yaml`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/config/default_runpod.yaml)

Update the following before the first real run if needed:
- `paths.volume_root`
- `artifacts.baseline_metrics`
- `training.*` hyperparameters
- `pruning.target_sparsities`

## Recommended Execution Order
1. Build calibration sets:

```bash
python -m crasp_v2.pipeline.build_calibration \
  --raw-medqa-dir data/raw/medqa \
  --raw-medhalt-dir data/raw/medhalt \
  --output-dir /workspace/crasp-vol/data/calibration
```

2. Verify safety calibration:

```bash
cat /workspace/crasp-vol/data/calibration/safety_calibration.validation.json
```

3. Run Phase 0 SFT:

```bash
python -m crasp_v2.pipeline.run_sft \
  --config crasp_v2/config/default_runpod.yaml \
  --mixed-sft-path /workspace/crasp-vol/data/calibration/mixed_sft.jsonl \
  --output-dir /workspace/crasp-vol/checkpoints/post_sft
```

4. Run Phase A saliency capture:

```bash
python -m crasp_v2.pipeline.capture_saliency \
  --config crasp_v2/config/default_runpod.yaml \
  --stage /workspace/crasp-vol/checkpoints/post_sft \
  --reasoning-path /workspace/crasp-vol/data/calibration/cot_calibration.jsonl \
  --safety-path /workspace/crasp-vol/data/calibration/safety_calibration.jsonl \
  --output-dir /workspace/crasp-vol/results/crasp_v2/saliency
```

5. Run Phase B pruning:

```bash
python -m crasp_v2.pipeline.run_prune \
  --config crasp_v2/config/default_runpod.yaml \
  --stage /workspace/crasp-vol/checkpoints/post_sft \
  --saliency-report /workspace/crasp-vol/results/crasp_v2/saliency/saliency_report.json \
  --output-dir /workspace/crasp-vol/checkpoints/post_prune
```

6. Run Phase R recovery:

```bash
python -m crasp_v2.pipeline.run_recovery \
  --config crasp_v2/config/default_runpod.yaml \
  --stage /workspace/crasp-vol/checkpoints/post_prune \
  --mixed-sft-path /workspace/crasp-vol/data/calibration/mixed_sft.jsonl \
  --output-dir /workspace/crasp-vol/checkpoints/post_recovery
```

7. Evaluate any stage:

```bash
python -m crasp_v2.pipeline.run_eval \
  --config crasp_v2/config/default_runpod.yaml \
  --stage /workspace/crasp-vol/checkpoints/post_recovery \
  --baseline-metrics results/baselines/raw_meta-llama_Llama-3.1-8B_20260309T204358Z.json \
  --output-dir /workspace/crasp-vol/results/crasp_v2/eval
```

8. Or run the whole pipeline:

```bash
python -m crasp_v2.pipeline.run_full \
  --config crasp_v2/config/default_runpod.yaml \
  --calibration-dir /workspace/crasp-vol/data/calibration \
  --checkpoint-root /workspace/crasp-vol/checkpoints \
  --results-root /workspace/crasp-vol/results/crasp_v2
```

## Resume Strategy
- `post_sft`, `post_prune`, and `post_recovery` each write `phase_artifact.json`.
- Reloading a stage reconstructs:
  - base model
  - parent adapters
  - pruning masks
  - recovery adapter if present
- You can resume from any saved stage directory by passing it to `--stage`.

## Smoke Validation Before Full Run
- Reduce `calibration.cot_samples` and `calibration.safety_samples` to `8`
- Set `eval.num_samples` to `20`
- Set `training.num_train_epochs` to `1`
- Keep `pruning.target_sparsities` to `[0.20]`

Use this first to validate:
- calibration build
- adapter save/load
- saliency report generation
- mask save/load
- evaluation wiring

## Expected Outputs
- calibration JSONL files under `/workspace/crasp-vol/data/calibration`
- phase checkpoints under `/workspace/crasp-vol/checkpoints`
- saliency/eval/comparison JSON under `/workspace/crasp-vol/results/crasp_v2`
- baseline comparison table emitted by `run_full`
