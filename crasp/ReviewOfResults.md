# Review Of `results/`

The checked-in artifacts broadly match the implemented CRASP v2 pipeline structure, especially for calibration, post-SFT, saliency capture, recovery, and legacy comparison reporting. The biggest exceptions are all around pruning: the current [`run_prune.py`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/pipeline/run_prune.py) control flow does not advance after the first failed candidate batch, the saved `post_prune` checkpoint is effectively unpruned, the saved `post_prune/metrics.json` is a baseline fallback rather than accepted pruned-model metrics, and the copied RunPod artifacts retain absolute `/workspace/crasp-vol/...` paths that are broken in the checked-in repo.

## Findings

1. **Pruning loop does not advance after the first failed candidate batch**

   Expected behavior from the docs is an iterative pruning phase that evaluates progressively larger pruning steps until thresholds fail too often; the implementation is also clearly intended to step through ranked heads in batches during [`run_prune.py`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/pipeline/run_prune.py). Observed behavior in the saved artifacts is two identical 51-head iterations and no progression to a larger candidate set in [`pruning_iterations.json`](/Users/ansh/Research/CRASP/crasp/crasp/results/crasp_v2/checkpoints/post_prune/pruning_iterations.json). This is an `implementation bug`: progress is keyed off `len(best_pruned_heads)` in [`run_prune.py#L61`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/pipeline/run_prune.py#L61), so when the first 51-head candidate fails thresholds, `best_pruned_heads` stays empty and the same 51-head slice is re-evaluated until patience is exhausted.

2. **Saved `post_prune` stage is “no accepted pruning,” not a successful pruned checkpoint**

   [`CRASPv2EXPLAINED.md#L261`](/Users/ansh/Research/CRASP/crasp/crasp/CRASPv2EXPLAINED.md#L261) describes Phase B as producing a pruned stage, mask, and pruning history after iterative structured pruning. The saved stage under [`results/crasp_v2/checkpoints/post_prune/`](/Users/ansh/Research/CRASP/crasp/crasp/results/crasp_v2/checkpoints/post_prune) does contain those files, but the accepted result is effectively unpruned: [`mask.json`](/Users/ansh/Research/CRASP/crasp/crasp/results/crasp_v2/checkpoints/post_prune/mask.json) is an all-ones mask and [`phase_artifact.json`](/Users/ansh/Research/CRASP/crasp/crasp/results/crasp_v2/checkpoints/post_prune/phase_artifact.json) records `"pruned_heads": 0`. This is best classified as an `implementation bug` consequence rather than a mere doc mismatch, because the stage shape is correct but the accepted output indicates that no pruning configuration passed the gate.

3. **`post_prune/metrics.json` is a baseline fallback, not metrics for an accepted pruned model**

   The narrative expectation is that post-prune metrics should be “measurable and comparable to raw baseline” for the accepted pruned stage; the code also writes either `best_metrics` or `baseline_metrics` to disk at [`run_prune.py#L99`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/pipeline/run_prune.py#L99). In the checked-in artifacts, [`post_prune/metrics.json`](/Users/ansh/Research/CRASP/crasp/crasp/results/crasp_v2/checkpoints/post_prune/metrics.json) exactly matches [`raw_meta-llama_Llama-3.1-8B_20260403T223229Z.json`](/Users/ansh/Research/CRASP/crasp/crasp/results/baselines/raw_meta-llama_Llama-3.1-8B_20260403T223229Z.json), including the timestamp, and it has `"retention": null`. This is an `implementation bug` outcome: because no candidate passed thresholds, the saved `post_prune` metrics are baseline fallback data rather than retained metrics for an accepted pruned checkpoint.

4. **Recovery metrics are technically wired correctly, but they are recovering from an effectively unpruned parent stage**

   Recovery is supposed to fine-tune from the pruned stage according to [`CRASPv2EXPLAINED.md#L288`](/Users/ansh/Research/CRASP/crasp/crasp/CRASPv2EXPLAINED.md#L288). The implementation does reconstruct stage lineage recursively in [`common.py#L80`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/pipeline/common.py#L80), applying any parent mask at [`common.py#L111`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/pipeline/common.py#L111), and the saved [`post_recovery/phase_artifact.json`](/Users/ansh/Research/CRASP/crasp/crasp/results/crasp_v2/checkpoints/post_recovery/phase_artifact.json) correctly points back to `post_prune`. The issue is that the parent `post_prune` stage accepted no pruning, so this is an `expected behavior` of the current wiring applied to an unsuccessful pruning stage, not evidence that recovery restored performance after actual head removal.

5. **Checked-in artifact paths are not portable after copying from RunPod**

   The saved artifacts were copied from `/workspace/crasp-vol`, and that origin still appears throughout the checked-in metadata. [`full_run_summary.json`](/Users/ansh/Research/CRASP/crasp/crasp/results/crasp_v2/results/crasp_v2/full_run_summary.json) and the stage [`phase_artifact.json`](/Users/ansh/Research/CRASP/crasp/crasp/results/crasp_v2/checkpoints/post_sft/phase_artifact.json), [`phase_artifact.json`](/Users/ansh/Research/CRASP/crasp/crasp/results/crasp_v2/checkpoints/post_prune/phase_artifact.json), and [`phase_artifact.json`](/Users/ansh/Research/CRASP/crasp/crasp/results/crasp_v2/checkpoints/post_recovery/phase_artifact.json) all contain absolute `/workspace/crasp-vol/...` references that do not exist in the checked-in repo copy. This is an `artifact portability issue`: it does not prove the original RunPod execution failed, but it does mean the committed artifact bundle is not self-consistent for local review or replay.

6. **Saved saliency run is structurally valid but does not demonstrate inactive-head separation**

   The narrative says the first serious result should include a non-empty saliency report and ideally should “clearly distinguish active vs inactive heads” in [`CRASPv2EXPLAINED.md#L365`](/Users/ansh/Research/CRASP/crasp/crasp/CRASPv2EXPLAINED.md#L365). The saved [`saliency_report.json`](/Users/ansh/Research/CRASP/crasp/crasp/results/crasp_v2/results/crasp_v2/saliency/saliency_report.json) is non-empty, spans 32 layers with 32 heads each, and contains per-head reasoning, safety, and combined values, so the basic capture path worked. But every combined entry has `never_activated: false`, and the per-source coverage values are uniformly `1.0`, so this is a `doc mismatch` relative to the stronger narrative claim that this run shows clear active-versus-inactive separation.

7. **Saliency schema in the artifacts differs from the simplified doc description**

   [`CRASPv2EXPLAINED.md#L241`](/Users/ansh/Research/CRASP/crasp/crasp/CRASPv2EXPLAINED.md#L241) describes saliency capture in terms of mean activation, coverage, and never-activated status, while [`docs/CRASP_V2_ARTIFACTS.md`](/Users/ansh/Research/CRASP/crasp/crasp/docs/CRASP_V2_ARTIFACTS.md) compresses the combined schema into a simplified summary. The actual implementation in [`stats.py#L75`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/saliency/stats.py#L75) writes separate `reasoning_coverage` and `safety_coverage` fields into combined entries, and the saved [`saliency_report.json`](/Users/ansh/Research/CRASP/crasp/crasp/results/crasp_v2/results/crasp_v2/saliency/saliency_report.json) follows that structure. This is a `doc mismatch`, not an artifact error.

## What Matches

- Calibration outputs match the builders: [`calibration_manifest.json`](/Users/ansh/Research/CRASP/crasp/crasp/results/crasp_v2/data/calibration/calibration_manifest.json) reports 128 CoT records, 128 plain records, 128 safety records, and 256 mixed SFT records, which matches the generation logic in [`build_calibration.py`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/pipeline/build_calibration.py) and [`calibration.py`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/data/calibration.py).
- Safety validation is clean as expected: [`safety_calibration.validation.json`](/Users/ansh/Research/CRASP/crasp/crasp/results/crasp_v2/data/calibration/safety_calibration.validation.json) has `invalid_option_prompts: 0` and `valid: true`.
- `post_sft` and `post_recovery` both contain the expected adapter directory, `metrics.json`, and `phase_artifact.json`, matching the output contracts in [`run_sft.py`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/pipeline/run_sft.py) and [`run_recovery.py`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/pipeline/run_recovery.py).
- The saliency output exists and contains real per-head data for reasoning, safety, and combined views, so the saved run did execute the saliency serialization path expected by [`capture_saliency.py`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/pipeline/capture_saliency.py).
- [`full_run_summary.json`](/Users/ansh/Research/CRASP/crasp/crasp/results/crasp_v2/results/crasp_v2/full_run_summary.json) includes CRASP v2 plus the latest Wanda and SparseGPT summaries, matching the comparison logic in [`reporting.py#L30`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/pipeline/reporting.py#L30). Its omission of LLM-Pruner is implementation-consistent because no `summary_*.json` was available under [`results/llmpruner/`](/Users/ansh/Research/CRASP/crasp/crasp/results/llmpruner).
- Retention being capped at `1.0` for improved stages is implementation-consistent, not a data error: [`helpers.py#L31`](/Users/ansh/Research/CRASP/crasp/crasp/crasp_v2/eval/helpers.py#L31) clamps retention ratios into `[0, 1]`, so the `post_sft` and `post_recovery` metrics saturating at `1.0` are expected whenever they beat the selected raw baseline.

## Per-Phase Comparison

### Calibration

Calibration is the cleanest part of the saved run. The outputs under [`results/crasp_v2/data/calibration/`](/Users/ansh/Research/CRASP/crasp/crasp/results/crasp_v2/data/calibration) match the implementation and the narrative: the manifest counts are correct, the mixed dataset size is exactly CoT plus safety, and safety validation is clean.

### Phase 0 SFT

The saved [`post_sft`](/Users/ansh/Research/CRASP/crasp/crasp/results/crasp_v2/checkpoints/post_sft) stage matches the implementation contract well. It has the expected adapter payload, stage metadata, and evaluated metrics. Its retention values being exactly `1.0` are consistent with the clamped retention helper rather than evidence of a serialization problem.

### Phase A Saliency

The saved saliency artifact is present and structurally consistent with the implementation. It contains reasoning, safety, combined, and `total_heads_per_layer`, and the head counts line up with the discovered model structure. The main mismatch is interpretive: this run proves saliency capture worked, but it does not prove strong separation between active and inactive heads because every head appears active in the saved report.

### Phase B Pruning

This is where the saved run diverges most sharply from the narrative expectation. The implementation bug in the pruning loop prevents progression after the first failed 51-head batch, the accepted `post_prune` output records zero pruned heads, the saved mask keeps every head, and the saved metrics are baseline fallback data rather than accepted pruned-stage metrics. The phase directory exists, but it should not be read as evidence that structured pruning succeeded in this run.

### Phase R Recovery

The recovery stage is wired and serialized correctly as a child of `post_prune`, and its metrics file is valid. The limitation is causal rather than structural: because `post_prune` accepted no pruning, the recovery result is recovery from an effectively unpruned parent stage, not recovery from an actually pruned model.

### Full Summary / Comparison Output

The full summary matches the implementation in shape and comparison method. It stores stage references, comparison rows, and a rendered Markdown table, and it correctly pulls in CRASP v2, Wanda, and SparseGPT. The local-review weakness is portability: the stored stage paths all point back to `/workspace/crasp-vol/...`, so the checked-in summary is not relocatable as-is.

## Bottom Line

These results are suitable as evidence that most of the CRASP v2 pipeline executed end to end on RunPod and produced the expected artifact families for calibration, SFT, saliency, recovery, and legacy comparison. They are not suitable as evidence of a successful first CRASP v2 pruning run, because the saved pruning stage reflects an implementation bug and falls back to an effectively unpruned accepted checkpoint.

Next fixes: repair the pruning loop so candidate batches advance independently of `best_pruned_heads`, rerun Phase B and Phase R, and rewrite copied artifact metadata to repo-valid relative paths if these results are going to be kept under version control.
