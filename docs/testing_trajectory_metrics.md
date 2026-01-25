# Testing Trajectory Metrics

This guide explains how to test the trajectory metrics implementation.

## Quick Start

### Method 1: Using eval.py (Recommended)

The simplest way to test is using the existing evaluation framework:

```bash
cd /workspaces/dllm/open-unlearning

# Test with a diffusion model (e.g., LLaDA)
python src/eval.py \
  --config-name=eval.yaml \
  eval=trajectory_test \
  model=<YOUR_DIFFUSION_MODEL_CONFIG> \
  model.model_args.pretrained_model_name_or_path=<MODEL_PATH> \
  task_name=trajectory_test \
  eval.trajectory_test.metrics.trajectory_metrics.trajectory_config.sampler_kwargs.steps=16 \
  eval.trajectory_test.metrics.trajectory_metrics.trajectory_config.sampler_kwargs.max_new_tokens=32
```

**Note**: Replace `<YOUR_DIFFUSION_MODEL_CONFIG>` and `<MODEL_PATH>` with your actual diffusion model config and path.

### Method 2: Add to Existing Evaluation

Add trajectory metrics to an existing MUSE evaluation:

1. Edit `configs/eval/muse.yaml`:
```yaml
defaults:
  - muse_metrics:
    - forget_knowmem_ROUGE
    - trajectory_metrics  # Add this
    # ... other metrics
```

2. Run evaluation:
```bash
python src/eval.py --config-name=eval.yaml eval=muse ...
```

### Method 3: Using dllm job CLI (K8s)

Deploy as a Kubernetes job:

```bash
cd /workspaces/dllm

uv run dllm job trajectory-test \
  --gpu-type A100-40 \
  -- open-unlearning/src/eval.py \
  --config-name=eval.yaml \
  eval=trajectory_test \
  model=<YOUR_MODEL_CONFIG> \
  model.model_args.pretrained_model_name_or_path=<MODEL_PATH> \
  task_name=trajectory_test
```

## Configuration

The trajectory metrics config is at `configs/eval/muse_metrics/trajectory_metrics.yaml`.

Key settings:
- `metrics`: List of metrics to compute (e.g., `["probability", "exact_memorization", "rouge"]`)
- `trajectory_config.return_logits`: Must be `true` to enable logits tracking
- `trajectory_config.sampler_kwargs.steps`: Number of diffusion steps (reduce for testing)
- `trajectory_config.sampler_kwargs.max_new_tokens`: Max tokens to generate (reduce for testing)

## Expected Output

Results will be saved to the output directory (default: `saves/eval/trajectory_test/`).

The output JSON will contain:
```json
{
  "trajectory_metrics": {
    "agg_value": {
      "steps": {
        "probability": [0.5, 0.6, 0.7, ...],  # Array of length S (steps)
        ...
      },
      "fixation": {...},
      "ratio": {...}
    },
    "value_by_index": {
      "0": {
        "trajectories": {
          "steps": {
            "step_0": {"probability": 0.5, ...},
            "step_1": {"probability": 0.6, ...},
            ...
          },
          ...
        }
      }
    }
  }
}
```

## Troubleshooting

### "Model does not have a sampler"
- Ensure you're using a diffusion model (LLaDA, Dream, etc.)
- The model should be wrapped with `DiffusionModelAdapter` (auto-detected)

### "No logits_history returned"
- Ensure `return_logits: true` is set in config
- Check that sampler supports `return_logits` parameter

### Memory errors
- Reduce `batch_size` to 1
- Reduce `steps` (e.g., 16 instead of 128)
- Reduce `max_new_tokens` (e.g., 32 instead of 128)

### Slow execution
- Reduce number of steps for testing
- Use smaller batch size
- Test with fewer samples first

## Verification

After running, verify:
1. ✅ Output file created in `saves/eval/<task_name>/`
2. ✅ Results contain `trajectory_metrics` key
3. ✅ `agg_value` has three trajectory types (steps, fixation, ratio)
4. ✅ Each trajectory has metric arrays of length S (number of steps)
5. ✅ `value_by_index` contains per-sample results

## Example: Testing with Small Config

For quick testing, use minimal settings:

```bash
python src/eval.py \
  --config-name=eval.yaml \
  eval=trajectory_test \
  eval.trajectory_test.metrics.trajectory_metrics.batch_size=1 \
  eval.trajectory_test.metrics.trajectory_metrics.metrics=[probability] \
  eval.trajectory_test.metrics.trajectory_metrics.trajectory_config.sampler_kwargs.steps=8 \
  eval.trajectory_test.metrics.trajectory_metrics.trajectory_config.sampler_kwargs.max_new_tokens=16 \
  data.max_samples=2 \
  task_name=quick_test
```

This will:
- Use batch size 1
- Compute only probability metric
- Use 8 diffusion steps
- Generate max 16 tokens
- Test on 2 samples only
