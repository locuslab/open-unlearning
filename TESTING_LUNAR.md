# Testing Guide for Lunar Unlearning Trainer

This guide explains how to test the updated Lunar unlearning trainer with the new features:

1. **Automatic direction computation** (`compute_direction=True`)
2. **Multi-GPU support**
3. **Single-GPU activation collection**
4. **Tokenizer input fix**

## Quick Test (Recommended for First Run)

### 1. Simple Test with Automatic Direction Computation

Test the basic functionality with a small model and minimal datasets:

```bash
# Run the simple test script
python scripts/test_lunar_simple.py
```

This script:
- Uses `meta-llama/Llama-3.2-1B-Instruct` (small model)
- Creates minimal forget (3 samples) and retain (5 samples) datasets
- Tests automatic direction computation (`compute_direction=True`)
- Tests single-GPU activation collection
- Runs for 2 epochs (quick test)

**Expected Output:**
- Activations extracted successfully
- Direction vectors computed automatically (if `compute_direction=True`)
- Estimated networks trained
- Model weights updated

### 2. Test with Manual Direction (Optional)

To test without automatic direction computation:

Edit `configs/trainer/Lunar.yaml`:
```yaml
method_args:
  compute_direction: False  # Disable automatic direction computation
  direction: []  # Empty - will use zero vectors
```

Or test with provided direction vectors (requires manual creation).

## Feature-Specific Tests

### Test 1: Automatic Direction Computation

This tests the new `compute_direction_from_activations()` function:

```bash
# Ensure compute_direction is enabled in config
python scripts/test_lunar_simple.py
```

**Check for:**
- Console output: "Computing direction vectors from activations..."
- Direction vectors computed for each layer
- No errors during activation computation

### Test 2: Multi-GPU Training

Test distributed training support:

```bash
# Use the multi-GPU test script
bash scripts/test_lunar_multi_gpu.sh
```

**Or manually with accelerate:**
```bash
# Set number of GPUs in configs/accelerate/default_config.yaml
accelerate launch --config_file configs/accelerate/default_config.yaml \
    src/train.py \
    trainer=Lunar \
    trainer.method_args.layer_idx_list=[7] \
    trainer.method_args.compute_direction=true \
    model.name_or_path=meta-llama/Llama-3.2-1B-Instruct \
    data.dataset_name=tofu \
    data.forget_split=forget10 \
    data.retain_split=holdout10
```

**Check for:**
- Training runs on multiple GPUs
- Activation collection runs only on main process (GPU 0)
- Model weights synchronized across processes
- No errors during distributed training

### Test 3: Single-GPU Activation Collection

This is automatically tested in the simple test script. Verify:

- Activations are collected on GPU 0 only
- No CUDA errors or device mismatches
- Flash Attention warning is not triggered (model not moved unnecessarily)

### Test 4: Tokenizer Input Fix

This is automatically tested when using ForgetRetainDataset. Test with different dataset formats:

**Test with string datasets:**
```python
forget_dataset = ["Question 1", "Question 2", "Question 3"]
retain_dataset = ["Answer 1", "Answer 2", "Answer 3", "Answer 4", "Answer 5"]
```

**Test with dict datasets:**
```python
forget_dataset = [
    {"question": "Q1", "answer": "A1"},
    {"instruction": "Q2", "response": "A2"}
]
```

Both should work without tokenizer errors.

## Full Integration Test

### Using TOFU Benchmark

Test with the full TOFU dataset:

```bash
bash scripts/test_lunar.sh
```

This tests:
- Full dataset loading
- Direction computation from real data
- Multi-layer unlearning
- Model saving and loading

## Configuration Options

### Enable Automatic Direction Computation

In `configs/trainer/Lunar.yaml`:
```yaml
method_args:
  compute_direction: True  # Automatically compute direction from activations
  layer_idx_list: [7, 8, 9]  # Modify multiple layers
  coeff_list: [2.0, 2.0, 2.0]  # Perturbation coefficients per layer
```

### Disable Automatic Direction Computation

```yaml
method_args:
  compute_direction: False  # Use provided or zero direction vectors
  direction: []  # Empty - will use zero vectors (forget dataset won't be perturbed)
```

## Expected Behavior

### With `compute_direction=True`:
1. **Activation Extraction**: Extracts activations from forget and retain datasets
2. **Direction Computation**: Computes direction vectors as `mean(forget_activations) - mean(retain_activations)`
3. **Activation Perturbation**: Perturbs forget activations by `coeff * direction`
4. **Estimated Network Training**: Trains LoRA networks to approximate the perturbed activations
5. **Model Update**: Updates model weights based on estimated networks

### With `compute_direction=False`:
1. **Activation Extraction**: Extracts activations from forget and retain datasets
2. **Direction Usage**: Uses provided direction vectors or zero vectors
3. **Activation Perturbation**: Perturbs forget activations by `coeff * direction` (may be zero if no direction provided)
4. **Estimated Network Training**: Trains LoRA networks
5. **Model Update**: Updates model weights

## Troubleshooting

### Error: "Both harmful and harmless activations must be provided"
- **Cause**: Empty datasets or activation extraction failed
- **Fix**: Check dataset loading and ensure datasets are not empty

### Error: "Direction list length doesn't match layer_idx_list"
- **Cause**: Mismatch between direction vectors and layers
- **Fix**: Enable `compute_direction=True` or provide correct number of direction vectors

### CUDA Out of Memory
- **Cause**: Batch size too large or model too big
- **Fix**: Reduce `batch_size` in `method_args` or use gradient checkpointing

### Flash Attention Warning
- **Cause**: Model moved to GPU unnecessarily
- **Fix**: Should be fixed in current code - verify model is only moved when needed

### Multi-GPU Sync Issues
- **Cause**: Model weights not synchronized across processes
- **Fix**: Verify DDP broadcasting is working (should be automatic in current code)

## Performance Benchmarks

For reference, expected timings on a single A100 GPU:

- **Activation Extraction**: ~30-60 seconds for 100 samples
- **Direction Computation**: <1 second
- **Estimated Network Training**: ~5-10 minutes per layer (100 epochs)
- **Model Update**: <1 second

## Next Steps

1. Run `python scripts/test_lunar_simple.py` to verify basic functionality
2. Test multi-GPU with `bash scripts/test_lunar_multi_gpu.sh`
3. Run full TOFU benchmark with `bash scripts/test_lunar.sh`
4. Verify unlearning metrics (MIA, utility, etc.)
