# Trajectory-Based Metrics Specification for dLLM Unlearning Evaluation

## Overview

This document specifies the trajectory-based metrics evaluation system for diffusion language models (dLLMs). The system computes metrics at each diffusion step across three trajectory types, enabling analysis of how metrics evolve during the denoising process.

## Definitions

### 1. Logits Tensor R

**Shape:** `[V, L, S]`

- **V**: Vocabulary size
- **L**: Output length (number of generated tokens)
- **S**: Number of diffusion steps

**Definition:**
- `R[v, l, s]` = logit for vocabulary token `v` at position `l` at diffusion step `s`
- Captures the model's logit distribution over the vocabulary at each position and step

**Source:**
- Computed by samplers during generation when `return_logits=True`
- Stored in `SamplerOutput.logits_history` as a list of `[B, L, V]` tensors (one per step)
- Stacked into `[V, L, S]` tensor using `stack_logits_history()`

### 2. Fixation Steps Tensor F

**Shape:** `[L]`

- **F[l]**: The diffusion step at which token at position `l` was fixed (committed)
- `0 ≤ F[l] < S` for all `l ∈ [0, L-1]`

**Computation:**
- Computed directly in the sampler during generation
- When tokens are committed (e.g., `x[transfer_index] = x0[transfer_index]`), the current global step is recorded: `fixation_steps[transfer_index] = global_step`
- Stored in `SamplerOutput.fixation_steps` as `[B, T]` tensor
- Extracted to `[L]` for trajectory computation (for single sample, takes first batch)

**Key Properties:**
- Fixation step is recorded **at the moment** tokens are committed
- No post-processing needed - step is known when token is fixed
- Positions that are never fixed (always mask) default to `S-1` (last step)

### 3. Trajectory Tensors

Three trajectory tensors, each of shape `[V, L, S]`:

#### 3.1 Steps Trajectory

**Definition:**
```
T_steps[v, l, s] = R[v, l, s]
```

**Semantics:**
- Direct copy of the logits tensor R
- Shows raw logits at each step without transformation

#### 3.2 Fixation Trajectory

**Definition:**
```
T_fixation[v, l, s] = R[v, l, max(0, F[l] - s)]
```

**Semantics:**
- For each token position `l` and step `s`, looks back `s` steps from the fixation step `F[l]`
- At `s=0`: Uses logits at fixation step `F[l]`
- At `s>0`: Uses logits from step `F[l] - s` (clamped to step 0)
- Creates a trajectory centered around when each token was fixed

**Example:**
- If token at position 5 was fixed at step 10 (`F[5] = 10`):
  - `s=0`: Uses logits from step 10
  - `s=1`: Uses logits from step 9
  - `s=2`: Uses logits from step 8
  - etc.

#### 3.3 Ratio Trajectory

**Definition:**
```
T_ratio[v, l, s] = R[v, l, floor(F[l] * (s / S))]
```

**Semantics:**
- Interpolates from step 0 to the fixation step `F[l]` proportionally
- At `s=0`: Uses logits from step 0
- At `s=S-1`: Uses logits from step `F[l]` (fixation step)
- Creates a smooth interpolation trajectory

**Example:**
- If token at position 5 was fixed at step 10 (`F[5] = 10`) and `S=20`:
  - `s=0`: Uses logits from step `floor(10 * 0/20) = 0`
  - `s=10`: Uses logits from step `floor(10 * 10/20) = 5`
  - `s=19`: Uses logits from step `floor(10 * 19/20) = 9` (clamped to valid range)

### 4. Metrics Tensor M

**Shape:** `[3, S, M]`

- **First dimension**: Trajectory type (0=steps, 1=fixation, 2=ratio)
- **Second dimension**: Diffusion step `s` (0 to S-1)
- **Third dimension**: Metric index (one per requested metric)

**Computation:**
For each trajectory type `t ∈ {0, 1, 2}` and step `s ∈ [0, S-1]`:

1. Extract logits `[V, L]` at step `s` from trajectory `t`
2. For logit-based metrics:
   - Wrap logits in `LogitModelWrapper`
   - Call metric function (e.g., `evaluate_probability`)
   - Extract metric value from result
3. For text-based metrics:
   - Decode logits to text via argmax
   - Call metric function (e.g., `eval_text_similarity`)
   - Extract metric value from result
4. Store in `M[t, s, m]` where `m` is the metric index

## Output Format

The `trajectory_metrics` function returns a dictionary following the standard metric format:

```python
{
    "agg_value": {
        # Aggregated values across all samples, steps, and trajectories
        "steps": {
            "probability": np.array([...]),  # [S] - mean across samples at each step
            "rouge": np.array([...]),        # [S] - mean across samples at each step
            ...
        },
        "fixation": {...},  # Same structure
        "ratio": {...}      # Same structure
    },
    "value_by_index": {
        # Per-sample values organized by trajectory, step, and metric
        "0": {  # Sample index
            "trajectories": {
                "steps": {
                    "step_0": {
                        "probability": 0.5,
                        "exact_memorization": 0.8,
                        "rouge": 0.75,
                        ...
                    },
                    "step_1": {...},
                    ...
                    "step_S-1": {...}
                },
                "fixation": {
                    "step_0": {...},
                    ...
                },
                "ratio": {
                    "step_0": {...},
                    ...
                }
            }
        },
        "1": {...},  # Next sample
        ...
    }
}
```

## Implementation Details

### Sampler Integration

Samplers (MDLMSampler, BD3LMSampler) are modified to:

1. **Store logits**: When `return_logits=True`, store logits after each forward pass:
   ```python
   logits_history.append(logits.clone())  # [B, L, V]
   ```

2. **Track fixation steps**: When tokens are committed:
   ```python
   fixation_steps[transfer_index] = global_step
   ```

3. **Return in SamplerOutput**:
   ```python
   return SamplerOutput(
       sequences=x,
       histories=histories,
       logits_history=logits_history,  # list of [B, L, V]
       fixation_steps=fixation_steps,   # [B, T]
   )
   ```

### Trajectory Computation

1. **Stack logits**: Convert `logits_history` list to `R [V, L, S]` tensor
2. **Extract F**: Extract fixation steps `F [L]` from `fixation_steps [B, T]`
3. **Compute trajectories**: Use `compute_trajectories(R, F, S)` to get three trajectory tensors

### Metric Computation

For each trajectory type and step:

1. **Extract logits**: `logits = trajectory[:, :, s]` → `[V, L]`
2. **Reshape**: `logits = logits.unsqueeze(0)` → `[1, L, V]` (add batch dimension)
3. **Compute metric**:
   - **Logit-based**: Use `LogitModelWrapper` to provide logits to metric function
   - **Text-based**: Decode logits to text, then call metric function
4. **Extract value**: Get metric value from result dict
5. **Store**: Add to results structure

## Configuration

Example configuration:

```yaml
trajectory_metrics:
  handler: trajectory_metrics
  metrics:
    - probability
    - exact_memorization
    - rouge
    - classifier_prob
  trajectory_config:
    logits_source: sampler  # or "external"
    return_logits: true  # Sampler config
    return_fixation_steps: true  # Sampler config
    sampler_kwargs:  # Additional sampler arguments
      steps: 128
      temperature: 0.0
```

## Usage

The trajectory metrics can be used like any other metric in the evaluation framework:

```python
from evals import Evaluator

evaluator = Evaluator(name="trajectory_eval", eval_cfg=config)
results = evaluator.evaluate(model=model, tokenizer=tokenizer)
```

Results will include trajectory metrics with values at each step for each trajectory type.

## Notes

- **Memory considerations**: Storing logits for all steps can be memory-intensive. Consider:
  - Computing metrics only at selected steps
  - Using smaller batch sizes
  - Optional quantization of logits
  
- **Performance**: Computing metrics at every step for every trajectory can be slow. Consider:
  - Computing metrics only at selected steps
  - Parallelizing metric computation
  - Caching decoded text for text-based metrics

- **Batched generation**: Current implementation handles single-sample trajectories. For batched generation, trajectories are computed per sample or averaged.
