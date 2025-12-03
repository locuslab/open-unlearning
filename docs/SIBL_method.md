# SIBL (Sparse Bilevel Augmented Lagrangian) Unlearning Method

## Overview

SIBL (S-BiAL) is a sparse bilevel optimization-based unlearning method that combines:
- **Bilevel optimization**: Inner loop maintains model utility on retain set, outer loop forgets target data
- **Augmented Lagrangian method**: Enforces retain loss constraints via dual variables
- **Sparsity constraints**: Applies sparse updates to model parameters for efficient unlearning
- **Implicit differentiation**: Uses conjugate gradient to compute accurate gradients through the inner optimization

## Key Features

1. **Sparse Updates**: Only modifies a sparse subset of model parameters (default: 10% of weights)
2. **Constraint-based**: Guarantees retain set performance stays within specified budget (ε)
3. **Bilevel Optimization**: Separates forget and retain objectives into inner/outer loops
4. **Implicit Differentiation**: Accounts for inner optimization when computing outer gradients

## Method Details

### Algorithm Flow

1. **Initialize**: Create sparsity mask using magnitude pruning (or other methods)
2. **Repeat for T outer iterations**:
   - **Inner loop** (K iterations): Optimize model on retain set with sparsity + L1 regularization
   - **Outer step**: Update model to forget target data while respecting retain constraint
     - Compute forget loss (logit margin flattening)
     - Compute retain loss
     - Form Augmented Lagrangian objective: `L = L_forget + λ*L_retain + ρ/2*(L_retain - ε)²`
     - Optionally apply implicit differentiation correction
     - Update model parameters with masked gradients
     - Update dual variable λ
3. **Output**: Unlearned model with sparse modifications

### Loss Functions

- **Forget Loss**: Logit margin flattening
  ```
  L_forget = mean(max(logits) - mean(logits))
  ```

- **Retain Loss**: Standard cross-entropy
  ```
  L_retain = CrossEntropy(model(x_retain), y_retain)
  ```

- **Sparsity Regularization**: L1 penalty on masked weights
  ```
  R(θ) = γ * Σ |θ ⊙ mask|
  ```

## Configuration

### Method Arguments

Located in `configs/trainer/SIBL.yaml`:

```yaml
method_args:
  # Sparsity configuration
  use_sparsity: true          # Enable sparsity constraints
  sparsity: 0.9               # Target sparsity (0.9 = 90% zeros)
  sparsity_method: layerwise_magnitude  # Pruning method

  # S-BiAL optimization parameters
  epsilon: 0.1                # Retain loss budget (constraint threshold)
  T: 20                       # Number of outer iterations
  K: 10                       # Inner iterations per outer iteration
  eta_theta: 1e-4             # Outer loop learning rate
  eta_in: 1e-4                # Inner loop learning rate
  rho: 1.0                    # Penalty parameter for Augmented Lagrangian
  gamma: 1e-4                 # L1 sparsity regularization coefficient

  # Implicit differentiation settings
  use_implicit: true          # Use implicit differentiation (recommended)
  cg_iters: 10                # Conjugate gradient max iterations
  cg_tol: 1e-3                # Conjugate gradient tolerance
```

### Sparsity Methods

Available sparsity methods (set via `sparsity_method`):

1. **`layerwise_magnitude`** (Recommended): Magnitude pruning applied per layer
2. **`magnitude_sampling`**: Fast global magnitude pruning using sampling
3. **`random`**: Random pruning (baseline)
4. **`structured`**: Prune entire neurons/heads (better hardware efficiency)
5. **`gradual`**: Gradually increase sparsity over iterations
6. **`movement`**: Data-aware pruning based on parameter movement

### Key Hyperparameters

| Parameter | Description | Typical Range | Impact |
|-----------|-------------|---------------|--------|
| `epsilon` | Retain loss budget | 0.05 - 0.2 | Lower = stronger retain constraint |
| `sparsity` | Fraction of weights to zero | 0.5 - 0.95 | Higher = sparser updates |
| `T` | Outer iterations | 10 - 50 | More = better convergence |
| `K` | Inner iterations | 5 - 20 | More = better retain performance |
| `eta_theta` | Outer learning rate | 1e-5 - 1e-3 | Controls forget speed |
| `eta_in` | Inner learning rate | 1e-5 - 1e-3 | Controls retain optimization |
| `rho` | AL penalty | 0.1 - 10.0 | Higher = stronger constraint enforcement |
| `gamma` | L1 regularization | 1e-5 - 1e-3 | Controls sparsity strength |

## Usage

### Running SIBL on TOFU

```bash
python src/train.py \
  experiment=unlearn/tofu/sibl \
  task_name=sibl_tofu_forget10 \
  +trainer.method_args.sparsity=0.9 \
  +trainer.method_args.epsilon=0.1
```

### Running SIBL on MUSE

Create a config file `configs/experiment/unlearn/muse/sibl.yaml` similar to the TOFU config, then:

```bash
python src/train.py \
  experiment=unlearn/muse/sibl \
  task_name=sibl_muse_news
```

### Custom Configuration

You can override any parameter via command line:

```bash
python src/train.py \
  experiment=unlearn/tofu/sibl \
  task_name=sibl_custom \
  +trainer.method_args.T=30 \
  +trainer.method_args.K=15 \
  +trainer.method_args.sparsity=0.95 \
  +trainer.method_args.sparsity_method=structured
```

## Implementation Details

### File Structure

```
src/trainer/
├── sparsity.py              # Sparsity mask creation utilities
└── unlearn/
    └── sibl.py              # SIBL trainer implementation

configs/trainer/
└── SIBL.yaml                # Default SIBL configuration

configs/experiment/unlearn/tofu/
└── sibl.yaml                # TOFU experiment config for SIBL
```

### Key Classes

1. **`SparsityManager`** (`src/trainer/sparsity.py`)
   - Static methods for creating sparsity masks
   - Supports multiple pruning strategies
   - Memory-efficient for large models

2. **`SIBL`** (`src/trainer/unlearn/sibl.py`)
   - Inherits from `UnlearnTrainer`
   - Implements bilevel optimization loop
   - Overrides `train()` method with custom algorithm

### Dependencies

- PyTorch
- HuggingFace Transformers
- scipy (optional, for faster conjugate gradient)
- Standard OpenUnlearning dependencies

If scipy is not available, SIBL falls back to a PyTorch-based conjugate gradient implementation.

## Performance Characteristics

### Computational Cost

- **Memory**: Similar to standard fine-tuning (no reference model needed unless `use_implicit=true`)
- **Time**: Approximately `T * (K + 1)` forward-backward passes
  - With default settings (T=20, K=10): ~220 iterations
  - Each outer step includes implicit differentiation (adds ~2x overhead if enabled)

### Typical Results

On TOFU forget10:
- **Forget Quality**: High (comparable to gradient ascent)
- **Retain Quality**: Better than gradient ascent (due to explicit constraint)
- **Model Utility**: Preserved (controlled by epsilon parameter)

## Troubleshooting

### Common Issues

1. **Retain loss too high**
   - Decrease `epsilon` for tighter constraint
   - Increase `K` (more inner iterations)
   - Increase `rho` (stronger penalty)
   - Decrease `eta_theta` (slower outer updates)

2. **Forget quality insufficient**
   - Increase `T` (more outer iterations)
   - Increase `eta_theta` (faster forgetting)
   - Decrease `sparsity` (allow more parameters to change)

3. **Training instability**
   - Decrease learning rates (`eta_theta`, `eta_in`)
   - Adjust `rho` (AL penalty)
   - Try `use_implicit: false` for simpler optimization

4. **Out of memory**
   - Decrease batch size
   - Reduce `K` (fewer inner iterations)
   - Set `use_implicit: false` (disables second-order computation)
   - Use `sparsity_method: magnitude_sampling` for faster mask creation

5. **Slow conjugate gradient**
   - Install scipy: `pip install scipy`
   - Reduce `cg_iters`
   - Increase `cg_tol`

## References

This implementation is based on bilevel optimization and Augmented Lagrangian methods for constrained machine unlearning with sparsity constraints.

## Citation

If you use SIBL in your research, please cite the OpenUnlearning framework:

```bibtex
@article{openunlearning2025,
  title={OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics},
  author={...},
  journal={arXiv preprint arXiv:2506.12618},
  year={2025}
}
```

## Contributing

To extend or modify SIBL:

1. **Add new sparsity methods**: Implement in `SparsityManager` class in `src/trainer/sparsity.py`
2. **Modify optimization**: Edit the `inner_step()` or `outer_step()` methods in `src/trainer/unlearn/sibl.py`
3. **Tune hyperparameters**: Adjust defaults in `configs/trainer/SIBL.yaml`
4. **Create variants**: Subclass `SIBL` and override specific methods

See `docs/components.md#trainer` for general guidance on adding trainers.
