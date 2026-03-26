# MPPI Hyperparameter Optimization

## Overview

Two scripts are provided for finding optimal MPPI hyperparameters:

1. **`optimize_mppi_hyperparams.py`** - Bayesian optimization using Optuna (recommended)
2. **`grid_search_mppi.py`** - Simple grid search with predefined ranges

Both optimize with `n_octo_samples=4` fixed.

## Quick Start

### Option 1: Bayesian Optimization (Recommended)

```bash
# Install Optuna if not already installed
pip install optuna

# Run optimization (100 trials, 20 examples per trial)
python scripts/optimize_mppi_hyperparams.py \
  --n_trials 100 \
  --n_eval_examples 20 \
  --output_dir results/mppi_optimization \
  --seed 42
```

**Advantages:**
- Efficient exploration of search space
- Automatic pruning of poor trials
- Converges faster than grid search
- Best for finding optimal values

### Option 2: Grid Search

```bash
# Run grid search (tests specific combinations)
python scripts/grid_search_mppi.py \
  --n_eval_examples 30 \
  --output_dir results/mppi_grid_search \
  --seed 42
```

**Advantages:**
- No additional dependencies
- More interpretable results
- Tests specific, reasonable ranges
- Good for comparing discrete options

## Search Spaces

### Bayesian Optimization Ranges:
- `mppi_iterations`: 2-20
- `mppi_samples`: 16-512 (log scale)
- `mppi_elites`: 4-64 (log scale)
- `mppi_temperature`: 0.1-2.0
- `mppi_max_std`: 0.5-3.0
- `mppi_min_std`: 0.01-0.5

### Grid Search Ranges:
- `mppi_iterations`: [3, 6, 12, 16]
- `mppi_samples`: [16, 32, 64, 128, 256]
- `mppi_elites`: [4, 8, 16, 32]
- `mppi_temperature`: [0.3, 0.5, 1.0, 1.5]
- `mppi_max_std`: [0.5, 1.0, 2.0]
- `mppi_min_std`: [0.01, 0.05, 0.1]

## Key Parameters

### Optimization Settings
- `--n_trials`: Number of trials (Bayesian) or combinations (Grid)
- `--n_eval_examples`: Validation examples per trial (more = better estimate, slower)
- `--seed`: Random seed for reproducibility

### Model Settings
- `--encoder_type`: "dino" or "vjepa"
- `--wm_checkpoint`: Path to world model checkpoint
- `--octo_ckpt_dir`: Path to Octo checkpoint directory
- `--octo_ckpt_step`: Octo checkpoint step number

### MPPI Settings
- `--mppi_horizon`: Planning horizon (default: 7)
- `--n_octo_samples`: Number of Octo samples for initialization (default: 4)

## Output

Both scripts produce:
1. **JSON results file** with all trials and best parameters
2. **Best command script** ready to run with optimal parameters
3. **SQLite database** (Optuna only) for visualization

### Analyzing Results

**Optuna visualization:**
```python
import optuna
study = optuna.load_study(
    study_name="mppi_optimization",
    storage="sqlite:///results/mppi_optimization/mppi_optimization.db"
)

# Plot optimization history
optuna.visualization.plot_optimization_history(study).show()

# Plot parameter importances
optuna.visualization.plot_param_importances(study).show()

# Plot parallel coordinate plot
optuna.visualization.plot_parallel_coordinate(study).show()
```

**JSON results:**
```python
import json
with open("results/mppi_optimization/mppi_optimization_results.json") as f:
    results = json.load(f)
    
best = results["best_trial"]
print(f"Best params: {best['params']}")
print(f"ATE XY Final: {best['user_attrs']['ate_xy_final_mean']:.4f}")
```

## Tips for Better Optimization

1. **Start with more examples per trial** (30-50) for reliable estimates
2. **Use early stopping** - Optuna's pruner stops poor trials early
3. **Run multiple seeds** to ensure robustness
4. **Balance compute vs quality:**
   - Quick test: 20 trials × 10 examples
   - Standard: 100 trials × 20 examples
   - Thorough: 200 trials × 50 examples

## Example Workflow

```bash
# 1. Quick exploration (2-3 hours)
python scripts/optimize_mppi_hyperparams.py \
  --n_trials 50 \
  --n_eval_examples 15 \
  --output_dir results/mppi_quick

# 2. Refine best candidates (4-6 hours)
python scripts/optimize_mppi_hyperparams.py \
  --n_trials 100 \
  --n_eval_examples 30 \
  --output_dir results/mppi_refined

# 3. Final evaluation with best params
python scripts/eval_octo_wm_plan.py \
  --num_examples 100 \
  --mppi_iterations <best_value> \
  --mppi_samples <best_value> \
  ... (use values from optimization)
```

## Customization

To modify search ranges, edit the `objective` method:

**Bayesian optimization (`optimize_mppi_hyperparams.py`):**
```python
mppi_iterations = trial.suggest_int("mppi_iterations", 2, 20)  # Change range here
```

**Grid search (`grid_search_mppi.py`):**
```python
search_space = {
    "mppi_iterations": [3, 6, 12],  # Edit values here
    ...
}
```
