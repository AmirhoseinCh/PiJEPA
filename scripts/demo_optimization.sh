#!/bin/bash
# Quick start example for MPPI hyperparameter optimization
# This runs a fast test with minimal examples to demonstrate the tools

set -e

echo "=============================================="
echo "MPPI Hyperparameter Optimization - Quick Demo"
echo "=============================================="
echo ""

# Check if Optuna is installed
if python -c "import optuna" 2>/dev/null; then
    HAS_OPTUNA=true
    echo "✓ Optuna is installed (Bayesian optimization available)"
else
    HAS_OPTUNA=false
    echo "✗ Optuna not installed (only grid search available)"
    echo "  Install with: pip install optuna"
fi

echo ""
echo "This demo will run a quick optimization with minimal settings:"
echo "  - 10 trials/combinations"
echo "  - 10 evaluation examples per trial"
echo "  - Estimated time: ~5-10 minutes"
echo ""

# Create output directory
OUTPUT_DIR="results/mppi_optimization_demo"
mkdir -p "$OUTPUT_DIR"

# Option 1: Bayesian optimization (if available)
if [ "$HAS_OPTUNA" = true ]; then
    echo "Running Bayesian optimization (option 1)..."
    python scripts/optimize_mppi_hyperparams.py \
        --n_trials 50 \
        --n_eval_examples 10 \
        --output_dir "$OUTPUT_DIR/bayesian" \
        --study_name "mppi_demo" \
        --seed 42
    
    echo ""
    echo "✓ Bayesian optimization complete!"
    echo "  Results: $OUTPUT_DIR/bayesian/mppi_demo_results.json"
    echo "  Command: $OUTPUT_DIR/bayesian/best_command.sh"
    echo ""
fi

# # Option 2: Grid search (simplified for demo)
# echo "Running grid search (option 2) with reduced search space..."

# # Create a temporary script with smaller search space for demo
# cat > /tmp/demo_grid_search.py << 'EOF'
# import sys
# sys.path.insert(0, 'scripts')

# # Modify search space for quick demo
# import grid_search_mppi
# original_main = grid_search_mppi.main

# def demo_main():
#     # Temporarily patch the search space
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--n_eval_examples", type=int, default=5)
#     parser.add_argument("--encoder_type", type=str, default="dino")
#     parser.add_argument("--wm_checkpoint", type=str,
#                         default="/mnt/weka/zhougrp/octo_wm_cast/dino_cast_wm_20260228_193050/best_model.pt")
#     parser.add_argument("--octo_ckpt_dir", type=str,
#                         default="/mnt/weka/zhougrp/octo_pt_cast/cast_dino_finetune_20260219_202345")
#     parser.add_argument("--octo_ckpt_step", type=int, default=2000)
#     parser.add_argument("--mppi_horizon", type=int, default=7)
#     parser.add_argument("--n_octo_samples", type=int, default=4)
#     parser.add_argument("--output_dir", type=str, default="results/mppi_optimization_demo/grid")
#     parser.add_argument("--seed", type=int, default=42)
#     args = parser.parse_args()
    
#     # Smaller search space for demo
#     grid_search_mppi.search_space = {
#         "mppi_iterations": [6, 12],
#         "mppi_samples": [32, 64],
#         "mppi_elites": [8, 16],
#         "mppi_temperature": [0.5, 1.0],
#         "mppi_max_std": [1.0, 2.0],
#         "mppi_min_std": [0.05],
#     }
    
#     original_main()

# if __name__ == "__main__":
#     demo_main()
# EOF

# python /tmp/demo_grid_search.py \
#     --n_eval_examples 5 \
#     --output_dir "$OUTPUT_DIR/grid" \
#     --seed 42

# rm /tmp/demo_grid_search.py

# echo ""
# echo "✓ Grid search complete!"
# echo "  Results: $OUTPUT_DIR/grid/grid_search_results.json"
# echo "  Command: $OUTPUT_DIR/grid/best_command.sh"
# echo ""

# echo "=============================================="
# echo "Demo Complete!"
# echo "=============================================="
# echo ""
# echo "Next steps for full optimization:"
# echo ""
# echo "1. Bayesian optimization (recommended):"
# echo "   python scripts/optimize_mppi_hyperparams.py \\"
# echo "     --n_trials 100 \\"
# echo "     --n_eval_examples 20 \\"
# echo "     --output_dir results/mppi_optimization_full"
# echo ""
# echo "2. Grid search (comprehensive):"
# echo "   python scripts/grid_search_mppi.py \\"
# echo "     --n_eval_examples 30 \\"
# echo "     --output_dir results/mppi_grid_search_full"
# echo ""
# echo "See scripts/README_optimization.md for details."
