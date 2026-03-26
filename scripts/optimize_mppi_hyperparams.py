#!/usr/bin/env python3
"""
Hyperparameter optimization for MPPI planner with Octo initialization.

Uses Optuna for Bayesian optimization to find optimal:
  - mppi_iterations
  - mppi_samples
  - mppi_elites
  - mppi_temperature
  - mppi_max_std
  - mppi_min_std

Fixed: n_octo_samples = 4

Objective: Minimize ATE XY Final error on validation set.

Usage:
    python scripts/optimize_mppi_hyperparams.py --n_trials 100 --n_eval_examples 20
"""

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# Import the evaluation functions from the main script
sys.path.insert(0, os.path.dirname(__file__))
from eval_octo_wm_plan import (
    build_encoder, build_predictor, build_cast_val_dataset,
    CastWorldModel, make_cast_wm_unroll, evaluate_one_example,
    custom_collate_fn,
)

from octo.data.utils.text_processing import HFTokenizer
from octo.model.octo_model_pt import OctoModelPt
from torch.utils.data import DataLoader
from octo.utils.torch_rlds_dataset import TorchRLDSDataset

import logging
import warnings

# Suppress verbose logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class MPPIOptimizer:
    """Manages the MPPI hyperparameter optimization process."""
    
    def __init__(
        self,
        wm_model,
        octo_model,
        octo_ds_stats,
        cfg,
        val_loader,
        device,
        n_eval_examples=20,
        mppi_horizon=7,
        n_octo_samples=4,
    ):
        self.wm_model = wm_model
        self.octo_model = octo_model
        self.octo_ds_stats = octo_ds_stats
        self.cfg = cfg
        self.val_loader = val_loader
        self.device = device
        self.n_eval_examples = n_eval_examples
        self.mppi_horizon = mppi_horizon
        self.n_octo_samples = n_octo_samples
        self.cast_wm_unroll = make_cast_wm_unroll(wm_model, cfg)
        
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function: evaluates one set of hyperparameters."""
        
        # Sample hyperparameters with step sizes for discretization
        mppi_iterations = trial.suggest_int("mppi_iterations", 2, 32, step=2)
        mppi_samples = trial.suggest_int("mppi_samples", 16, 512, log=True)  # log scale, no step
        mppi_elites = trial.suggest_int("mppi_elites", 4, 64, log=True)  # log scale, no step
        mppi_temperature = trial.suggest_float("mppi_temperature", 0.1, 2.0, step=0.1)
        mppi_max_std = trial.suggest_float("mppi_max_std", 0.5, 1.0, step=0.1)
        mppi_min_std = trial.suggest_float("mppi_min_std", 0.01, 0.5, step=0.01)
        
        # Skip trial if elites >= samples (invalid configuration)
        if mppi_elites >= mppi_samples:
            raise optuna.TrialPruned()
        
        # Build MPPI config
        mppi_cfg = {
            "horizon": self.mppi_horizon,
            "iterations": mppi_iterations,
            "num_samples": mppi_samples,
            "num_elites": mppi_elites,
            "temperature": mppi_temperature,
            "max_std": mppi_max_std,
            "min_std": mppi_min_std,
            "n_octo_samples": self.n_octo_samples,
        }
        
        # Evaluate on validation examples
        ate_xy_finals = []
        ate_xy_means = []
        rpe_xy_means = []
        total_time = 0.0
        
        val_iter = iter(self.val_loader)
        
        for i in range(self.n_eval_examples):
            batch = None
            while batch is None:
                try:
                    candidate = next(val_iter)
                except StopIteration:
                    break
                if (candidate["observation"]["image_primary"][:, -1] == 255).all():
                    continue
                batch = candidate
            
            if batch is None:
                break
            
            t_start = time.time()
            result = evaluate_one_example(
                batch=batch,
                wm_model=self.wm_model,
                octo_model=self.octo_model,
                octo_ds_stats=self.octo_ds_stats,
                cfg=self.cfg,
                device=self.device,
                cast_wm_unroll=self.cast_wm_unroll,
                mppi_cfg=mppi_cfg,
            )
            t_elapsed = time.time() - t_start
            total_time += t_elapsed
            
            # Extract metrics for octo_mppi (the method we're optimizing)
            metrics = result.metrics_octo_mppi
            ate_xy_finals.append(metrics.get("ate_xy_final", float("inf")))
            ate_xy_means.append(metrics.get("ate_xy_mean", float("inf")))
            rpe_xy_means.append(metrics.get("rpe_xy_mean", float("inf")))
            
            # Optuna pruning: stop early if performance is poor
            trial.report(np.mean(ate_xy_finals), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        if not ate_xy_finals:
            return float("inf")
        
        # Primary objective: minimize final ATE XY
        # Secondary: also consider mean ATE and RPE
        primary_metric = np.mean(ate_xy_means)
        secondary_metric = np.mean(ate_xy_finals)
        tertiary_metric = np.mean(rpe_xy_means)
        
        # Store additional metrics for analysis
        trial.set_user_attr("ate_xy_final_mean", primary_metric)
        trial.set_user_attr("ate_xy_final_std", np.std(ate_xy_finals))
        trial.set_user_attr("ate_xy_mean_mean", secondary_metric)
        trial.set_user_attr("rpe_xy_mean_mean", tertiary_metric)
        trial.set_user_attr("avg_time_per_example_ms", (total_time / len(ate_xy_finals)) * 1000)
        
        # Weighted objective (prioritize final ATE but also consider others)
        combined_objective = primary_metric + 0.2 * secondary_metric + 0.1 * tertiary_metric
        
        return combined_objective


def main():
    parser = argparse.ArgumentParser(description="Optimize MPPI hyperparameters")
    parser.add_argument("--n_trials", type=int, default=100,
                        help="Number of Optuna trials to run")
    parser.add_argument("--n_eval_examples", type=int, default=20,
                        help="Number of validation examples per trial")
    parser.add_argument("--encoder_type", type=str, default="dino", choices=["dino", "vjepa"])
    parser.add_argument("--wm_checkpoint", type=str,
                        default="/mnt/weka/zhougrp/octo_wm_cast/dino_cast_wm_20260228_193050/best_model.pt")
    parser.add_argument("--octo_ckpt_dir", type=str,
                        default="/mnt/weka/zhougrp/octo_pt_cast/cast_dino_finetune_20260219_202345")
    parser.add_argument("--octo_ckpt_step", type=int, default=2000)
    parser.add_argument("--mppi_horizon", type=int, default=7)
    parser.add_argument("--n_octo_samples", type=int, default=4)
    parser.add_argument("--study_name", type=str, default="mppi_optimization")
    parser.add_argument("--output_dir", type=str, default="results/mppi_optimization")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel jobs (1 for sequential)")
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tf.random.set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f"MPPI Hyperparameter Optimization")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Trials: {args.n_trials}")
    print(f"Eval examples per trial: {args.n_eval_examples}")
    print(f"Fixed n_octo_samples: {args.n_octo_samples}")
    print(f"{'='*80}\n")
    
    # Load world model config
    config_path = os.path.join(REPO_ROOT, "scripts/configs/train_cast_wm_config.py")
    spec = importlib.util.spec_from_file_location("cfg_module", config_path)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    cfg = cfg_module.get_config(encoder_type=args.encoder_type)
    
    # Build and load world model
    print("Loading world model...")
    predictor = build_predictor(cfg)
    encoder = build_encoder(cfg)
    wm_model = CastWorldModel(encoder, predictor, cfg).to(device)
    for p in wm_model.encoder.parameters():
        p.requires_grad = False
    wm_model.encoder.eval()
    
    ckpt = torch.load(args.wm_checkpoint, map_location=device, weights_only=False)
    wm_model.load_state_dict(ckpt["model_state_dict"])
    wm_model.eval()
    print(f"  ✓ WM loaded (step={ckpt.get('step', 'N/A')})")
    
    # Load Octo model
    print(f"Loading Octo model...")
    octo_load = OctoModelPt.load_pretrained(args.octo_ckpt_dir, step=args.octo_ckpt_step)
    octo_model = octo_load["octo_model"].to(device)
    octo_model.eval()
    octo_ds_stats = octo_model.dataset_statistics
    print(f"  ✓ Octo loaded (step={args.octo_ckpt_step})")
    
    # Build validation dataset
    print("Building validation dataset...")
    text_processor = HFTokenizer(
        tokenizer_name="t5-base", encode_with_model=False,
        tokenizer_kwargs={
            "max_length": 16, "padding": "max_length",
            "truncation": True, "return_tensors": "np",
        },
    )
    val_tf_ds = build_cast_val_dataset(cfg)
    val_pt = TorchRLDSDataset(val_tf_ds, text_processor, train=False)
    val_loader = DataLoader(
        val_pt, batch_size=1, num_workers=0,
        collate_fn=custom_collate_fn, pin_memory=True,
    )
    print("  ✓ Validation dataset ready\n")
    
    # Create optimizer
    optimizer = MPPIOptimizer(
        wm_model=wm_model,
        octo_model=octo_model,
        octo_ds_stats=octo_ds_stats,
        cfg=cfg,
        val_loader=val_loader,
        device=device,
        n_eval_examples=args.n_eval_examples,
        mppi_horizon=args.mppi_horizon,
        n_octo_samples=args.n_octo_samples,
    )
    
    # Create Optuna study
    os.makedirs(args.output_dir, exist_ok=True)
    db_path = os.path.join(args.output_dir, f"{args.study_name}.db")
    storage_url = f"sqlite:///{db_path}"
    
    # Check if study exists and warn about step constraints
    if os.path.exists(db_path):
        print(f"⚠ WARNING: Existing study database found at {db_path}")
        print(f"  Old trials may not respect current step constraints.")
        print(f"  To start fresh, either:")
        print(f"    1. Delete the database: rm {db_path}")
        print(f"    2. Use a different --study_name")
        print(f"  Continuing with existing study...\n")
    
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    
    print(f"Starting optimization ({args.n_trials} trials)...\n")
    
    # Run optimization
    study.optimize(
        optimizer.objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )
    
    # Results
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    
    # Best trial
    best_trial = study.best_trial
    print(f"\n{'='*80}")
    print(f"BEST TRIAL (Trial #{best_trial.number})")
    print(f"{'='*80}")
    print(f"Combined objective value: {best_trial.value:.6f}")
    print(f"  ATE XY Final (mean): {best_trial.user_attrs.get('ate_xy_final_mean', 'N/A'):.6f}")
    print(f"  ATE XY Final (std):  {best_trial.user_attrs.get('ate_xy_final_std', 'N/A'):.6f}")
    print(f"  ATE XY Mean (mean):  {best_trial.user_attrs.get('ate_xy_mean_mean', 'N/A'):.6f}")
    print(f"  RPE XY Mean (mean):  {best_trial.user_attrs.get('rpe_xy_mean_mean', 'N/A'):.6f}")
    print(f"  Avg time per example: {best_trial.user_attrs.get('avg_time_per_example_ms', 'N/A'):.1f} ms")
    
    print(f"\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results = {
        "study_name": args.study_name,
        "n_trials": len(study.trials),
        "n_complete_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "config": {
            "encoder_type": args.encoder_type,
            "wm_checkpoint": args.wm_checkpoint,
            "octo_ckpt_dir": args.octo_ckpt_dir,
            "octo_ckpt_step": args.octo_ckpt_step,
            "mppi_horizon": args.mppi_horizon,
            "n_octo_samples": args.n_octo_samples,
            "n_eval_examples": args.n_eval_examples,
            "seed": args.seed,
        },
        "best_trial": {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params,
            "user_attrs": best_trial.user_attrs,
        },
        "all_trials": [
            {
                "number": t.number,
                "value": t.value if t.state == optuna.trial.TrialState.COMPLETE else None,
                "params": t.params,
                "user_attrs": t.user_attrs,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }
    
    output_json = os.path.join(args.output_dir, f"{args.study_name}_results.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_json}")
    
    # Generate command for best parameters
    print(f"\n{'='*80}")
    print(f"RECOMMENDED COMMAND")
    print(f"{'='*80}")
    cmd = (
        f"python scripts/eval_octo_wm_plan.py "
        f"--mppi_iterations {best_trial.params['mppi_iterations']} "
        f"--mppi_samples {best_trial.params['mppi_samples']} "
        f"--mppi_elites {best_trial.params['mppi_elites']} "
        f"--mppi_temperature {best_trial.params['mppi_temperature']:.4f} "
        f"--mppi_max_std {best_trial.params['mppi_max_std']:.4f} "
        f"--mppi_min_std {best_trial.params['mppi_min_std']:.4f} "
        f"--n_octo_samples {args.n_octo_samples}"
    )
    print(cmd)
    print(f"{'='*80}\n")
    
    # Save command to file
    cmd_file = os.path.join(args.output_dir, "best_command.sh")
    with open(cmd_file, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Best hyperparameters from optimization trial #{best_trial.number}\n")
        f.write(f"# Combined objective: {best_trial.value:.6f}\n\n")
        f.write(cmd + "\n")
    os.chmod(cmd_file, 0o755)
    print(f"Command saved to: {cmd_file}")


if __name__ == "__main__":
    main()
