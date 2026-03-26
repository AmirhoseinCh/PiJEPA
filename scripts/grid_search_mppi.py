#!/usr/bin/env python3
"""
Grid search for MPPI hyperparameter optimization.

Simpler alternative to Bayesian optimization - tests all combinations
in a predefined grid. No external dependencies beyond the main script.

Usage:
    python scripts/grid_search_mppi.py --n_eval_examples 30
"""

import argparse
import importlib.util
import itertools
import json
import os
import random
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# Import from main evaluation script
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

# Suppress logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def evaluate_hyperparams(
    hyperparams,
    wm_model,
    octo_model,
    octo_ds_stats,
    cfg,
    val_loader,
    device,
    n_eval_examples,
    mppi_horizon,
    n_octo_samples,
):
    """Evaluate one set of hyperparameters."""
    
    mppi_cfg = {
        "horizon": mppi_horizon,
        "iterations": hyperparams["mppi_iterations"],
        "num_samples": hyperparams["mppi_samples"],
        "num_elites": hyperparams["mppi_elites"],
        "temperature": hyperparams["mppi_temperature"],
        "max_std": hyperparams["mppi_max_std"],
        "min_std": hyperparams["mppi_min_std"],
        "n_octo_samples": n_octo_samples,
    }
    
    cast_wm_unroll = make_cast_wm_unroll(wm_model, cfg)
    
    ate_xy_finals = []
    ate_xy_means = []
    rpe_xy_means = []
    total_time = 0.0
    
    val_iter = iter(val_loader)
    
    for i in range(n_eval_examples):
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
            wm_model=wm_model,
            octo_model=octo_model,
            octo_ds_stats=octo_ds_stats,
            cfg=cfg,
            device=device,
            cast_wm_unroll=cast_wm_unroll,
            mppi_cfg=mppi_cfg,
        )
        t_elapsed = time.time() - t_start
        total_time += t_elapsed
        
        metrics = result.metrics_octo_mppi
        ate_xy_finals.append(metrics.get("ate_xy_final", float("inf")))
        ate_xy_means.append(metrics.get("ate_xy_mean", float("inf")))
        rpe_xy_means.append(metrics.get("rpe_xy_mean", float("inf")))
    
    if not ate_xy_finals:
        return None
    
    return {
        "hyperparams": hyperparams,
        "ate_xy_final_mean": float(np.mean(ate_xy_finals)),
        "ate_xy_final_std": float(np.std(ate_xy_finals)),
        "ate_xy_mean_mean": float(np.mean(ate_xy_means)),
        "rpe_xy_mean_mean": float(np.mean(rpe_xy_means)),
        "avg_time_per_example_ms": (total_time / len(ate_xy_finals)) * 1000,
        "n_examples": len(ate_xy_finals),
    }


def main():
    parser = argparse.ArgumentParser(description="Grid search for MPPI hyperparameters")
    parser.add_argument("--n_eval_examples", type=int, default=30,
                        help="Number of validation examples per configuration")
    parser.add_argument("--encoder_type", type=str, default="dino", choices=["dino", "vjepa"])
    parser.add_argument("--wm_checkpoint", type=str,
                        default="/mnt/weka/zhougrp/octo_wm_cast/dino_cast_wm_20260228_193050/best_model.pt")
    parser.add_argument("--octo_ckpt_dir", type=str,
                        default="/mnt/weka/zhougrp/octo_pt_cast/cast_dino_finetune_20260219_202345")
    parser.add_argument("--octo_ckpt_step", type=int, default=2000)
    parser.add_argument("--mppi_horizon", type=int, default=7)
    parser.add_argument("--n_octo_samples", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="results/mppi_grid_search")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Define search space
    search_space = {
        "mppi_iterations": [3, 6, 12, 16],
        "mppi_samples": [16, 32, 64, 128, 256],
        "mppi_elites": [4, 8, 16, 32],
        "mppi_temperature": [0.3, 0.5, 1.0, 1.5],
        "mppi_max_std": [0.5, 1.0, 2.0],
        "mppi_min_std": [0.01, 0.05, 0.1],
    }
    
    # Generate all combinations
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    all_combinations = list(itertools.product(*values))
    
    # Filter invalid combinations (elites >= samples)
    valid_combinations = []
    for combo in all_combinations:
        param_dict = dict(zip(keys, combo))
        if param_dict["mppi_elites"] < param_dict["mppi_samples"]:
            valid_combinations.append(param_dict)
    
    total_combinations = len(valid_combinations)
    
    print(f"\n{'='*80}")
    print(f"MPPI Grid Search")
    print(f"{'='*80}")
    print(f"Total combinations: {total_combinations}")
    print(f"Eval examples per config: {args.n_eval_examples}")
    print(f"Fixed n_octo_samples: {args.n_octo_samples}")
    print(f"Seed: {args.seed}")
    print(f"Estimated time: {total_combinations * args.n_eval_examples * 10 / 3600:.1f} hours")
    print(f"  (assuming ~10s per example)")
    print(f"{'='*80}\n")
    
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
    print(f"Device: {device}\n")
    
    # Load models
    config_path = os.path.join(REPO_ROOT, "scripts/configs/train_cast_wm_config.py")
    spec = importlib.util.spec_from_file_location("cfg_module", config_path)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    cfg = cfg_module.get_config(encoder_type=args.encoder_type)
    
    print("Loading models...")
    predictor = build_predictor(cfg)
    encoder = build_encoder(cfg)
    wm_model = CastWorldModel(encoder, predictor, cfg).to(device)
    for p in wm_model.encoder.parameters():
        p.requires_grad = False
    wm_model.encoder.eval()
    
    ckpt = torch.load(args.wm_checkpoint, map_location=device, weights_only=False)
    wm_model.load_state_dict(ckpt["model_state_dict"])
    wm_model.eval()
    print(f"  ✓ WM loaded")
    
    octo_load = OctoModelPt.load_pretrained(args.octo_ckpt_dir, step=args.octo_ckpt_step)
    octo_model = octo_load["octo_model"].to(device)
    octo_model.eval()
    octo_ds_stats = octo_model.dataset_statistics
    print(f"  ✓ Octo loaded")
    
    print("Building dataset...")
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
    print(f"  ✓ Dataset ready\n")
    
    # Run grid search
    results = []
    best_result = None
    best_score = float("inf")
    
    pbar = tqdm(valid_combinations, desc="Grid Search")
    for i, hyperparams in enumerate(pbar):
        result = evaluate_hyperparams(
            hyperparams=hyperparams,
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
        
        if result is not None:
            result["config_id"] = i
            results.append(result)
            
            # Update best
            score = result["ate_xy_final_mean"]
            if score < best_score:
                best_score = score
                best_result = result
            
            pbar.set_postfix_str(
                f"Best ATE: {best_score:.4f}m | "
                f"Current: {score:.4f}m"
            )
    
    # Sort results
    results.sort(key=lambda x: x["ate_xy_final_mean"])
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"GRID SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"Configurations tested: {len(results)}")
    
    if best_result:
        print(f"\n{'='*80}")
        print(f"BEST CONFIGURATION")
        print(f"{'='*80}")
        print(f"ATE XY Final (mean): {best_result['ate_xy_final_mean']:.6f} m")
        print(f"ATE XY Final (std):  {best_result['ate_xy_final_std']:.6f} m")
        print(f"ATE XY Mean (mean):  {best_result['ate_xy_mean_mean']:.6f} m")
        print(f"RPE XY Mean (mean):  {best_result['rpe_xy_mean_mean']:.6f} m")
        print(f"Avg time per example: {best_result['avg_time_per_example_ms']:.1f} ms")
        
        print(f"\nHyperparameters:")
        for key, value in best_result["hyperparams"].items():
            print(f"  {key}: {value}")
        
        # Top 5 configurations
        print(f"\n{'='*80}")
        print(f"TOP 5 CONFIGURATIONS")
        print(f"{'='*80}")
        for i, res in enumerate(results[:5], 1):
            print(f"\n{i}. ATE XY Final: {res['ate_xy_final_mean']:.6f} m")
            print(f"   Parameters: {res['hyperparams']}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_data = {
        "search_space": search_space,
        "total_combinations": total_combinations,
        "valid_combinations": len(valid_combinations),
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
        "best_result": best_result,
        "all_results": results,
    }
    
    output_json = os.path.join(args.output_dir, "grid_search_results.json")
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_json}")
    
    if best_result:
        # Generate command
        cmd = (
            f"python scripts/eval_octo_wm_plan.py "
            f"--mppi_iterations {best_result['hyperparams']['mppi_iterations']} "
            f"--mppi_samples {best_result['hyperparams']['mppi_samples']} "
            f"--mppi_elites {best_result['hyperparams']['mppi_elites']} "
            f"--mppi_temperature {best_result['hyperparams']['mppi_temperature']:.4f} "
            f"--mppi_max_std {best_result['hyperparams']['mppi_max_std']:.4f} "
            f"--mppi_min_std {best_result['hyperparams']['mppi_min_std']:.4f} "
            f"--n_octo_samples {args.n_octo_samples}"
        )
        
        print(f"\n{'='*80}")
        print(f"RECOMMENDED COMMAND")
        print(f"{'='*80}")
        print(cmd)
        print(f"{'='*80}\n")
        
        cmd_file = os.path.join(args.output_dir, "best_command.sh")
        with open(cmd_file, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Best hyperparameters from grid search\n")
            f.write(f"# ATE XY Final: {best_result['ate_xy_final_mean']:.6f} m\n\n")
            f.write(cmd + "\n")
        os.chmod(cmd_file, 0o755)
        print(f"Command saved to: {cmd_file}")


if __name__ == "__main__":
    main()
