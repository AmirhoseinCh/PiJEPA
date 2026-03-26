#!/usr/bin/env python3
"""
Load saved .npz samples from save_eval_samples.py and create
publication-quality figures for the CVPR paper.

Produces:
  1. Qualitative grid: start image | goal image | trajectory overlay (per example)
  2. Trajectory-only comparison plots (GT vs 4 methods)
  3. Aggregate bar chart of ATE/RPE metrics across methods
  4. Individual sample folders (with --save_individual flag):
     - start_image.png, goal_image.png
     - frame_XX.png (all frames)
     - decoded_XX.png (decoded WM predictions)
     - trajectory.png
     - all_frames.png (image strip)
     - metrics.json

Usage:
    # Generate aggregate figures
    python plot_paper_figures.py \
        --load_dir ./eval_samples/dino_run1 \
        --out_dir  ./figures \
        --sample_idxs 0 3 7 12 \
        --dpi 300
    
    # Save each sample to its own folder
    python plot_paper_figures.py \
        --load_dir ./eval_samples/dino_run1 \
        --out_dir  ./figures \
        --save_individual \
        --dpi 150
"""

import argparse
import json
import os
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import FancyArrowPatch
from PIL import Image


# ═══════════════════════════════════════════════════════════════════════════════
# Loading utilities
# ═══════════════════════════════════════════════════════════════════════════════

def load_sample(npz_path):
    """Load one .npz sample and return a dict with all fields."""
    data = np.load(npz_path, allow_pickle=True)
    sample = {
        "images": data["images"],               # (T, H, W, C) uint8
        "start_image": data["start_image"],      # (H, W, C)
        "goal_image": data["goal_image"],        # (H, W, C)
        "decoded_images": data.get("decoded_images", None),  # (H_octo, C, H, W) or None
        "text": str(data["text"]),
        "dataset_name": str(data["dataset_name"]),
        # Trajectories
        "gt_xy": data["gt_xy"],
        "gt_heading": data["gt_heading"],
        "default_mppi_xy": data["default_mppi_xy"],
        "default_mppi_heading": data["default_mppi_heading"],
        "octo_mppi_xy": data["octo_mppi_xy"],
        "octo_mppi_heading": data["octo_mppi_heading"],
        "octo_wm_xy": data["octo_wm_xy"],
        "octo_wm_heading": data["octo_wm_heading"],
        "octo_mean_xy": data["octo_mean_xy"],
        "octo_mean_heading": data["octo_mean_heading"],
        # Metrics
        "metrics_default_mppi": json.loads(str(data["metrics_default_mppi"])),
        "metrics_octo_mppi": json.loads(str(data["metrics_octo_mppi"])),
        "metrics_octo_wm": json.loads(str(data["metrics_octo_wm"])),
        "metrics_octo_mean": json.loads(str(data["metrics_octo_mean"])),
    }
    return sample


def load_all_samples(load_dir):
    """Load all .npz samples from a directory, sorted by index."""
    files = sorted(glob(os.path.join(load_dir, "sample_*.npz")))
    samples = []
    for f in files:
        samples.append(load_sample(f))
    return samples


def load_summary(load_dir):
    """Load summary.json if it exists."""
    path = os.path.join(load_dir, "summary.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Color / style config (CVPR-friendly)
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "gt":           "#2d2d2d",   # dark gray
    "default_mppi": "#d62728",   # red
    "octo_mppi":    "#2ca02c",   # green
    "octo_wm":      "#ff7f0e",   # orange
    "octo_mean":    "#1f77b4",   # blue
}

LABELS = {
    "gt":           "GT",
    "default_mppi": "MPPI",
    "octo_mppi":    "PiJEPA",
    "octo_wm":      "Octo-WM",
    "octo_mean":    "Octo",
}

MARKERS = {
    "gt":           "s",
    "default_mppi": "^",
    "octo_mppi":    "o",
    "octo_wm":      "D",
    "octo_mean":    "v",
}


def _style_traj_ax(ax, title=None):
    """Apply clean styling to a trajectory axis."""
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=7)
    if title:
        ax.set_title(title, fontsize=8, pad=4)


def plot_trajectory(ax, xy, color, label, marker="o", lw=1.5, ms=4, zorder=3):
    """Plot a 2D trajectory on an axis."""
    ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=lw, label=label,
            zorder=zorder, solid_capstyle="round")
    # Start marker
    ax.plot(xy[0, 0], xy[0, 1], marker=marker, color=color, markersize=ms+1,
            markeredgecolor="white", markeredgewidth=0.5, zorder=zorder+1)
    # End marker
    ax.plot(xy[-1, 0], xy[-1, 1], marker="*", color=color, markersize=ms+2,
            markeredgecolor="white", markeredgewidth=0.5, zorder=zorder+1)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Qualitative grid (images + trajectories)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_qualitative_grid(samples, idxs, out_path, dpi=300):
    """
    Create a grid: each row is one example.
    Columns: start image | goal image | trajectory plot.
    """
    n = len(idxs)
    fig = plt.figure(figsize=(7.0, 2.0 * n + 0.4), dpi=dpi)
    gs = gridspec.GridSpec(n, 3, width_ratios=[1, 1, 1.6],
                           hspace=0.35, wspace=0.15,
                           left=0.02, right=0.98, top=0.95, bottom=0.05)

    for row, idx in enumerate(idxs):
        s = samples[idx]

        # Start image
        ax_start = fig.add_subplot(gs[row, 0])
        ax_start.imshow(s["start_image"])
        ax_start.axis("off")
        if row == 0:
            ax_start.set_title("Start $o_t$", fontsize=9, pad=4)

        # Goal image
        ax_goal = fig.add_subplot(gs[row, 1])
        ax_goal.imshow(s["goal_image"])
        ax_goal.axis("off")
        if row == 0:
            ax_goal.set_title("Goal $o_g$", fontsize=9, pad=4)

        # Trajectory
        ax_traj = fig.add_subplot(gs[row, 2])
        plot_trajectory(ax_traj, s["gt_xy"], COLORS["gt"], LABELS["gt"],
                        marker=MARKERS["gt"], lw=2.0, zorder=5)
        plot_trajectory(ax_traj, s["default_mppi_xy"], COLORS["default_mppi"],
                        LABELS["default_mppi"], marker=MARKERS["default_mppi"])
        plot_trajectory(ax_traj, s["octo_mean_xy"], COLORS["octo_mean"],
                        LABELS["octo_mean"], marker=MARKERS["octo_mean"])
        plot_trajectory(ax_traj, s["octo_wm_xy"], COLORS["octo_wm"],
                        LABELS["octo_wm"], marker=MARKERS["octo_wm"])
        plot_trajectory(ax_traj, s["octo_mppi_xy"], COLORS["octo_mppi"],
                        LABELS["octo_mppi"], marker=MARKERS["octo_mppi"], lw=2.0, zorder=4)
        _style_traj_ax(ax_traj)

        # Instruction text as annotation
        instr = s["text"]
        if len(instr) > 50:
            instr = instr[:47] + "..."
        ax_traj.text(0.02, 0.97, f'"{instr}"', transform=ax_traj.transAxes,
                     fontsize=5.5, va="top", ha="left", style="italic",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, lw=0.3))

        if row == 0:
            ax_traj.set_title("Predicted Trajectories", fontsize=9, pad=4)

    # Shared legend at bottom
    handles, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7,
               frameon=True, fancybox=False, edgecolor="0.8",
               bbox_to_anchor=(0.5, -0.01))

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved qualitative grid: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Single example trajectory comparison (larger, more detail)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_single_trajectory(sample, out_path, dpi=300):
    """Large trajectory comparison for one example with heading arrows."""
    fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.5), dpi=dpi)

    methods = ["gt", "default_mppi", "octo_mean", "octo_wm", "octo_mppi"]
    for m in methods:
        xy = sample[f"{m}_xy"]
        lw = 2.5 if m in ("gt", "octo_mppi") else 1.5
        zorder = 5 if m == "gt" else (4 if m == "octo_mppi" else 3)
        plot_trajectory(ax, xy, COLORS[m], LABELS[m], marker=MARKERS[m],
                        lw=lw, ms=5, zorder=zorder)

        # Draw heading arrows at each waypoint
        heading = sample[f"{m}_heading"]
        arrow_len = 0.02 * max(xy[:, 0].ptp(), xy[:, 1].ptp()) + 0.005
        for t in range(len(heading)):
            if t >= len(xy):
                break
            dx = arrow_len * np.cos(heading[t])
            dy = arrow_len * np.sin(heading[t])
            ax.annotate("", xy=(xy[t, 0]+dx, xy[t, 1]+dy),
                         xytext=(xy[t, 0], xy[t, 1]),
                         arrowprops=dict(arrowstyle="->", color=COLORS[m],
                                         lw=0.8, mutation_scale=6),
                         zorder=zorder)

    _style_traj_ax(ax)
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9)

    instr = sample["text"]
    if len(instr) > 60:
        instr = instr[:57] + "..."
    ax.set_title(f'"{instr}"', fontsize=8, style="italic", pad=6)

    ax.legend(fontsize=7, loc="best", framealpha=0.9, edgecolor="0.8")

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved trajectory plot: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Aggregate bar chart of metrics
# ═══════════════════════════════════════════════════════════════════════════════

def fig_aggregate_bars(samples, out_path, dpi=300):
    """Bar chart comparing methods on ATE and RPE (mean ± std across samples)."""
    method_keys = ["default_mppi", "octo_mean", "octo_wm", "octo_mppi"]
    method_labels = [LABELS[m] for m in method_keys]
    method_colors = [COLORS[m] for m in method_keys]

    # Metrics to display
    metric_groups = [
        ("ATE XY Final (m)",       "ate_xy_final"),
        ("ATE XY RMSE (m)",        "ate_xy_rmse"),
        ("ATE Heading Final (°)",  "ate_heading_final_deg"),
        ("RPE XY RMSE (m)",        "rpe_xy_rmse"),
    ]

    n_groups = len(metric_groups)
    n_methods = len(method_keys)
    x = np.arange(n_groups)
    bar_w = 0.8 / n_methods

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.8), dpi=dpi)

    for j, mkey in enumerate(method_keys):
        means, stds = [], []
        for _, metric_name in metric_groups:
            vals = [s[f"metrics_{mkey}"][metric_name] for s in samples]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        offset = (j - n_methods / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, means, bar_w, yerr=stds,
                       color=method_colors[j], label=method_labels[j],
                       edgecolor="white", linewidth=0.5,
                       capsize=2, error_kw={"linewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in metric_groups], fontsize=7.5)
    ax.set_ylabel("Error", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9, edgecolor="0.8")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved aggregate bar chart: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Per-example ATE scatter / violin
# ═══════════════════════════════════════════════════════════════════════════════

def fig_ate_scatter(samples, out_path, dpi=300):
    """Scatter plot: each dot is one example, comparing Octo-MPPI vs others."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), dpi=dpi)

    comparisons = [
        ("default_mppi", "Default MPPI"),
        ("octo_mean",    "Octo-Mean"),
        ("octo_wm",      "Octo-WM"),
    ]

    for ax, (mkey, mlabel) in zip(axes, comparisons):
        ours_vals = [s["metrics_octo_mppi"]["ate_xy_final"] for s in samples]
        other_vals = [s[f"metrics_{mkey}"]["ate_xy_final"] for s in samples]

        ax.scatter(other_vals, ours_vals, s=18, alpha=0.7, c=COLORS["octo_mppi"],
                   edgecolors="white", linewidths=0.3, zorder=3)
        lim = max(max(ours_vals), max(other_vals)) * 1.1 + 0.01
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5, zorder=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel(f"{mlabel} ATE (m)", fontsize=7.5)
        ax.set_ylabel("Octo-MPPI ATE (m)", fontsize=7.5)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6.5)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        # Count wins
        wins = sum(o < b for o, b in zip(ours_vals, other_vals))
        ax.text(0.95, 0.05, f"Ours better: {wins}/{len(samples)}",
                transform=ax.transAxes, fontsize=6, ha="right", va="bottom",
                bbox=dict(fc="white", alpha=0.8, lw=0.3, boxstyle="round,pad=0.2"))

    fig.suptitle("Per-Example ATE XY Final: Octo-MPPI vs. Baselines", fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved ATE scatter: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: Image strip (all frames in a row for one example)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_image_strip(sample, out_path, dpi=300):
    """Show all observation frames in a horizontal strip."""
    imgs = sample["images"]  # (T, H, W, C)
    T = imgs.shape[0]
    fig, axes = plt.subplots(1, T, figsize=(1.5 * T, 1.5), dpi=dpi)
    if T == 1:
        axes = [axes]
    for t, ax in enumerate(axes):
        ax.imshow(imgs[t])
        ax.axis("off")
        label = "$o_t$" if t == 0 else ("$o_g$" if t == T - 1 else f"$o_{{{t}}}$")
        ax.set_title(label, fontsize=8, pad=2)
    fig.suptitle(f'"{sample["text"]}"', fontsize=7, style="italic", y=0.02)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved image strip: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Save individual sample to folder
# ═══════════════════════════════════════════════════════════════════════════════

def save_sample_to_folder(sample, idx, sample_folder, dpi=300, ext="png"):
    """
    Save all data for one sample to its own folder:
    - start_image.png
    - goal_image.png
    - decoded_*.png (if available)
    - trajectory.png
    - all_frames.png (image strip)
    - metrics.json
    """
    os.makedirs(sample_folder, exist_ok=True)
    
    # Save start and goal images
    Image.fromarray(sample["start_image"]).save(os.path.join(sample_folder, "start_image.png"))
    Image.fromarray(sample["goal_image"]).save(os.path.join(sample_folder, "goal_image.png"))
    
    # Save all frames as separate images
    for t, img in enumerate(sample["images"]):
        Image.fromarray(img).save(os.path.join(sample_folder, f"frame_{t:02d}.png"))
    
    # Save decoded images if available
    if sample["decoded_images"] is not None:
        decoded = sample["decoded_images"]  # Expected: (H_octo, ...) with various shapes
        for t in range(decoded.shape[0]):
            img = decoded[t]  # Extract single frame
            
            # Squeeze out any singleton dimensions (batch dims, etc.)
            while img.ndim > 3:
                img = np.squeeze(img, axis=0)
            
            # Handle different formats: (C, H, W) vs (H, W, C)
            if img.ndim == 3:
                if img.shape[0] in [1, 3, 4]:  # Likely (C, H, W)
                    img = img.transpose(1, 2, 0)  # -> (H, W, C)
                # else: already (H, W, C)
            
            # Normalize to [0, 255] uint8
            if img.dtype == np.float32 or img.dtype == np.float64:
                if img.max() <= 1.0:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                else:
                    img = img.clip(0, 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            
            # Handle grayscale or single channel
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            
            Image.fromarray(img).save(os.path.join(sample_folder, f"decoded_{t:02d}.png"))
    
    # Save trajectory plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=dpi)
    methods = ["gt", "default_mppi", "octo_mean", "octo_wm", "octo_mppi"]
    for m in methods:
        xy = sample[f"{m}_xy"]
        lw = 2.5 if m in ("gt", "octo_mppi") else 1.5
        zorder = 5 if m == "gt" else (4 if m == "octo_mppi" else 3)
        plot_trajectory(ax, xy, COLORS[m], LABELS[m], marker=MARKERS[m],
                        lw=lw, ms=5, zorder=zorder)
    
    _style_traj_ax(ax)
    ax.set_xlabel("x (m)", fontsize=10)
    ax.set_ylabel("y (m)", fontsize=10)
    
    # Add instruction text inside the plot at the top
    # instr = sample["text"]
    # if len(instr) > 60:
    #     instr = instr[:57] + "..."
    # ax.text(0.5, 0.98, f'"{instr}"', transform=ax.transAxes,
    #         fontsize=7.5, va="top", ha="center", style="italic",
    #         bbox=dict(boxstyle="round,pad=0.4", fc="wheat", alpha=0.85, lw=0.5))
    
    # ax.set_title(f"Sample {idx}", fontsize=10, pad=8, fontweight="bold")
    
    # Place legend outside the plot area on the right
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5), 
              framealpha=0.95, edgecolor="0.8")
    
    fig.savefig(os.path.join(sample_folder, f"trajectory.{ext}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    
    # Save image strip
    imgs = sample["images"]
    T = imgs.shape[0]
    fig, axes = plt.subplots(1, T, figsize=(1.5 * T, 1.5), dpi=dpi)
    if T == 1:
        axes = [axes]
    for t, ax in enumerate(axes):
        ax.imshow(imgs[t])
        ax.axis("off")
        label = "$o_t$" if t == 0 else ("$o_g$" if t == T - 1 else f"$o_{{{t}}}$")
        ax.set_title(label, fontsize=9, pad=3)
    fig.savefig(os.path.join(sample_folder, f"all_frames.{ext}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    
    # Save metrics as JSON
    metrics_data = {
        "idx": idx,
        "text": sample["text"],
        "dataset_name": sample["dataset_name"],
        "metrics": {
            "default_mppi": sample["metrics_default_mppi"],
            "octo_mppi": sample["metrics_octo_mppi"],
            "octo_wm": sample["metrics_octo_wm"],
            "octo_mean": sample["metrics_octo_mean"],
        }
    }
    with open(os.path.join(sample_folder, "metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"  Saved sample {idx} to {sample_folder}/")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Plot CVPR figures from saved samples")
    parser.add_argument("--load_dir", type=str, required=True,
                        help="Directory with sample_*.npz and summary.json")
    parser.add_argument("--out_dir", type=str, default="./figures",
                        help="Output directory for figures")
    parser.add_argument("--sample_idxs", type=int, nargs="*", default=None,
                        help="Which sample indices to use for qualitative figs (default: first 4)")
    parser.add_argument("--save_individual", action="store_true",
                        help="Save each sample to its own folder with separate images and plots")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png", "svg"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ext = args.format

    print(f"Loading samples from {args.load_dir}...")
    samples = load_all_samples(args.load_dir)
    summary = load_summary(args.load_dir)
    print(f"Loaded {len(samples)} samples.")

    if summary:
        encoder = summary["config"].get("encoder_type", "?")
        print(f"Encoder: {encoder}, MPPI config: {summary['config'].get('mppi', {})}")

    # Pick indices for qualitative figures
    if args.sample_idxs is not None:
        qual_idxs = [i for i in args.sample_idxs if i < len(samples)]
    else:
        qual_idxs = list(range(min(1000, len(samples))))

    # ── Save individual samples to folders ───────────────────────────────
    if args.save_individual:
        print("\nSaving individual samples to folders...")
        # items_to_save = args.sample_idxs if args.sample_idxs is not None else range(len(samples))
        items_to_save = range(len(samples))
        for idx in items_to_save:
            sample_folder = os.path.join(args.out_dir, f"sample_{idx:04d}")
            save_sample_to_folder(samples[idx], idx, sample_folder, dpi=args.dpi, ext="png")

    # ── Generate all figures ─────────────────────────────────────────────

    # 1. Qualitative grid
    if len(qual_idxs) >= 1:
        fig_qualitative_grid(
            samples, qual_idxs,
            os.path.join(args.out_dir, f"qualitative_grid.{ext}"), dpi=args.dpi)

    # 2. Individual trajectory plots
    for idx in qual_idxs:
        fig_single_trajectory(
            samples[idx],
            os.path.join(args.out_dir, f"trajectory_sample_{idx:04d}.{ext}"), dpi=args.dpi)

    # 3. Image strips
    for idx in qual_idxs:
        fig_image_strip(
            samples[idx],
            os.path.join(args.out_dir, f"image_strip_{idx:04d}.{ext}"), dpi=args.dpi)

    # 4. Aggregate bar chart
    if len(samples) >= 3:
        fig_aggregate_bars(
            samples,
            os.path.join(args.out_dir, f"aggregate_bars.{ext}"), dpi=args.dpi)

    # 5. ATE scatter comparison
    if len(samples) >= 5:
        fig_ate_scatter(
            samples,
            os.path.join(args.out_dir, f"ate_scatter.{ext}"), dpi=args.dpi)

    print(f"\nAll figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
