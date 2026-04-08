#!/usr/bin/env python3
"""Post-hoc analysis and visualization for VLM-guided video editing optimization.

Generates diagnostic plots from optimization logs to understand:
  1. Loss trajectories (Qwen NLL, perceptual, regularization)
  2. Per-rubric breakdown (motion vs entities vs overall)
  3. Gradient norm dynamics (detect adversarial regime)
  4. Cross-experiment comparison (successful vs failing edits)
  5. Pareto analysis: edit quality vs perceptual preservation

Usage
-----
    # Single experiment:
    python editing/visualize_optimization.py /path/to/results/QwenVL/experiment/both/

    # Compare all experiments:
    python editing/visualize_optimization.py /path/to/results/QwenVL/ --compare-all

    # Compare specific experiments:
    python editing/visualize_optimization.py \\
        /path/to/results/QwenVL/a_dog_yawning/both/ \\
        /path/to/results/QwenVL/the_dog_rolls_on_the/both/ \\
        --labels "dog yawn (success)" "dog roll (fail)"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _HAS_MPL = True
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f8f8",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    })
except ImportError:
    _HAS_MPL = False
    print("matplotlib not available — generating text reports only.", file=sys.stderr)


def _load_csv(csv_path: Path) -> dict[str, list[float]]:
    """Load optimization CSV log into a dict of column lists."""
    import csv
    data: dict[str, list[float]] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(val))
                except (ValueError, TypeError):
                    data[key].append(float("nan"))
    return data


def _load_config(experiment_dir: Path) -> dict:
    """Load run_config.json from an experiment directory."""
    config_path = experiment_dir / "run_config.json"
    if config_path.exists():
        with config_path.open() as f:
            return json.load(f)
    # Try parent
    config_path = experiment_dir.parent / "run_config.json"
    if config_path.exists():
        with config_path.open() as f:
            return json.load(f)
    return {}


def _find_csv(experiment_dir: Path) -> Path | None:
    """Find the optimization CSV log in an experiment directory."""
    # Direct mode_both/ or mode_audio/ etc.
    candidates = list(experiment_dir.glob("mode_*/optimization_log_qwen_*.csv"))
    if not candidates:
        candidates = list(experiment_dir.glob("optimization_log_qwen_*.csv"))
    if not candidates:
        candidates = list(experiment_dir.glob("**/optimization_log_qwen_*.csv"))
    return candidates[0] if candidates else None


def _get_label(experiment_dir: Path, config: dict) -> str:
    """Generate a human-readable label from experiment path/config."""
    args = config.get("args", {})
    prompt = args.get("edit_prompt", "")
    if prompt:
        words = prompt.split()[:5]
        return " ".join(words)
    return experiment_dir.name


# ---------------------------------------------------------------------------
# Plot 1: Single experiment comprehensive dashboard
# ---------------------------------------------------------------------------

def plot_experiment_dashboard(experiment_dir: Path, output_path: Path | None = None) -> None:
    """Generate a comprehensive dashboard for a single optimization run."""
    csv_path = _find_csv(experiment_dir)
    if csv_path is None:
        print(f"No optimization CSV found in {experiment_dir}", file=sys.stderr)
        return

    data = _load_csv(csv_path)
    config = _load_config(experiment_dir)
    label = _get_label(experiment_dir, config)
    iters = data.get("iter", [])

    if not iters:
        print(f"Empty CSV: {csv_path}", file=sys.stderr)
        return

    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)

    # --- 1. Qwen NLL + yes_prob ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(iters, data["qwen_nll"], "b-", linewidth=2, label="Qwen NLL")
    ax1.set_ylabel("Qwen NLL", color="b")
    ax1.set_xlabel("Iteration")
    ax1.set_title("Qwen Alignment Loss")
    ax1b = ax1.twinx()
    ax1b.plot(iters, data["qwen_yes_prob"], "g--", linewidth=1.5, label="yes_prob")
    ax1b.set_ylabel("P(yes)", color="g")
    ax1b.set_ylim(0, 1)
    # Mark best iteration
    best_mask = data.get("is_best", [])
    if best_mask:
        best_iters = [it for it, b in zip(iters, best_mask) if b == 1.0]
        best_nll = [nll for nll, b in zip(data["qwen_nll"], best_mask) if b == 1.0]
        ax1.scatter(best_iters, best_nll, c="red", zorder=5, s=50, marker="*", label="best")
    ax1.legend(loc="upper left")

    # --- 2. Per-rubric breakdown ---
    ax2 = fig.add_subplot(gs[0, 1])
    has_rubric = False
    for name, color in [("motion", "red"), ("entities", "blue"), ("overall", "green")]:
        key_nll = f"{name}_nll"
        key_prob = f"{name}_yes_prob"
        if key_prob in data:
            has_rubric = True
            probs = [v for v in data[key_prob] if not np.isnan(v)]
            if probs:
                ax2.plot(iters[:len(probs)], probs, color=color, linewidth=1.5, label=name)
    if has_rubric:
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("P(yes)")
        ax2.set_title("Per-Rubric yes_prob")
        ax2.set_ylim(0, 1)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No rubric data", transform=ax2.transAxes, ha="center")

    # --- 3. Gradient norm ---
    ax3 = fig.add_subplot(gs[0, 2])
    grad_norms = data.get("grad_norm", [])
    if grad_norms:
        ax3.plot(iters, grad_norms, "purple", linewidth=1.5)
        ax3.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="clip=1.0")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Gradient Norm")
        ax3.set_title("Gradient Norm (pre-clip)")
        ax3.legend()

    # --- 4. Total loss + components ---
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(iters, data.get("total_loss", []), "k-", linewidth=2, label="total")
    ax4.plot(iters, data.get("qwen_nll", []), "b--", linewidth=1, label="qwen_nll")
    audio_reg = data.get("audio_reg", [])
    text_reg = data.get("text_reg", [])
    perceptual = data.get("perceptual_loss", [])
    if audio_reg and any(v > 0 for v in audio_reg if not np.isnan(v)):
        ax4.plot(iters, audio_reg, "orange", linewidth=1, label="audio_reg")
    if text_reg and any(v > 0 for v in text_reg if not np.isnan(v)):
        ax4.plot(iters, text_reg, "cyan", linewidth=1, label="text_reg")
    if perceptual and any(v > 0 for v in perceptual if not np.isnan(v)):
        ax4.plot(iters, perceptual, "red", linewidth=1.5, label="perceptual")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Loss")
    ax4.set_title("Loss Components")
    ax4.legend(loc="best")

    # --- 5. Perceptual metrics (LPIPS + temporal) ---
    ax5 = fig.add_subplot(gs[1, 1])
    lpips_raw = data.get("lpips_raw", [])
    temporal_raw = data.get("temporal_raw", [])
    has_perceptual = False
    if lpips_raw and any(v > 0 for v in lpips_raw if not np.isnan(v)):
        valid_lpips = [(it, v) for it, v in zip(iters, lpips_raw) if not np.isnan(v) and v > 0]
        if valid_lpips:
            ax5.plot([x[0] for x in valid_lpips], [x[1] for x in valid_lpips],
                     "r-", linewidth=1.5, label="LPIPS (source)")
            has_perceptual = True
    if temporal_raw and any(v > 0 for v in temporal_raw if not np.isnan(v)):
        valid_temp = [(it, v) for it, v in zip(iters, temporal_raw) if not np.isnan(v) and v > 0]
        if valid_temp:
            ax5.plot([x[0] for x in valid_temp], [x[1] for x in valid_temp],
                     "m-", linewidth=1.5, label="Temporal excess")
            has_perceptual = True
    if has_perceptual:
        ax5.set_xlabel("Iteration")
        ax5.set_ylabel("Perceptual Distance")
        ax5.set_title("Perceptual Quality Metrics")
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, "No perceptual data\n(enable with --lpips-weight)",
                 transform=ax5.transAxes, ha="center", va="center", fontsize=10, color="gray")

    # --- 6. Loss component magnitude comparison (bar chart) ---
    ax6 = fig.add_subplot(gs[1, 2])
    component_names = []
    component_values = []
    final_idx = -1
    for name, key in [("Qwen NLL", "qwen_nll"), ("Audio Reg", "audio_reg"),
                       ("Text Reg", "text_reg"), ("Perceptual", "perceptual_loss")]:
        vals = data.get(key, [])
        if vals:
            final_val = vals[final_idx]
            if not np.isnan(final_val) and final_val > 0:
                component_names.append(name)
                component_values.append(final_val)
    if component_values:
        colors = ["#2196F3", "#FF9800", "#00BCD4", "#F44336"][:len(component_values)]
        bars = ax6.bar(component_names, component_values, color=colors)
        ax6.set_ylabel("Loss Value")
        ax6.set_title("Final Loss Component Magnitudes")
        ax6.set_yscale("log")
        for bar, val in zip(bars, component_values):
            ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.2e}", ha="center", va="bottom", fontsize=8)

    # --- 7. Adversarial regime detector ---
    ax7 = fig.add_subplot(gs[2, 0:2])
    qwen_nll = data.get("qwen_nll", [])
    grad_norms_clean = [g for g in grad_norms if not np.isnan(g)]
    if len(qwen_nll) >= 3 and len(grad_norms_clean) >= 3:
        # Compute rolling loss improvement rate
        window = 3
        improvement_rate = []
        for i in range(window, len(qwen_nll)):
            delta = qwen_nll[i - window] - qwen_nll[i]
            improvement_rate.append(delta / window)

        ax7.plot(iters[window:], improvement_rate, "b-", linewidth=1.5, label="Loss improvement rate")
        ax7.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="zero improvement")

        ax7b = ax7.twinx()
        ax7b.plot(iters[:len(grad_norms)], grad_norms, "purple", alpha=0.4, linewidth=1, label="grad_norm")
        ax7b.set_ylabel("Gradient Norm", color="purple")

        ax7.set_xlabel("Iteration")
        ax7.set_ylabel("Loss Improvement Rate (3-iter window)")
        ax7.set_title("Adversarial Regime Detector: declining improvement + high gradients = adversarial drift")
        ax7.legend(loc="upper left")
        ax7b.legend(loc="upper right")
    else:
        ax7.text(0.5, 0.5, "Insufficient data for regime detection",
                 transform=ax7.transAxes, ha="center")

    # --- 8. Config summary ---
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis("off")
    args_dict = config.get("args", {})
    config_text = (
        f"Edit: {args_dict.get('edit_prompt', 'N/A')}\n"
        f"Mode: {args_dict.get('opt_mode', 'N/A')}\n"
        f"LR: {args_dict.get('lr', 'N/A')}\n"
        f"Iterations: {args_dict.get('iterations', 'N/A')}\n"
        f"Latent Reg: {args_dict.get('latent_reg_weight', 'N/A')}\n"
        f"Text Reg: {args_dict.get('text_reg_weight', 'N/A')}\n"
        f"LPIPS Weight: {args_dict.get('lpips_weight', 0.0)}\n"
        f"Temporal Weight: {args_dict.get('temporal_weight', 0.0)}\n"
        f"LR Schedule: {args_dict.get('lr_schedule', 'constant')}\n"
        f"Reg Schedule: {args_dict.get('reg_schedule', 'constant')}\n"
        f"Qwen Frames: {args_dict.get('qwen_max_frames', 'N/A')}\n"
        f"Grad Clip: {args_dict.get('grad_clip', 'N/A')}\n"
    )
    ax8.text(0.05, 0.95, config_text, transform=ax8.transAxes,
             fontsize=9, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax8.set_title("Configuration")

    fig.suptitle(f"Optimization Dashboard: {label}", fontsize=15, fontweight="bold")

    out = output_path or (experiment_dir / "optimization_dashboard.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved dashboard: {out}")


# ---------------------------------------------------------------------------
# Plot 2: Cross-experiment comparison
# ---------------------------------------------------------------------------

def plot_comparison(
    experiment_dirs: list[Path],
    labels: list[str] | None = None,
    output_path: Path | None = None,
) -> None:
    """Compare multiple experiments side-by-side."""
    experiments = []
    for i, exp_dir in enumerate(experiment_dirs):
        csv_path = _find_csv(exp_dir)
        if csv_path is None:
            print(f"Skipping {exp_dir}: no CSV found", file=sys.stderr)
            continue
        data = _load_csv(csv_path)
        config = _load_config(exp_dir)
        label = labels[i] if labels and i < len(labels) else _get_label(exp_dir, config)
        experiments.append({"dir": exp_dir, "data": data, "config": config, "label": label})

    if len(experiments) < 2:
        print("Need at least 2 experiments to compare", file=sys.stderr)
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    cmap = plt.cm.tab10
    colors = [cmap(i) for i in range(len(experiments))]

    # --- 1. Qwen yes_prob comparison ---
    ax = axes[0, 0]
    for exp, c in zip(experiments, colors):
        iters = exp["data"].get("iter", [])
        probs = exp["data"].get("qwen_yes_prob", [])
        ax.plot(iters, probs, color=c, linewidth=1.5, label=exp["label"])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("P(yes)")
    ax.set_title("Qwen yes_prob Comparison")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)

    # --- 2. Total loss comparison ---
    ax = axes[0, 1]
    for exp, c in zip(experiments, colors):
        iters = exp["data"].get("iter", [])
        total = exp["data"].get("total_loss", [])
        ax.plot(iters, total, color=c, linewidth=1.5, label=exp["label"])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss Comparison")
    ax.legend(fontsize=7)

    # --- 3. Gradient norm comparison ---
    ax = axes[0, 2]
    for exp, c in zip(experiments, colors):
        iters = exp["data"].get("iter", [])
        gnorms = exp["data"].get("grad_norm", [])
        if gnorms:
            ax.plot(iters, gnorms, color=c, linewidth=1, alpha=0.7, label=exp["label"])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm Comparison")
    ax.legend(fontsize=7)

    # --- 4. Motion yes_prob comparison ---
    ax = axes[1, 0]
    has_motion = False
    for exp, c in zip(experiments, colors):
        iters = exp["data"].get("iter", [])
        motion = exp["data"].get("motion_yes_prob", [])
        if motion:
            valid = [(it, v) for it, v in zip(iters, motion) if not np.isnan(v)]
            if valid:
                ax.plot([x[0] for x in valid], [x[1] for x in valid],
                        color=c, linewidth=1.5, label=exp["label"])
                has_motion = True
    if has_motion:
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Motion P(yes)")
        ax.set_title("Motion Rubric Comparison")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No motion rubric data", transform=ax.transAxes, ha="center")

    # --- 5. LPIPS comparison (if available) ---
    ax = axes[1, 1]
    has_lpips = False
    for exp, c in zip(experiments, colors):
        iters = exp["data"].get("iter", [])
        lpips = exp["data"].get("lpips_raw", [])
        if lpips:
            valid = [(it, v) for it, v in zip(iters, lpips) if not np.isnan(v) and v > 0]
            if valid:
                ax.plot([x[0] for x in valid], [x[1] for x in valid],
                        color=c, linewidth=1.5, label=exp["label"])
                has_lpips = True
    if has_lpips:
        ax.set_xlabel("Iteration")
        ax.set_ylabel("LPIPS Distance")
        ax.set_title("Source Preservation (LPIPS)")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No LPIPS data\n(run with --lpips-weight > 0)",
                transform=ax.transAxes, ha="center", color="gray")

    # --- 6. Summary bar chart: final best yes_prob ---
    ax = axes[1, 2]
    final_probs = []
    exp_labels = []
    for exp in experiments:
        probs = exp["data"].get("qwen_yes_prob", [])
        if probs:
            best_prob = max(probs)
        else:
            best_prob = 0
        final_probs.append(best_prob)
        exp_labels.append(exp["label"][:20])

    bars = ax.barh(range(len(experiments)), final_probs, color=colors)
    ax.set_yticks(range(len(experiments)))
    ax.set_yticklabels(exp_labels, fontsize=8)
    ax.set_xlabel("Best P(yes)")
    ax.set_title("Best Achieved yes_prob")
    ax.set_xlim(0, 1)
    for i, (bar, val) in enumerate(zip(bars, final_probs)):
        ax.text(val + 0.01, i, f"{val:.3f}", va="center", fontsize=8)

    fig.suptitle("Cross-Experiment Comparison", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = output_path or Path("comparison_dashboard.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison: {out}")


# ---------------------------------------------------------------------------
# Text-based analysis (no matplotlib needed)
# ---------------------------------------------------------------------------

def print_experiment_summary(experiment_dir: Path) -> dict:
    """Print a text summary of one experiment. Returns summary dict."""
    csv_path = _find_csv(experiment_dir)
    if csv_path is None:
        return {}
    data = _load_csv(csv_path)
    config = _load_config(experiment_dir)
    label = _get_label(experiment_dir, config)
    args_dict = config.get("args", {})

    iters = data.get("iter", [])
    if not iters:
        return {}

    qwen_nll = data.get("qwen_nll", [])
    yes_prob = data.get("qwen_yes_prob", [])
    grad_norms = data.get("grad_norm", [])
    motion_prob = [v for v in data.get("motion_yes_prob", []) if not np.isnan(v)]
    entities_prob = [v for v in data.get("entities_yes_prob", []) if not np.isnan(v)]

    best_yes = max(yes_prob) if yes_prob else 0
    initial_yes = yes_prob[0] if yes_prob else 0
    best_iter_idx = yes_prob.index(best_yes) if yes_prob else 0
    mean_grad = np.nanmean(grad_norms) if grad_norms else 0
    max_grad = np.nanmax(grad_norms) if grad_norms else 0

    summary = {
        "label": label,
        "dir": str(experiment_dir),
        "initial_yes_prob": initial_yes,
        "best_yes_prob": best_yes,
        "best_iter": int(iters[best_iter_idx]) if iters else 0,
        "total_iters": int(iters[-1]) if iters else 0,
        "initial_nll": qwen_nll[0] if qwen_nll else 0,
        "best_nll": min(qwen_nll) if qwen_nll else 0,
        "mean_grad_norm": float(mean_grad),
        "max_grad_norm": float(max_grad),
        "best_motion_prob": max(motion_prob) if motion_prob else None,
        "best_entities_prob": max(entities_prob) if entities_prob else None,
        "lr": args_dict.get("lr", "?"),
        "latent_reg": args_dict.get("latent_reg_weight", "?"),
        "text_reg": args_dict.get("text_reg_weight", "?"),
    }
    return summary


def print_comparison_table(results_root: Path) -> None:
    """Print a text comparison table for all experiments."""
    experiment_dirs = []
    for csv_path in sorted(results_root.rglob("optimization_log_qwen_*.csv")):
        exp_dir = csv_path.parent
        if exp_dir.name.startswith("mode_"):
            exp_dir = exp_dir.parent
        experiment_dirs.append(exp_dir)

    seen = set()
    unique_dirs = []
    for d in experiment_dirs:
        if str(d) not in seen:
            seen.add(str(d))
            unique_dirs.append(d)

    if not unique_dirs:
        print(f"No experiments found under {results_root}")
        return

    summaries = []
    for d in unique_dirs:
        s = print_experiment_summary(d)
        if s:
            summaries.append(s)

    # Sort by best_yes_prob descending
    summaries.sort(key=lambda x: x["best_yes_prob"], reverse=True)

    print("\n" + "=" * 120)
    print("OPTIMIZATION COMPARISON — ALL EXPERIMENTS")
    print("=" * 120)
    print(f"{'Edit Prompt':<40s} {'Init P(y)':>9s} {'Best P(y)':>9s} {'Best It':>7s} "
          f"{'Mean Grad':>9s} {'Max Grad':>9s} {'Motion':>8s} {'Entities':>8s} {'LR':>6s}")
    print("-" * 120)

    for s in summaries:
        motion_str = f"{s['best_motion_prob']:.3f}" if s["best_motion_prob"] is not None else "N/A"
        entities_str = f"{s['best_entities_prob']:.3f}" if s["best_entities_prob"] is not None else "N/A"
        print(f"{s['label']:<40s} {s['initial_yes_prob']:>9.4f} {s['best_yes_prob']:>9.4f} "
              f"{s['best_iter']:>7d} {s['mean_grad_norm']:>9.3f} {s['max_grad_norm']:>9.3f} "
              f"{motion_str:>8s} {entities_str:>8s} {s['lr']:>6}")

    print()
    print("KEY INSIGHTS:")
    print("-" * 80)

    # Identify adversarial regime experiments
    high_grad = [s for s in summaries if s["mean_grad_norm"] > 1.0]
    low_initial = [s for s in summaries if s["initial_yes_prob"] < 0.1]
    easy_edits = [s for s in summaries if s["initial_yes_prob"] > 0.5]

    if easy_edits:
        print(f"\n  EASY EDITS (initial P(yes) > 0.5): {len(easy_edits)} experiments")
        for s in easy_edits:
            print(f"    - {s['label']}: init={s['initial_yes_prob']:.3f} → best={s['best_yes_prob']:.3f}")

    if low_initial:
        print(f"\n  HARD EDITS (initial P(yes) < 0.1): {len(low_initial)} experiments")
        print("  These are at risk of adversarial optimization (artifacts that fool Qwen).")
        for s in low_initial:
            print(f"    - {s['label']}: init={s['initial_yes_prob']:.3f} → best={s['best_yes_prob']:.3f}, "
                  f"max_grad={s['max_grad_norm']:.2f}")

    if high_grad:
        print(f"\n  HIGH GRADIENT REGIME (mean grad > 1.0): {len(high_grad)} experiments")
        print("  Consider enabling: --lpips-weight 0.1 --temporal-weight 0.05 --lr-schedule cosine --reg-schedule cosine_increase")
        for s in high_grad:
            print(f"    - {s['label']}: mean_grad={s['mean_grad_norm']:.3f}")

    print("\n  RECOMMENDATIONS:")
    print("  For hard edits (low initial P(yes)), use these anti-adversarial settings:")
    print("    --lpips-weight 0.1 --temporal-weight 0.05 --lr-schedule cosine --reg-schedule cosine_increase")
    print("  For easy edits (high initial P(yes)), default settings should work fine.")
    print()


# ---------------------------------------------------------------------------
# Auto-discover and compare all experiments
# ---------------------------------------------------------------------------

def compare_all(results_root: Path, output_path: Path | None = None) -> None:
    """Find all experiments under results_root and generate comparison plots."""
    # Always print text analysis
    print_comparison_table(results_root)

    experiment_dirs = []
    for csv_path in sorted(results_root.rglob("optimization_log_qwen_*.csv")):
        # Go up to the mode_* or experiment directory
        exp_dir = csv_path.parent
        if exp_dir.name.startswith("mode_"):
            exp_dir = exp_dir.parent
        experiment_dirs.append(exp_dir)

    # Deduplicate
    seen = set()
    unique_dirs = []
    for d in experiment_dirs:
        if str(d) not in seen:
            seen.add(str(d))
            unique_dirs.append(d)

    if not unique_dirs:
        print(f"No experiments found under {results_root}", file=sys.stderr)
        return

    if not _HAS_MPL:
        print("Skipping plot generation (matplotlib not available).")
        return

    print(f"\nGenerating dashboards for {len(unique_dirs)} experiments...")

    # Generate individual dashboards
    for exp_dir in unique_dirs:
        plot_experiment_dashboard(exp_dir)

    # Generate comparison
    out = output_path or (results_root / "all_experiments_comparison.png")
    plot_comparison(unique_dirs, output_path=out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path,
                       help="Experiment directories or a single results root with --compare-all")
    parser.add_argument("--compare-all", action="store_true",
                       help="Recursively find all experiments under the given path and compare them")
    parser.add_argument("--labels", nargs="*",
                       help="Custom labels for comparison (one per experiment)")
    parser.add_argument("--output", type=Path, default=None,
                       help="Output path for the plot")
    args = parser.parse_args()

    if args.compare_all:
        compare_all(args.paths[0], args.output)
    elif len(args.paths) == 1:
        plot_experiment_dashboard(args.paths[0], args.output)
    else:
        plot_comparison(args.paths, labels=args.labels, output_path=args.output)


if __name__ == "__main__":
    main()
