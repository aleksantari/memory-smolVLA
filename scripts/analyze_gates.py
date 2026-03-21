"""Analyze and visualize gate activation timelines from eval results.

Reads the gate_alpha_timeline from an eval results JSON file and
produces matplotlib plots showing how memory influence evolves over
episode timesteps.

Usage:
    python scripts/analyze_gates.py --results eval_results/results.json
    python scripts/analyze_gates.py --results eval_results/results.json --output plots/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot gate activation timelines")
    parser.add_argument("--results", required=True, help="Path to results.json from eval.py")
    parser.add_argument("--output", default="plots", help="Output directory for figures")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib and numpy required: pip install matplotlib numpy")
        sys.exit(1)

    with open(args.results) as f:
        results = json.load(f)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_names = [k for k in results if k != "aggregate"]

    for task in task_names:
        task_data = results[task]
        timelines = task_data.get("gate_alpha_timeline", [])
        if not timelines:
            continue

        # Pad to same length for averaging
        max_len = max(len(ep) for ep in timelines)
        padded = np.full((len(timelines), max_len), np.nan)
        for i, ep in enumerate(timelines):
            padded[i, : len(ep)] = ep

        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        timesteps = np.arange(max_len)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(timesteps, mean, label="mean gate α", color="steelblue")
        ax.fill_between(
            timesteps,
            mean - std,
            mean + std,
            alpha=0.3,
            color="steelblue",
            label="±1 std",
        )
        ax.set_xlabel("Timestep within episode")
        ax.set_ylabel("Gate α (memory influence)")
        ax.set_title(f"Gate activation — {task}")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(alpha=0.3)

        safe_name = task.replace("/", "_").replace(" ", "_")[:60]
        out_path = out_dir / f"gate_{safe_name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    # Summary plot: mean α per task
    fig, ax = plt.subplots(figsize=(max(6, len(task_names) * 1.2), 4))
    means = [results[t].get("avg_gate_alpha_mean", 0) for t in task_names]
    ax.bar(range(len(task_names)), means, color="steelblue")
    ax.set_xticks(range(len(task_names)))
    ax.set_xticklabels(
        [t[-30:] for t in task_names], rotation=45, ha="right", fontsize=8
    )
    ax.set_ylabel("Mean gate α")
    ax.set_title("Average memory influence per task")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "gate_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary: {out_path}")


if __name__ == "__main__":
    main()
