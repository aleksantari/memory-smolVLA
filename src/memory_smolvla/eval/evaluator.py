"""Evaluation loop for MemorySmolVLAPolicy on LIBERO tasks.

Loads a checkpoint, runs rollouts via LeRobot environment wrappers,
and records per-task success rates, gate activation timelines, and
memory bank utilization statistics.

Results are written to a JSON file and optionally logged to wandb.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from memory_smolvla.policy.memory_smolvla import MemorySmolVLAPolicy

logger = logging.getLogger(__name__)


class MemorySmolVLAEvaluator:
    """Evaluates a checkpoint on LIBERO rollouts.

    Args:
        policy: A loaded :class:`MemorySmolVLAPolicy` in eval mode.
        env_factory: Callable that returns a LeRobot-compatible gym
            environment given a task name string.
        output_dir: Directory where ``results.json`` is written.
        n_rollouts_per_task: Number of evaluation episodes per task.
        max_steps_per_episode: Hard episode length cap.
        wandb_project: Optional wandb project for logging results.
    """

    def __init__(
        self,
        policy: MemorySmolVLAPolicy,
        env_factory,
        output_dir: str = "eval_results",
        n_rollouts_per_task: int = 10,
        max_steps_per_episode: int = 500,
        wandb_project: str | None = None,
    ) -> None:
        self.policy = policy
        self.env_factory = env_factory
        self.output_dir = Path(output_dir)
        self.n_rollouts = n_rollouts_per_task
        self.max_steps = max_steps_per_episode
        self._wandb_project = wandb_project
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, task_names: list[str]) -> dict:
        """Run rollouts on each task and return aggregated results.

        Args:
            task_names: List of LIBERO task name strings to evaluate.

        Returns:
            Dict with per-task success rates and aggregate stats.
        """
        self.policy.eval()
        results = {}

        for task in task_names:
            logger.info("Evaluating task: %s (%d rollouts)", task, self.n_rollouts)
            task_results = self._evaluate_task(task)
            results[task] = task_results
            logger.info(
                "  success_rate=%.2f  avg_gate_alpha=%.4f",
                task_results["success_rate"],
                task_results["avg_gate_alpha_mean"],
            )

        results["aggregate"] = {
            "mean_success_rate": sum(
                r["success_rate"] for r in results.values() if "success_rate" in r
            ) / max(len(task_names), 1),
        }

        out_path = self.output_dir / "results.json"
        out_path.write_text(json.dumps(results, indent=2))
        logger.info("Results written to %s", out_path)

        if self._wandb_project:
            self._log_wandb(results)

        return results

    # ------------------------------------------------------------------
    # Per-task rollout
    # ------------------------------------------------------------------

    def _evaluate_task(self, task_name: str) -> dict:
        env = self.env_factory(task_name)
        successes = []
        gate_alpha_timeline: list[list[float]] = []

        for _ in range(self.n_rollouts):
            obs, _ = env.reset()
            self.policy.reset()
            ep_gate_alphas = []
            done = False
            step = 0

            while not done and step < self.max_steps:
                batch = self._obs_to_batch(obs)
                with torch.no_grad():
                    action = self.policy.select_action(batch)

                obs, _reward, terminated, truncated, info = env.step(
                    action.squeeze(0).cpu().numpy()
                )
                done = terminated or truncated
                step += 1

                gate_stats = self.policy.get_gate_statistics()
                if gate_stats:
                    ep_gate_alphas.append(gate_stats.get("gate_alpha_mean", 0.0))

            successes.append(float(info.get("success", False)))
            gate_alpha_timeline.append(ep_gate_alphas)

        env.close()
        return {
            "success_rate": sum(successes) / len(successes),
            "n_rollouts": self.n_rollouts,
            "avg_gate_alpha_mean": (
                sum(a for ep in gate_alpha_timeline for a in ep)
                / max(sum(len(ep) for ep in gate_alpha_timeline), 1)
            ),
            "gate_alpha_timeline": gate_alpha_timeline,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _obs_to_batch(self, obs) -> dict:
        """Convert a gym observation to a single-item batch dict."""
        if isinstance(obs, dict):
            return {
                k: torch.as_tensor(v, dtype=torch.float32).unsqueeze(0)
                for k, v in obs.items()
            }
        return {"observation.state": torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)}

    def _log_wandb(self, results: dict) -> None:
        try:
            import wandb
            wandb.log(
                {
                    f"eval/{task}/success_rate": v["success_rate"]
                    for task, v in results.items()
                    if "success_rate" in v
                }
            )
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        policy: MemorySmolVLAPolicy,
        **kwargs,
    ) -> "MemorySmolVLAEvaluator":
        """Load policy weights from a checkpoint and return an evaluator.

        Args:
            checkpoint_path: Path to a ``.pt`` checkpoint saved by
                :class:`MemorySmolVLATrainer`.
            policy: A :class:`MemorySmolVLAPolicy` instance with the
                same architecture as was used during training.
            **kwargs: Additional arguments forwarded to ``__init__``.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        policy.load_state_dict(ckpt["policy_state_dict"])
        logger.info(
            "Loaded checkpoint from %s (step=%d)", checkpoint_path, ckpt.get("step", -1)
        )
        return cls(policy=policy, **kwargs)
