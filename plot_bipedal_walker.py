from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from bipedal_walker_common import LOG_DIR


def plot_eval_log(log_path: Path, show: bool) -> None:
    eval_log_path = log_path / "evaluations.npz"
    if not eval_log_path.exists():
        raise FileNotFoundError(f"評価ログが見つかりません: {eval_log_path}")

    data = np.load(eval_log_path)
    timesteps = data["timesteps"]
    results = data["results"]
    ep_lengths = data["ep_lengths"]

    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)
    mean_lengths = ep_lengths.mean(axis=1)

    best_index = int(np.argmax(mean_rewards))
    best_timestep = timesteps[best_index]
    best_reward = mean_rewards[best_index]

    k = min(10, len(timesteps))
    if k >= 2:
        recent_x = timesteps[-k:]
        recent_y = mean_rewards[-k:]
        x_scaled = recent_x - recent_x[0]
        slope, intercept = np.polyfit(x_scaled, recent_y, 1)
        regression_y = slope * x_scaled + intercept
        improvement_per_10000_steps = slope * 10_000
    else:
        recent_x = None
        regression_y = None
        improvement_per_10000_steps = None

    reward_fig = plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_rewards, label="Mean evaluation reward")
    plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2, label="±1 std")
    plt.scatter(best_timestep, best_reward, label=f"Best: {best_reward:.2f}", zorder=5)

    if recent_x is not None:
        plt.plot(recent_x, regression_y, linestyle="--", label=f"Recent trend: {improvement_per_10000_steps:.2f} / 10k steps")

    plt.xlabel("Timesteps")
    plt.ylabel("Mean episode reward")
    plt.title("PPO Training Performance on BipedalWalker-v3")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    reward_plot_path = log_path / "training_rewards.png"
    reward_fig.savefig(reward_plot_path, dpi=150)

    length_fig = plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_lengths, label="Mean episode length")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean episode length")
    plt.title("Episode Length During PPO Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    length_plot_path = log_path / "episode_lengths.png"
    length_fig.savefig(length_plot_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close("all")

    print(f"最高平均報酬: {best_reward:.2f}")
    print(f"最高報酬を記録したステップ数: {best_timestep}")
    if improvement_per_10000_steps is not None:
        print(f"直近{len(recent_x)}回の1万ステップあたり改善量: {improvement_per_10000_steps:.3f}")
    print(f"保存先: {reward_plot_path}")
    print(f"保存先: {length_plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot BipedalWalker PPO evaluation logs")
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_eval_log(args.log_dir, args.show)


if __name__ == "__main__":
    main()