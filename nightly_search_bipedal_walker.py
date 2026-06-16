from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from bipedal_walker_common import (
    ALGORITHM_CLASSES,
    BEST_MODEL_DIR,
    BEST_NORMALIZE_PATH,
    LOG_DIR,
    PLOT_DIR,
    RewardConfig,
    make_eval_vec_env,
    make_train_vec_env,
    prepare_output_dirs,
)


@dataclass(frozen=True)
class TrialSpec:
    name: str
    algorithm: str
    total_timesteps: int
    chunk_timesteps: int
    eval_episodes: int
    patience_evals: int
    min_improvement: float
    model_kwargs: dict[str, Any]


@dataclass
class EvaluationResult:
    mean_reward: float
    std_reward: float
    mean_final_x: float
    mean_max_x: float
    mean_episode_length: float


@dataclass
class EvaluationPoint:
    step: int
    metrics: EvaluationResult


def evaluate_model(model, eval_env, n_episodes: int) -> EvaluationResult:
    episode_rewards: list[float] = []
    episode_final_xs: list[float] = []
    episode_max_xs: list[float] = []
    episode_lengths: list[int] = []

    for _ in range(n_episodes):
        obs = eval_env.reset()
        done = [False]
        total_reward = 0.0
        final_x = float("nan")
        max_x = float("-inf")
        step_count = 0

        while not done[0]:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, infos = eval_env.step(action)
            info = infos[0]
            total_reward += float(reward[0])
            final_x = float(info.get("x_after", final_x))
            max_x = max(max_x, float(info.get("episode_max_x", max_x)))
            step_count += 1

        episode_rewards.append(total_reward)
        episode_final_xs.append(final_x)
        episode_max_xs.append(max_x)
        episode_lengths.append(step_count)

    return EvaluationResult(
        mean_reward=float(np.mean(episode_rewards)),
        std_reward=float(np.std(episode_rewards)),
        mean_final_x=float(np.mean(episode_final_xs)),
        mean_max_x=float(np.mean(episode_max_xs)),
        mean_episode_length=float(np.mean(episode_lengths)),
    )


def build_trial_specs(total_timesteps: int, chunk_timesteps: int, eval_episodes: int) -> list[TrialSpec]:
    return [
        TrialSpec(
            name="ppo_baseline",
            algorithm="ppo",
            total_timesteps=total_timesteps,
            chunk_timesteps=chunk_timesteps,
            eval_episodes=eval_episodes,
            patience_evals=3,
            min_improvement=2.0,
            model_kwargs={
                "policy": "MlpPolicy",
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "verbose": 1,
            },
        ),
        TrialSpec(
            name="sac_balanced",
            algorithm="sac",
            total_timesteps=total_timesteps,
            chunk_timesteps=chunk_timesteps,
            eval_episodes=eval_episodes,
            patience_evals=3,
            min_improvement=2.0,
            model_kwargs={
                "policy": "MlpPolicy",
                "learning_rate": 3e-4,
                "buffer_size": 300_000,
                "learning_starts": 10_000,
                "batch_size": 256,
                "tau": 0.02,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1,
                "ent_coef": "auto",
                "verbose": 1,
            },
        ),
        TrialSpec(
            name="ddpg_conservative",
            algorithm="ddpg",
            total_timesteps=total_timesteps,
            chunk_timesteps=chunk_timesteps,
            eval_episodes=eval_episodes,
            patience_evals=3,
            min_improvement=2.0,
            model_kwargs={
                "policy": "MlpPolicy",
                "learning_rate": 1e-3,
                "buffer_size": 300_000,
                "learning_starts": 10_000,
                "batch_size": 256,
                "tau": 0.02,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1,
                "verbose": 1,
            },
        ),
    ]


def build_model(algorithm: str, env, model_kwargs: dict[str, Any]):
    model_class = ALGORITHM_CLASSES[algorithm]
    if algorithm in {"sac", "ddpg"}:
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions),
        )
        model_kwargs = dict(model_kwargs)
        model_kwargs["action_noise"] = action_noise

    return model_class(env=env, **model_kwargs)


def save_best_metadata(metadata_path: Path, trial_name: str, algorithm: str, model_path: Path, normalize_path: Path, metrics: EvaluationResult) -> None:
    metadata = {
        "trial_name": trial_name,
        "algorithm": algorithm,
        "model_path": str(model_path),
        "normalize_path": str(normalize_path),
        "metrics": asdict(metrics),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def build_run_plot_dir() -> Path:
    run_plot_dir = PLOT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_plot_dir.mkdir(parents=True, exist_ok=True)
    return run_plot_dir


def build_plot_stem(trial_spec: TrialSpec) -> str:
    return (
        f"{trial_spec.name}"
        f"_algo-{trial_spec.algorithm}"
        f"_t{trial_spec.total_timesteps}"
        f"_chunk{trial_spec.chunk_timesteps}"
        f"_eval{trial_spec.eval_episodes}"
    )


def build_algorithm_plot_stem(algorithm: str, total_timesteps: int, chunk_timesteps: int, eval_episodes: int) -> str:
    return f"algorithm_{algorithm}_t{total_timesteps}_chunk{chunk_timesteps}_eval{eval_episodes}"


def build_run_plot_stem(total_timesteps: int, chunk_timesteps: int, eval_episodes: int) -> str:
    return f"run_all_algorithms_t{total_timesteps}_chunk{chunk_timesteps}_eval{eval_episodes}"


def save_trial_history(run_plot_dir: Path, trial_spec: TrialSpec, history: list[EvaluationPoint]) -> tuple[Path, Path]:
    stem = build_plot_stem(trial_spec)
    json_path = run_plot_dir / f"{stem}.json"
    png_path = run_plot_dir / f"{stem}.png"

    payload = {
        "trial": asdict(trial_spec),
        "history": [
            {
                "step": point.step,
                "metrics": asdict(point.metrics),
            }
            for point in history
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path, png_path


def render_history_plot(
    title: str,
    output_path: Path,
    series_items: list[tuple[str, list[EvaluationPoint]]],
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title)

    for label, history in series_items:
        if not history:
            continue

        steps = [point.step for point in history]
        mean_rewards = [point.metrics.mean_reward for point in history]
        mean_max_xs = [point.metrics.mean_max_x for point in history]
        mean_final_xs = [point.metrics.mean_final_x for point in history]
        mean_episode_lengths = [point.metrics.mean_episode_length for point in history]

        axes[0, 0].plot(steps, mean_rewards, marker="o", label=label)
        axes[0, 1].plot(steps, mean_max_xs, marker="o", label=label)
        axes[1, 0].plot(steps, mean_final_xs, marker="o", label=label)
        axes[1, 1].plot(steps, mean_episode_lengths, marker="o", label=label)

    axes[0, 0].set_title("Mean Reward")
    axes[0, 0].set_xlabel("Timesteps")
    axes[0, 0].grid(True)

    axes[0, 1].set_title("Mean Max X")
    axes[0, 1].set_xlabel("Timesteps")
    axes[0, 1].grid(True)

    axes[1, 0].set_title("Mean Final X")
    axes[1, 0].set_xlabel("Timesteps")
    axes[1, 0].grid(True)

    axes[1, 1].set_title("Mean Episode Length")
    axes[1, 1].set_xlabel("Timesteps")
    axes[1, 1].grid(True)

    for axis in axes.flat:
        if axis.lines:
            axis.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_trial_history(run_plot_dir: Path, trial_spec: TrialSpec, history: list[EvaluationPoint]) -> Path:
    _, png_path = save_trial_history(run_plot_dir, trial_spec, history)
    title = (
        f"{trial_spec.name} ({trial_spec.algorithm})\n"
        f"timesteps={trial_spec.total_timesteps}, chunk={trial_spec.chunk_timesteps}, eval={trial_spec.eval_episodes}"
    )
    return render_history_plot(title, png_path, [(trial_spec.name, history)])


def save_algorithm_history(
    run_plot_dir: Path,
    algorithm: str,
    trial_specs: list[TrialSpec],
    histories: dict[str, list[EvaluationPoint]],
    total_timesteps: int,
    chunk_timesteps: int,
    eval_episodes: int,
) -> tuple[Path, Path]:
    stem = build_algorithm_plot_stem(algorithm, total_timesteps, chunk_timesteps, eval_episodes)
    json_path = run_plot_dir / f"{stem}.json"
    png_path = run_plot_dir / f"{stem}.png"
    payload = {
        "algorithm": algorithm,
        "timesteps": total_timesteps,
        "chunk_timesteps": chunk_timesteps,
        "eval_episodes": eval_episodes,
        "trials": [
            {
                "trial": asdict(trial_spec),
                "history": [
                    {
                        "step": point.step,
                        "metrics": asdict(point.metrics),
                    }
                    for point in histories.get(trial_spec.name, [])
                ],
            }
            for trial_spec in trial_specs
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path, png_path


def plot_algorithm_history(
    run_plot_dir: Path,
    algorithm: str,
    trial_specs: list[TrialSpec],
    histories: dict[str, list[EvaluationPoint]],
    total_timesteps: int,
    chunk_timesteps: int,
    eval_episodes: int,
) -> Path:
    _, png_path = save_algorithm_history(
        run_plot_dir,
        algorithm,
        trial_specs,
        histories,
        total_timesteps,
        chunk_timesteps,
        eval_episodes,
    )
    title = f"Algorithm: {algorithm}\ntimesteps={total_timesteps}, chunk={chunk_timesteps}, eval={eval_episodes}"
    series_items = [(trial_spec.name, histories.get(trial_spec.name, [])) for trial_spec in trial_specs]
    return render_history_plot(title, png_path, series_items)


def save_run_history(
    run_plot_dir: Path,
    trial_specs: list[TrialSpec],
    histories: dict[str, list[EvaluationPoint]],
    total_timesteps: int,
    chunk_timesteps: int,
    eval_episodes: int,
) -> tuple[Path, Path]:
    stem = build_run_plot_stem(total_timesteps, chunk_timesteps, eval_episodes)
    json_path = run_plot_dir / f"{stem}.json"
    png_path = run_plot_dir / f"{stem}.png"
    payload = {
        "timesteps": total_timesteps,
        "chunk_timesteps": chunk_timesteps,
        "eval_episodes": eval_episodes,
        "trials": [
            {
                "trial": asdict(trial_spec),
                "history": [
                    {
                        "step": point.step,
                        "metrics": asdict(point.metrics),
                    }
                    for point in histories.get(trial_spec.name, [])
                ],
            }
            for trial_spec in trial_specs
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path, png_path


def plot_run_history(
    run_plot_dir: Path,
    trial_specs: list[TrialSpec],
    histories: dict[str, list[EvaluationPoint]],
    total_timesteps: int,
    chunk_timesteps: int,
    eval_episodes: int,
) -> Path:
    _, png_path = save_run_history(
        run_plot_dir,
        trial_specs,
        histories,
        total_timesteps,
        chunk_timesteps,
        eval_episodes,
    )
    title = f"Nightly Search Run\ntimesteps={total_timesteps}, chunk={chunk_timesteps}, eval={eval_episodes}"
    series_items = [
        (f"{trial_spec.algorithm}:{trial_spec.name}", histories.get(trial_spec.name, []))
        for trial_spec in trial_specs
    ]
    return render_history_plot(title, png_path, series_items)


def save_run_parameters(run_plot_dir: Path, total_timesteps: int, chunk_timesteps: int, eval_episodes: int) -> Path:
    params_path = run_plot_dir / f"run_params_t{total_timesteps}_chunk{chunk_timesteps}_eval{eval_episodes}.json"
    payload = {
        "timesteps": total_timesteps,
        "chunk_timesteps": chunk_timesteps,
        "eval_episodes": eval_episodes,
    }
    params_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return params_path


def search(total_timesteps: int, chunk_timesteps: int, eval_episodes: int) -> None:
    prepare_output_dirs()

    reward_config = RewardConfig(
        fall_penalty=40.0,
        torque_penalty_coef=0.003,
        alive_bonus=0.0,
        forward_reward_coef=3.0,
    )

    trial_specs = build_trial_specs(total_timesteps, chunk_timesteps, eval_episodes)
    run_plot_dir = build_run_plot_dir()
    run_params_path = save_run_parameters(run_plot_dir, total_timesteps, chunk_timesteps, eval_episodes)
    print(f"Plot directory: {run_plot_dir}")
    print(f"Run parameters: {run_params_path}")

    best_score = float("-inf")
    best_metadata_path = BEST_MODEL_DIR / "best_trial.json"
    global_best_summary: dict[str, Any] | None = None
    all_histories: dict[str, list[EvaluationPoint]] = {trial_spec.name: [] for trial_spec in trial_specs}

    for trial_index, trial_spec in enumerate(trial_specs, start=1):
        print(f"\n=== Trial {trial_index}/{len(trial_specs)}: {trial_spec.name} ({trial_spec.algorithm}) ===")

        trial_dir = LOG_DIR / trial_spec.name
        trial_dir.mkdir(parents=True, exist_ok=True)

        train_env = make_train_vec_env(reward_config)
        eval_env = make_eval_vec_env(reward_config)
        model = build_model(trial_spec.algorithm, train_env, trial_spec.model_kwargs)

        learned_timesteps = 0
        no_improve_count = 0
        trial_best_metrics: EvaluationResult | None = None
        trial_best_step = 0
        trial_best_model_path = trial_dir / "best_model.zip"
        trial_best_normalize_path = trial_dir / "vecnormalize.pkl"
        eval_history = all_histories[trial_spec.name]

        while learned_timesteps < trial_spec.total_timesteps:
            chunk = min(trial_spec.chunk_timesteps, trial_spec.total_timesteps - learned_timesteps)
            model.learn(total_timesteps=chunk, reset_num_timesteps=False)
            learned_timesteps += chunk

            eval_env.obs_rms = train_env.obs_rms
            eval_env.ret_rms = train_env.ret_rms

            metrics = evaluate_model(model, eval_env, trial_spec.eval_episodes)
            eval_history.append(EvaluationPoint(step=learned_timesteps, metrics=metrics))
            trial_plot_path = plot_trial_history(run_plot_dir, trial_spec, eval_history)
            algorithm_trial_specs = [spec for spec in trial_specs if spec.algorithm == trial_spec.algorithm]
            algorithm_plot_path = plot_algorithm_history(
                run_plot_dir,
                trial_spec.algorithm,
                algorithm_trial_specs,
                all_histories,
                total_timesteps,
                chunk_timesteps,
                eval_episodes,
            )
            run_plot_path = plot_run_history(
                run_plot_dir,
                trial_specs,
                all_histories,
                total_timesteps,
                chunk_timesteps,
                eval_episodes,
            )
            print(
                f"step={learned_timesteps} reward={metrics.mean_reward:.2f} "
                f"max_x={metrics.mean_max_x:.2f} final_x={metrics.mean_final_x:.2f}"
            )
            print(f"  trial plot updated: {trial_plot_path}")
            print(f"  algorithm plot updated: {algorithm_plot_path}")
            print(f"  run plot updated: {run_plot_path}")

            score = metrics.mean_max_x
            if trial_best_metrics is None or score > trial_best_metrics.mean_max_x + trial_spec.min_improvement:
                trial_best_metrics = metrics
                trial_best_step = learned_timesteps
                model.save(trial_best_model_path)
                train_env.save(trial_best_normalize_path)
                no_improve_count = 0
                print(f"  new trial best at step {trial_best_step}")
            else:
                no_improve_count += 1
                print(f"  no improvement count: {no_improve_count}/{trial_spec.patience_evals}")

            if trial_best_metrics is not None and trial_best_metrics.mean_max_x > best_score + trial_spec.min_improvement:
                best_score = trial_best_metrics.mean_max_x
                global_best_summary = {
                    "trial_name": trial_spec.name,
                    "algorithm": trial_spec.algorithm,
                    "step": trial_best_step,
                    "metrics": asdict(trial_best_metrics),
                }
                BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
                (BEST_MODEL_DIR / "best_model.zip").write_bytes(trial_best_model_path.read_bytes())
                (BEST_MODEL_DIR / "vecnormalize.pkl").write_bytes(trial_best_normalize_path.read_bytes())
                save_best_metadata(best_metadata_path, trial_spec.name, trial_spec.algorithm, BEST_MODEL_DIR / "best_model.zip", BEST_NORMALIZE_PATH, trial_best_metrics)
                print(f"  global best updated: {best_score:.2f}")

            if no_improve_count >= trial_spec.patience_evals:
                print("  stopping trial early because the right-edge score no longer improves.")
                break

        train_env.close()
        eval_env.close()

    if global_best_summary is None:
        raise RuntimeError("有効な best trial を見つけられませんでした。")

    print("\n=== Search complete ===")
    print(json.dumps(global_best_summary, ensure_ascii=False, indent=2))
    print(f"Best model: {BEST_MODEL_DIR / 'best_model.zip'}")
    print(f"Best normalize: {BEST_NORMALIZE_PATH}")
    print(f"Best metadata: {best_metadata_path}")
    print(f"Plots: {run_plot_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nightly BipedalWalker algorithm search")
    parser.add_argument("--timesteps", type=int, default=600_000)
    parser.add_argument("--chunk-timesteps", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search(args.timesteps, args.chunk_timesteps, args.eval_episodes)


if __name__ == "__main__":
    main()
