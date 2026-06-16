from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder

from bipedal_walker_common import ALGORITHM_CLASSES, BEST_NORMALIZE_PATH, RewardConfig, make_env


def resolve_model_source(model_path: Path, normalize_path: Path) -> tuple[str, Path, Path]:
    metadata_path = model_path.parent / "best_trial.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        algorithm = metadata["algorithm"]
        resolved_model_path = Path(metadata["model_path"])
        resolved_normalize_path = Path(metadata["normalize_path"])
        return algorithm, resolved_model_path, resolved_normalize_path

    return "ppo", model_path, normalize_path


def record_video(model_path: Path, normalize_path: Path, video_folder: Path, video_length: int) -> None:
    reward_config = RewardConfig(
        fall_penalty=40.0,
        torque_penalty_coef=0.003,
        alive_bonus=0.0,
        forward_reward_coef=3.0,
    )

    algorithm, resolved_model_path, resolved_normalize_path = resolve_model_source(model_path, normalize_path)

    env = DummyVecEnv([lambda: make_env(reward_config, render_mode="rgb_array")])
    env = VecNormalize.load(resolved_normalize_path, env)
    env.training = False
    env.norm_reward = False

    video_folder.mkdir(parents=True, exist_ok=True)
    venv = VecVideoRecorder(
        env,
        str(video_folder),
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix=f"bipedal-walker-{algorithm}",
    )

    model_class = ALGORITHM_CLASSES[algorithm]
    model = model_class.load(resolved_model_path, env=venv)
    obs = venv.reset()

    total_reward = 0.0
    total_original_reward = 0.0
    total_torque_penalty = 0.0
    step_count = 0

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, infos = venv.step(action)

        info = infos[0]
        total_reward += float(reward[0])
        total_original_reward += float(info["original_reward"])
        total_torque_penalty += float(info["torque_penalty"])
        step_count += 1

        if done[0]:
            break

    venv.close()

    print(f"ステップ数: {step_count}")
    print(f"累積カスタム報酬: {total_reward:.2f}")
    print(f"累積元報酬: {total_original_reward:.2f}")
    print(f"累積トルク罰則: {total_torque_penalty:.2f}")
    print(f"使用アルゴリズム: {algorithm}")
    print(f"動画保存先: {video_folder}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a BipedalWalker PPO rollout")
    parser.add_argument("--model-path", type=Path, default=Path("ppo_bipedalwalker_result/best_model/best_model.zip"))
    parser.add_argument("--normalize-path", type=Path, default=BEST_NORMALIZE_PATH)
    parser.add_argument("--video-folder", type=Path, default=Path("bipedal_walker_videos_practice"))
    parser.add_argument("--video-length", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record_video(args.model_path, args.normalize_path, args.video_folder, args.video_length)


if __name__ == "__main__":
    main()