from __future__ import annotations

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnNoModelImprovement

from bipedal_walker_common import (
    BEST_MODEL_DIR,
    BEST_NORMALIZE_PATH,
    FINAL_MODEL_PATH,
    FINAL_NORMALIZE_PATH,
    LOG_DIR,
    RewardConfig,
    make_eval_vec_env,
    make_train_vec_env,
    prepare_output_dirs,
)


class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_path, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        training_env = self.parent.training_env if self.parent is not None else self.training_env
        if training_env is not None:
            training_env.save(self.save_path)
            if self.verbose > 0:
                print(f"VecNormalize を保存しました: {self.save_path}")
        return True


def train(total_timesteps: int) -> None:
    prepare_output_dirs()

    reward_config = RewardConfig(
        fall_penalty=40.0,
        torque_penalty_coef=0.003,
        alive_bonus=0.5,
        forward_reward_coef=8.0,
    )

    train_env = make_train_vec_env(reward_config)
    eval_env = make_eval_vec_env(reward_config)

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=20,
        min_evals=10,
        verbose=1,
    )

    save_vecnormalize_callback = SaveVecNormalizeCallback(
        save_path=BEST_NORMALIZE_PATH,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        callback_on_new_best=save_vecnormalize_callback,
        callback_after_eval=stop_callback,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    model.save(FINAL_MODEL_PATH)
    train_env.save(FINAL_NORMALIZE_PATH)

    train_env.close()
    eval_env.close()

    print("学習完了")
    print(f"最終モデル: {FINAL_MODEL_PATH.with_suffix('.zip')}")
    print(f"最終正規化情報: {FINAL_NORMALIZE_PATH}")
    print(f"最良モデル: {BEST_MODEL_DIR / 'best_model.zip'}")
    print(f"最良モデル用正規化情報: {BEST_NORMALIZE_PATH}")
    print(f"評価ログ: {LOG_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on BipedalWalker-v3")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args.timesteps)


if __name__ == "__main__":
    main()