from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


ENV_ID = "BipedalWalker-v3"
SAVE_DIR = Path("ppo_bipedalwalker_result")
BEST_MODEL_DIR = SAVE_DIR / "best_model"
LOG_DIR = SAVE_DIR / "logs"
FINAL_MODEL_PATH = SAVE_DIR / "final_model"
FINAL_NORMALIZE_PATH = SAVE_DIR / "vecnormalize.pkl"
BEST_NORMALIZE_PATH = BEST_MODEL_DIR / "vecnormalize.pkl"


@dataclass(frozen=True)
class RewardConfig:
    fall_penalty: float = 40.0
    torque_penalty_coef: float = 0.003
    alive_bonus: float = 0.0
    forward_reward_coef: float = 3.0


class BipedalRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_config: RewardConfig):
        super().__init__(env)
        self.reward_config = reward_config

    def step(self, action):
        x_before = self.unwrapped.hull.position.x
        obs, reward, terminated, truncated, info = self.env.step(action)
        x_after = self.unwrapped.hull.position.x

        forward_progress = x_after - x_before
        forward_bonus = self.reward_config.forward_reward_coef * forward_progress
        torque_penalty = self.reward_config.torque_penalty_coef * np.sum(np.square(action))
        fall_penalty_value = self.reward_config.fall_penalty if terminated else 0.0

        custom_reward = reward
        custom_reward += self.reward_config.alive_bonus
        custom_reward += forward_bonus
        custom_reward -= torque_penalty
        custom_reward -= fall_penalty_value

        info["original_reward"] = reward
        info["custom_reward"] = custom_reward
        info["forward_progress"] = forward_progress
        info["forward_bonus"] = forward_bonus
        info["torque_penalty"] = torque_penalty
        info["fall_penalty"] = fall_penalty_value

        return obs, custom_reward, terminated, truncated, info


def prepare_output_dirs() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def make_env(reward_config: RewardConfig, render_mode: str | None = None) -> gym.Env:
    env = gym.make(ENV_ID, render_mode=render_mode)
    return BipedalRewardWrapper(env, reward_config)


def make_train_vec_env(reward_config: RewardConfig) -> VecNormalize:
    env = DummyVecEnv([lambda: make_env(reward_config)])
    return VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)


def make_eval_vec_env(reward_config: RewardConfig, render_mode: str | None = None) -> VecNormalize:
    env = DummyVecEnv([lambda: make_env(reward_config, render_mode=render_mode)])
    vec_env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env