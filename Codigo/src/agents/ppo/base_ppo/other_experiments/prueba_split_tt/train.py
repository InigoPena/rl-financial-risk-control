import os
import sys
import pandas as pd
import numpy as np

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.utils import get_linear_fn

# ---------------------------------------------------------------------
# Path Setup
# ---------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, '../../../../'))
sys.path.append(src_path)
sys.path.append(os.path.join(src_path, 'envs'))

from envs.gold_hedge_env2_patched import GoldHedgeEnv

def load_data():
    """
    Loads and aligns gold and usd data.
    """
    base_dir = os.path.dirname(src_path)
    data_dir = os.path.join(base_dir, 'data')

    gold_path = os.path.join(data_dir, 'gold_data.csv')
    usd_path = os.path.join(data_dir, 'usd_yfinance_2000_2025.csv')

    print(f"Loading data from:\n {gold_path}\n {usd_path}")

    gold_df = pd.read_csv(gold_path, parse_dates=['Date'], index_col='Date')
    usd_df = pd.read_csv(usd_path, parse_dates=['Date'], index_col='Date')
    usd_df.sort_index(inplace=True)

    if 'RETURN_1D' not in usd_df.columns:
        usd_df['RETURN_1D'] = usd_df['Close'].pct_change()

    if 'VOLATILITY' not in usd_df.columns:
        usd_df['VOLATILITY'] = usd_df['RETURN_1D'].rolling(window=20).std()

    usd_df.dropna(inplace=True)
    gold_df.dropna(inplace=True)

    common_index = gold_df.index.intersection(usd_df.index)
    gold_df = gold_df.loc[common_index]
    usd_df = usd_df.loc[common_index]

    print(f"Aligned Data Points: {len(gold_df)}")
    return gold_df, usd_df

def main():
    # 1) Load Data
    gold_df, usd_df = load_data()

    # 2) Split Data 70/30
    n_total = len(gold_df)
    train_split_idx = int(n_total * 0.70)
    print(f"Total samples: {n_total}")
    print(f"Training samples: {train_split_idx}")

    # 3) Create Environments
    window_size = 10
    train_frame_bound = (window_size, train_split_idx)

    env_kwargs = {
        'gold_df': gold_df,
        'hedge_df': usd_df,
        'window_size': window_size,
        'frame_bound': train_frame_bound
    }

    train_vec_env = make_vec_env(GoldHedgeEnv, n_envs=8, env_kwargs=env_kwargs)
    train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    test_frame_bound = (train_split_idx, n_total)
    test_env_kwargs = {
        'gold_df': gold_df,
        'hedge_df': usd_df,
        'window_size': window_size,
        'frame_bound': test_frame_bound,
        'render_mode': None
    }
    test_vec_env = make_vec_env(GoldHedgeEnv, n_envs=1, env_kwargs=test_env_kwargs)
    test_vec_env = VecNormalize(test_vec_env, norm_obs=True, norm_reward=False, training=False, clip_obs=10.)

    # 4) Setup PPO Training
    log_dir = os.path.join(current_dir, "tensorboard_logs")
    os.makedirs(log_dir, exist_ok=True)
    best_dir = os.path.join(current_dir, "best_model")
    os.makedirs(best_dir, exist_ok=True)

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=th.nn.ReLU,
        ortho_init=False
    )
    lr_schedule = get_linear_fn(7e-4, 1e-4, end_fraction=1.0)

    model = PPO(
        "MlpPolicy",
        train_vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=lr_schedule,
        n_steps=1024,
        batch_size=512,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=2e-3, # Increased slightly for better exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
    )

    stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=5, verbose=1)
    eval_callback = EvalCallback(
        test_vec_env,
        best_model_save_path=best_dir,
        log_path=best_dir,
        eval_freq=50_000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        callback_after_eval=stop_cb
    )

    print("Starting PPO Training...")
    TOTAL_TIMESTEPS = 1_000_000
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)

    model_save_path = os.path.join(current_dir, "ppo_gold_hedge")
    model.save(model_save_path)
    train_vec_env.save(os.path.join(current_dir, "vec_normalize.pkl"))
    print(f"Final model saved to {model_save_path}")

if __name__ == "__main__":
    main()
