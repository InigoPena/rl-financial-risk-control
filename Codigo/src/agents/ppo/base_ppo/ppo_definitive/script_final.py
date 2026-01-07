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
# Current script: Codigo/src/agents/ppo/train_ppo.py
# Goal: Import from Codigo/src/envs
current_dir = os.path.dirname(os.path.abspath(__file__))
# up to agents, up to src
src_path = os.path.abspath(os.path.join(current_dir, '../../../../'))
sys.path.append(src_path)
# Also add envs directly because gold_hedge_env does 'from trading_env import...'
sys.path.append(os.path.join(src_path, 'envs'))

from envs.gold_hedge_env2 import GoldHedgeEnv


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
def load_data():
    """
    Loads and aligns gold and usd data.
    """
    # Base dir: Codigo/
    base_dir = os.path.dirname(src_path)
    data_dir = os.path.join(base_dir, 'data')

    gold_path = os.path.join(data_dir, 'gold_data.csv')
    usd_path = os.path.join(data_dir, 'usd_yfinance_2000_2025.csv')

    print(f"Loading data from:\n {gold_path}\n {usd_path}")

    # Load Gold Data
    gold_df = pd.read_csv(gold_path, parse_dates=['Date'], index_col='Date')

    # Load USD Data (CSV format)
    usd_df = pd.read_csv(usd_path, parse_dates=['Date'], index_col='Date')

    # Sort index to ensure chronological order
    usd_df.sort_index(inplace=True)

    # Feature Engineering for USD
    if 'RETURN_1D' not in usd_df.columns:
        usd_df['RETURN_1D'] = usd_df['Close'].pct_change()

    if 'VOLATILITY' not in usd_df.columns:
        usd_df['VOLATILITY'] = usd_df['RETURN_1D'].rolling(window=20).std()

    # Clean NaNs created by rolling/pct_change
    usd_df.dropna(inplace=True)
    gold_df.dropna(inplace=True)

    # Intersection of indices
    common_index = gold_df.index.intersection(usd_df.index)

    # Filter data
    gold_df = gold_df.loc[common_index]
    usd_df = usd_df.loc[common_index]

    print(f"Aligned Data Points: {len(gold_df)}")

    return gold_df, usd_df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    # 1) Load Data
    gold_df, usd_df = load_data()

    # 2) Split Data 70/30
    n_total = len(gold_df)
    train_split_idx = int(n_total * 0.70)

    print(f"Total samples: {n_total}")
    print(f"Training samples: {train_split_idx}")
    print(f"Testing samples: {n_total - train_split_idx}")

    # 3) Create Environments
    window_size = 10

    # Training Environment (Indices: 0 to train_split_idx)
    train_frame_bound = (window_size, train_split_idx)

    env_kwargs = {
        'gold_df': gold_df,
        'hedge_df': usd_df,
        'window_size': window_size,
        'frame_bound': train_frame_bound
    }

    # Vectorized env
    train_vec_env = make_vec_env(GoldHedgeEnv, n_envs=8, env_kwargs=env_kwargs)

    # Normalize obs + reward during training
    train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Testing Environment (Indices: train_split_idx to end)
    test_frame_bound = (train_split_idx, n_total)

    test_env_kwargs = {
        'gold_df': gold_df,
        'hedge_df': usd_df,
        'window_size': window_size,
        'frame_bound': test_frame_bound,
        'render_mode': None
    }

    test_vec_env = make_vec_env(GoldHedgeEnv, n_envs=1, env_kwargs=test_env_kwargs)

    # Normalize obs only on test, freeze stats
    test_vec_env = VecNormalize(
        test_vec_env,
        norm_obs=True,
        norm_reward=False,
        training=False,
        clip_obs=10.
    )
    test_vec_env.training = False
    test_vec_env.norm_reward = False

    # 4) Setup PPO Training
    log_dir = os.path.join(current_dir, "tensorboard_logs")
    os.makedirs(log_dir, exist_ok=True)

    best_dir = os.path.join(current_dir, "best_model")
    os.makedirs(best_dir, exist_ok=True)

    # Policy network
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=th.nn.ReLU,
        ortho_init=False
    )

    # LR schedule: start high -> decay to low
    lr_schedule = get_linear_fn(7e-4, 1e-4, end_fraction=1.0)

    model = PPO(
        "MlpPolicy",
        train_vec_env,
        verbose=1,
        tensorboard_log=log_dir,

        learning_rate=lr_schedule,
        n_steps=1024,          # 8 envs => 8192 rollout steps per update
        batch_size=512,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,

        policy_kwargs=policy_kwargs,
    )

    # --- Callbacks: save best + early stop if no improvement
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=5,
        verbose=1
    )

    eval_callback = EvalCallback(
        test_vec_env,
        best_model_save_path=best_dir,
        log_path=best_dir,
        eval_freq=50_000,        # adjust if needed
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        callback_after_eval=stop_cb
    )

    print("Starting PPO Training...")

    TOTAL_TIMESTEPS = 1_000_000
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)

    print("Training Finished.")

    # Save the final model and normalization stats
    model_save_path = os.path.join(current_dir, "ppo_gold_hedge")
    model.save(model_save_path)
    train_vec_env.save(os.path.join(current_dir, "vec_normalize.pkl"))
    print(f"Final model saved to {model_save_path}")
    print(f"Best model (if improved) saved under: {best_dir}")

    # 5) Simple Evaluation Loop (full test episode)
    # Sync stats for testing
    test_vec_env.obs_rms = train_vec_env.obs_rms

    print("Evaluating on Test Set...")
    obs = test_vec_env.reset()

    test_log = []
    step = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = test_vec_env.step(action)

        info = infos[0]

        test_log.append({
            "step": step,
            "action": int(action[0]),
            "reward": float(reward[0]),
            "total_profit": info.get("total_profit", np.nan),
            "gold_weight": info.get("gold_weight", np.nan),
            "hedge_weight": info.get("hedge_weight", np.nan),
            "drawdown": info.get("drawdown", np.nan),
        })

        if step % 100 == 0:
            print(
                f"[STEP {step:05d}] "
                f"A={action[0]} | "
                f"R={reward[0]:+.5f} | "
                f"V={info.get('total_profit', 0):.4f}"
            )

        step += 1

        if dones[0]:
            total_profit = info.get("total_profit", 0.0)
            print("Episode finished.")
            break

    df_test = pd.DataFrame(test_log)
    trace_path = os.path.join(current_dir, "test_trace.csv")
    df_test.to_csv(trace_path, index=False)

    print(f"Test trace saved to {trace_path}")
    print(f"Test Set Total Profit: {total_profit:.4f}")


if __name__ == "__main__":
    main()
