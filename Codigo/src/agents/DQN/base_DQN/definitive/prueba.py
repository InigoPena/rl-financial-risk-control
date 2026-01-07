# train_dqn_basic.py
import os
import sys
import io
from contextlib import redirect_stdout

import pandas as pd
import numpy as np
import gymnasium as gym
import torch as th

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement


# ---------------------------------------------------------------------
# Wrapper: silence noisy env prints (optional but recommended)
# ---------------------------------------------------------------------
class SilentWrapper(gym.Wrapper):
    def step(self, action):
        with redirect_stdout(io.StringIO()):
            return self.env.step(action)

    def reset(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return self.env.reset(**kwargs)


# ---------------------------------------------------------------------
# Path Setup (same style as your PPO script)
# ---------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../../../../"))
sys.path.append(src_path)
sys.path.append(os.path.join(src_path, "envs"))

from envs.gold_hedge_env2 import GoldHedgeEnv  # same env as PPO :contentReference[oaicite:1]{index=1}


# ---------------------------------------------------------------------
# Data (same logic as PPO script)
# ---------------------------------------------------------------------
def load_data():
    base_dir = os.path.dirname(src_path)  # Codigo/
    data_dir = os.path.join(base_dir, "data")

    gold_path = os.path.join(data_dir, "gold_data.csv")
    usd_path = os.path.join(data_dir, "usd_yfinance_2000_2025.csv")

    print(f"Loading data from:\n {gold_path}\n {usd_path}")

    gold_df = pd.read_csv(gold_path, parse_dates=["Date"], index_col="Date")
    usd_df = pd.read_csv(usd_path, parse_dates=["Date"], index_col="Date")
    usd_df.sort_index(inplace=True)

    if "RETURN_1D" not in usd_df.columns:
        usd_df["RETURN_1D"] = usd_df["Close"].pct_change()
    if "VOLATILITY" not in usd_df.columns:
        usd_df["VOLATILITY"] = usd_df["RETURN_1D"].rolling(window=20).std()

    usd_df.dropna(inplace=True)
    gold_df.dropna(inplace=True)

    common_index = gold_df.index.intersection(usd_df.index)
    gold_df = gold_df.loc[common_index]
    usd_df = usd_df.loc[common_index]

    print(f"Aligned Data Points: {len(gold_df)}")
    print(f"Date range: {gold_df.index.min().date()} -> {gold_df.index.max().date()}")

    return gold_df, usd_df


# ---------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------
def make_env(gold_df, hedge_df, window_size, frame_bound, silence: bool):
    def _wrap(env: gym.Env):
        return SilentWrapper(env) if silence else env

    env = make_vec_env(
        GoldHedgeEnv,
        n_envs=1,  # DQN works best with n_envs=1 in SB3
        env_kwargs=dict(
            gold_df=gold_df,
            hedge_df=hedge_df,
            window_size=window_size,
            frame_bound=frame_bound,
            render_mode=None,
        ),
        wrapper_class=_wrap,
    )
    return env


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    # 1) Load data
    gold_df, usd_df = load_data()

    # 2) Split 70/30 (same as PPO script)
    n_total = len(gold_df)
    train_split_idx = int(n_total * 0.70)

    print(f"Total samples: {n_total}")
    print(f"Training samples: {train_split_idx}")
    print(f"Testing samples: {n_total - train_split_idx}")

    # 3) Create envs
    window_size = 10

    train_frame_bound = (window_size, train_split_idx)
    test_frame_bound = (train_split_idx, n_total)

    silence = True  # set False if you want env prints

    train_env = make_env(gold_df, usd_df, window_size, train_frame_bound, silence=silence)
    test_env = make_env(gold_df, usd_df, window_size, test_frame_bound, silence=silence)

    # 4) VecNormalize (IMPORTANT NOTE for DQN):
    # - norm_obs=True helps; norm_reward for off-policy can be risky
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=True)

    test_env = VecNormalize(
        test_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
    )
    test_env.training = False
    test_env.norm_reward = False

    # 5) Logging/saving
    log_dir = os.path.join(current_dir, "tensorboard_logs_dqn")
    os.makedirs(log_dir, exist_ok=True)

    best_dir = os.path.join(current_dir, "best_model_dqn")
    os.makedirs(best_dir, exist_ok=True)

    # 6) DQN hyperparams (good baseline for discrete actions)
    policy_kwargs = dict(
        net_arch=[256, 256],
        activation_fn=th.nn.ReLU,
    )

    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log=log_dir,

        learning_rate=1e-4,
        buffer_size=200_000,
        learning_starts=10_000,
        batch_size=256,
        gamma=0.995,

        train_freq=(4, "step"),
        gradient_steps=1,

        target_update_interval=10_000,
        exploration_fraction=0.30,
        exploration_final_eps=0.05,

        policy_kwargs=policy_kwargs,
        device="auto",
    )

    # 7) Callbacks: best model + early stop
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=5,
        verbose=1,
    )

    eval_callback = EvalCallback(
        test_env,
        best_model_save_path=best_dir,
        log_path=best_dir,
        eval_freq=50_000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        callback_after_eval=stop_cb,
    )

    # 8) Train
    print("Starting DQN Training...")
    TOTAL_TIMESTEPS = 1_000_000
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    print("Training Finished.")

    # 9) Save model + VecNormalize stats
    model_save_path = os.path.join(current_dir, "dqn_gold_hedge")
    model.save(model_save_path)
    train_env.save(os.path.join(current_dir, "vec_normalize_dqn.pkl"))

    print(f"Final model saved to {model_save_path}")
    print(f"Best model (if improved) saved under: {best_dir}")

    # 10) Simple Evaluation Loop (full test episode)
    # Sync obs normalization stats from training -> testing
    test_env.obs_rms = train_env.obs_rms

    print("Evaluating on Test Set...")
    obs = test_env.reset()

    test_log = []
    step = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = test_env.step(action)

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
    trace_path = os.path.join(current_dir, "test_trace_dqn.csv")
    df_test.to_csv(trace_path, index=False)

    print(f"Test trace saved to {trace_path}")
    print(f"Test Set Total Profit: {total_profit:.4f}")


if __name__ == "__main__":
    main()
