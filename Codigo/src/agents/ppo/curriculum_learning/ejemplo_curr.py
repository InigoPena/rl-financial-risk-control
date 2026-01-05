# curriculum_train_ppo.py
import os
import sys
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# ---------------------------------------------------------------------
# Path Setup
# ---------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(src_path)
sys.path.append(os.path.join(src_path, "envs"))

from envs.gold_hedge_env import GoldHedgeEnv


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
def load_data():
    base_dir = os.path.dirname(src_path)
    data_dir = os.path.join(base_dir, "data")

    gold_df = pd.read_csv(
        os.path.join(data_dir, "gold_data.csv"),
        parse_dates=["Date"],
        index_col="Date",
    )
    usd_df = pd.read_csv(
        os.path.join(data_dir, "usd_yfinance_2000_2025.csv"),
        parse_dates=["Date"],
        index_col="Date",
    )

    usd_df.sort_index(inplace=True)
    usd_df["RETURN_1D"] = usd_df["Close"].pct_change()
    usd_df["VOLATILITY"] = usd_df["RETURN_1D"].rolling(20).std()

    gold_df.dropna(inplace=True)
    usd_df.dropna(inplace=True)

    common_idx = gold_df.index.intersection(usd_df.index)
    return gold_df.loc[common_idx], usd_df.loc[common_idx]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def date_to_idx(df, date):
    return max(df.index.searchsorted(pd.to_datetime(date)), 0)


def frame(df, start, end, window):
    s = max(date_to_idx(df, start), window)
    e = date_to_idx(df, end)
    return (s, e)


def make_env(gold, hedge, window, bound, n_envs=8, train=True):
    env = make_vec_env(
        GoldHedgeEnv,
        n_envs=n_envs,
        env_kwargs=dict(
            gold_df=gold,
            hedge_df=hedge,
            window_size=window,
            frame_bound=bound,
        ),
    )
    return VecNormalize(
        env,
        norm_obs=True,
        norm_reward=train,
        training=train,
        clip_obs=10.0,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    gold, hedge = load_data()
    window_size = 10
    n_envs = 8

    # ===============================================================
    # CURRICULUM STAGES (MATCHING THE REPORT)
    # ===============================================================

    curriculum = [
        # Stage 1 – Trending market regime
        {
            "name": "stage1_trend_2000_2010",
            "start": "2000-01-01",
            "end": "2010-12-31",
            "timesteps": 1_000_000,
        },
        {
            "name": "stage1_trend_2024_2025",
            "start": "2024-01-01",
            "end": "2025-11-14",
            "timesteps": 500_000,
        },
        # Stage 2 – Stable market regime
        {
            "name": "stage2_stable_2020_2024",
            "start": "2020-01-01",
            "end": "2024-12-31",
            "timesteps": 1_000_000,
        },
        # Stage 3 – Volatile market regime
        {
            "name": "stage3_volatile_2010_2015",
            "start": "2010-01-01",
            "end": "2015-12-31",
            "timesteps": 1_500_000,
        },
    ]

    test_period = ("2015-01-01", "2020-01-01")

    # ===============================================================
    # PPO Model
    # ===============================================================
    first_env = make_env(
        gold,
        hedge,
        window_size,
        frame(gold, curriculum[0]["start"], curriculum[0]["end"], window_size),
        n_envs,
        train=True,
    )

    model = PPO(
        "MlpPolicy",
        first_env,
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0075,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,
        verbose=1,
        tensorboard_log=os.path.join(current_dir, "tensorboard_logs_curriculum_prueba"),
    )

    obs_rms, ret_rms = None, None

    # ===============================================================
    # Curriculum loop
    # ===============================================================
    for stage in curriculum:
        print(f"\n=== Training {stage['name']} ===")

        env = make_env(
            gold,
            hedge,
            window_size,
            frame(gold, stage["start"], stage["end"], window_size),
            n_envs,
            train=True,
        )

        if obs_rms is not None:
            env.obs_rms = obs_rms
            env.ret_rms = ret_rms

        model.set_env(env)
        model.learn(stage["timesteps"])

        obs_rms, ret_rms = env.obs_rms, env.ret_rms
        env.close()

    model.save("ppo_curriculum_final")

    # ===============================================================
    # TEST (2015–2020)
    # ===============================================================
    test_env = make_env(
        gold,
        hedge,
        window_size,
        frame(gold, *test_period, window_size),
        n_envs=1,
        train=False,
    )
    test_env.obs_rms = obs_rms

    obs = test_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = test_env.step(action)

    print("Final evaluation completed.")


if __name__ == "__main__":
    main()
