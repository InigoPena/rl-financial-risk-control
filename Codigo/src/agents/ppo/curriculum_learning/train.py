# curriculum_train_ppo.py
import os
import sys
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# ---------------------------------------------------------------------
# Path Setup (como tu train_ppo.py)
# ---------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(src_path)
sys.path.append(os.path.join(src_path, "envs"))

from envs.gold_hedge_env import GoldHedgeEnv


# ---------------------------------------------------------------------
# Data (adaptado de tu train_ppo.py)
# ---------------------------------------------------------------------
def load_data():
    """
    Loads and aligns gold and usd data.
    """
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
# Helpers: date -> (start_idx, end_idx) for frame_bound
# ---------------------------------------------------------------------
def date_to_index(df: pd.DataFrame, date_str: str) -> int:
    """
    Returns the integer position of the first row whose index >= date_str.
    """
    date = pd.to_datetime(date_str)
    pos = int(df.index.searchsorted(date, side="left"))
    pos = max(0, min(pos, len(df)))
    return pos


def build_frame_bound(df: pd.DataFrame, start_date: str, end_date: str, window_size: int):
    """
    Builds (start, end) indices for GoldHedgeEnv.frame_bound.
    end is exclusive.
    """
    start_idx = date_to_index(df, start_date)
    end_idx = date_to_index(df, end_date)

    if end_idx <= start_idx:
        raise ValueError(f"Invalid slice: {start_date} -> {end_date} gives {start_idx}..{end_idx}")

    # Ensure start has room for window_size history
    start_idx = max(start_idx, window_size)

    return (start_idx, end_idx)


def make_train_env(gold_df, hedge_df, window_size, frame_bound, n_envs: int):
    env_kwargs = {
        "gold_df": gold_df,
        "hedge_df": hedge_df,
        "window_size": window_size,
        "frame_bound": frame_bound,
    }
    vec_env = make_vec_env(GoldHedgeEnv, n_envs=n_envs, env_kwargs=env_kwargs)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return vec_env


def make_test_env(gold_df, hedge_df, window_size, frame_bound):
    env_kwargs = {
        "gold_df": gold_df,
        "hedge_df": hedge_df,
        "window_size": window_size,
        "frame_bound": frame_bound,
        "render_mode": None,
    }
    vec_env = make_vec_env(GoldHedgeEnv, n_envs=1, env_kwargs=env_kwargs)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=False, clip_obs=10.0)
    return vec_env


def run_episode_and_log(model, vec_env, max_steps: int = None):
    """
    Runs one full episode on vec_env (n_envs=1 recommended), returns dataframe log and final profit.
    """
    obs = vec_env.reset()
    step = 0
    test_log = []
    final_profit = None

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)

        info = infos[0]
        test_log.append(
            {
                "step": step,
                "action": int(action[0]),
                "reward": float(reward[0]),
                "total_profit": info.get("total_profit", np.nan),
                "gold_weight": info.get("gold_weight", np.nan),
                "hedge_weight": info.get("hedge_weight", np.nan),
                "drawdown": info.get("drawdown", np.nan),
            }
        )

        if step % 200 == 0:
            print(
                f"[STEP {step:04d}] A={action[0]} | R={reward[0]:+.5f} | "
                f"V={info.get('total_profit', 0):.4f}"
            )

        step += 1

        if max_steps is not None and step >= max_steps:
            print("Max steps reached, stopping episode early.")
            final_profit = info.get("total_profit", np.nan)
            break

        if dones[0]:
            final_profit = info.get("total_profit", np.nan)
            print("Episode finished.")
            break

    return pd.DataFrame(test_log), final_profit


def main():
    gold_df, hedge_df = load_data()

    # -------------------------
    # Curriculum definition
    # -------------------------
    window_size = 10
    n_envs = 8

    phases = [
        {
            "name": "phase1_trend_2000_2010",
            "start": "2000-01-01",
            "end": "2010-12-31",
            "timesteps": 1_000_000,
        },
        {
            "name": "phase2_volatile_2010_2015",
            "start": "2010-01-01",
            "end": "2015-12-31",
            "timesteps": 1_500_000,
        },
        {
            "name": "phase3_stable_2020_2025",
            "start": "2020-01-01",
            "end": "2025-11-14",
            "timesteps": 1_000_000,
        },
    ]

    test_period = {"name": "test_2015_2020", "start": "2015-01-01", "end": "2020-01-01"}

    # -------------------------
    # Logging + save paths
    # -------------------------
    log_dir = os.path.join(current_dir, "tensorboard_logs_curriculum")
    os.makedirs(log_dir, exist_ok=True)

    save_dir = os.path.join(current_dir, "curriculum_runs")
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------
    # Build first training env
    # -------------------------
    first_fb = build_frame_bound(gold_df, phases[0]["start"], phases[0]["end"], window_size)
    train_vec_env = make_train_env(gold_df, hedge_df, window_size, first_fb, n_envs=n_envs)

    # -------------------------
    # PPO hyperparameters (recomendado)
    # -------------------------
    model = PPO(
        "MlpPolicy",
        train_vec_env,
        verbose=1,
        tensorboard_log=log_dir,
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
    )

    # -------------------------
    # Curriculum loop
    # -------------------------
    prev_obs_rms = None
    prev_ret_rms = None

    for i, ph in enumerate(phases, start=1):
        print("\n" + "=" * 80)
        print(f"CURRICULUM {i}/{len(phases)}: {ph['name']}  [{ph['start']} -> {ph['end']}]")
        print("=" * 80)

        fb = build_frame_bound(gold_df, ph["start"], ph["end"], window_size)

        new_train_env = make_train_env(gold_df, hedge_df, window_size, fb, n_envs=n_envs)

        # Carry over VecNormalize stats
        if prev_obs_rms is not None:
            new_train_env.obs_rms = prev_obs_rms
        if prev_ret_rms is not None:
            new_train_env.ret_rms = prev_ret_rms

        model.set_env(new_train_env)

        model.learn(total_timesteps=int(ph["timesteps"]))

        # Save checkpoint after phase
        ckpt_path = os.path.join(save_dir, f"ppo_{ph['name']}")
        model.save(ckpt_path)
        new_train_env.save(os.path.join(save_dir, f"vecnorm_{ph['name']}.pkl"))
        print(f"Saved checkpoint: {ckpt_path}")

        prev_obs_rms = new_train_env.obs_rms
        prev_ret_rms = new_train_env.ret_rms

        try:
            train_vec_env.close()
        except Exception:
            pass
        train_vec_env = new_train_env

    # -------------------------
    # Test on 2015-2020
    # -------------------------
    print("\n" + "=" * 80)
    print(f"FINAL TEST: {test_period['name']} [{test_period['start']} -> {test_period['end']}]")
    print("=" * 80)

    test_fb = build_frame_bound(gold_df, test_period["start"], test_period["end"], window_size)
    test_vec_env = make_test_env(gold_df, hedge_df, window_size, test_fb)

    # Sync obs normalization stats
    test_vec_env.obs_rms = prev_obs_rms

    df_test, final_profit = run_episode_and_log(model, test_vec_env)

    trace_path = os.path.join(save_dir, f"{test_period['name']}_trace.csv")
    df_test.to_csv(trace_path, index=False)

    print(f"Test trace saved to: {trace_path}")
    print(f"FINAL Test Total Profit: {final_profit:.4f}")


if __name__ == "__main__":
    main()

