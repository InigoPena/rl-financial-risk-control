# curriculum_prueba_ppo.py
import os
import sys
import io
from contextlib import redirect_stdout
import pandas as pd
import numpy as np

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.utils import get_linear_fn


import gymnasium as gym


# ---------------------------------------------------------------------
# Wrapper: silence noisy env prints (GoldHedgeEnv2 prints each step)
# ---------------------------------------------------------------------
class SilentWrapper(gym.Wrapper):
    """
    Suppresses stdout during env.step/reset to avoid huge training slowdowns/log spam.
    Useful because gold_hedge_env2.py prints a lot every step.
    """
    def step(self, action):
        with redirect_stdout(io.StringIO()):
            return self.env.step(action)

    def reset(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return self.env.reset(**kwargs)


# ---------------------------------------------------------------------
# Path Setup (same style as prueba.py)
# ---------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
# Adjust if your folder structure differs; this mirrors your prueba.py approach
src_path = os.path.abspath(os.path.join(current_dir, "../../../../"))
sys.path.append(src_path)
sys.path.append(os.path.join(src_path, "envs"))

# IMPORTANT: you said you decided to use gold_hedge_env2
from envs.gold_hedge_env2 import GoldHedgeEnv  # :contentReference[oaicite:3]{index=3}


# ---------------------------------------------------------------------
# Data (same logic as prueba.py)
# ---------------------------------------------------------------------
def load_data():
    """
    Loads and aligns gold and usd data.
    Mirrors prueba.py loader.
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
# Helpers: date -> index, and build frame_bound safely
# ---------------------------------------------------------------------
def date_to_idx(df: pd.DataFrame, date_str: str) -> int:
    # searchsorted returns insertion index in sorted DateTimeIndex
    dt = pd.to_datetime(date_str)
    idx = int(df.index.searchsorted(dt))
    return max(idx, 0)


def frame_bound_from_dates(df: pd.DataFrame, start_date: str, end_date: str, window_size: int):
    """
    Returns (start_idx, end_idx) indices for frame_bound, consistent with your env usage:
    env uses prices[start-window:end] internally, so start must be >= window_size.
    """
    s = date_to_idx(df, start_date)
    e = date_to_idx(df, end_date)

    s = max(s, window_size)
    e = max(e, s + 1)  # ensure at least 1 step

    return (s, e)


# ---------------------------------------------------------------------
# Env factory (VecNormalize + optional SilentWrapper)
# ---------------------------------------------------------------------
def make_norm_env(
    gold_df,
    hedge_df,
    window_size,
    frame_bound,
    n_envs,
    train: bool,
    silence: bool,
):
    def _wrap_env(env: gym.Env):
        return SilentWrapper(env) if silence else env

    vec = make_vec_env(
        GoldHedgeEnv,
        n_envs=n_envs,
        env_kwargs=dict(
            gold_df=gold_df,
            hedge_df=hedge_df,
            window_size=window_size,
            frame_bound=frame_bound,
            render_mode=None,
        ),
        wrapper_class=_wrap_env,
    )

    vec = VecNormalize(
        vec,
        norm_obs=True,
        norm_reward=train,
        training=train,
        clip_obs=10.0,
    )

    # Make sure test env never updates reward stats
    if not train:
        vec.training = False
        vec.norm_reward = False

    return vec


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    # 1) Load data
    gold_df, usd_df = load_data()

    window_size = 10
    n_envs = 8

    # ===============================================================
    # CURRICULUM STAGES (edit dates/timesteps to match your report)
    # ===============================================================
    curriculum = [
        # Stage 1 – Trending market regime (example split like your PPO curriculum)
        {"name": "stage1_trend_2000_2010", "start": "2000-01-01", "end": "2010-12-31", "timesteps": 800_000},
        {"name": "stage1_trend_2024_2025", "start": "2024-01-01", "end": "2025-11-14", "timesteps": 400_000},

        # Stage 2 – Stable market regime
        {"name": "stage2_stable_2020_2024", "start": "2020-01-01", "end": "2024-12-31", "timesteps": 900_000},

        # Stage 3 – Volatile market regime
        {"name": "stage3_volatile_2010_2015", "start": "2010-01-01", "end": "2015-12-31", "timesteps": 1_000_000},
    ]

    # Hold-out test period (never trained on)
    test_period = ("2015-01-01", "2020-01-01")

    # ===============================================================
    # Logging / saving
    # ===============================================================
    log_dir = os.path.join(current_dir, "tensorboard_logs_curriculum")
    os.makedirs(log_dir, exist_ok=True)

    best_dir = os.path.join(current_dir, "best_model_curriculum")
    os.makedirs(best_dir, exist_ok=True)

    # ===============================================================
    # Build FIRST training env (stage 1)
    # ===============================================================
    first_bound = frame_bound_from_dates(gold_df, curriculum[0]["start"], curriculum[0]["end"], window_size)

    train_env = make_norm_env(
        gold_df, usd_df, window_size, first_bound,
        n_envs=n_envs, train=True, silence=True
    )

    # ===============================================================
    # Test env (fixed, for EvalCallback)
    # ===============================================================
    test_bound = frame_bound_from_dates(gold_df, test_period[0], test_period[1], window_size)

    test_env = make_norm_env(
        gold_df, usd_df, window_size, test_bound,
        n_envs=1, train=False, silence=True
    )

    # ===============================================================
    # PPO config (mirrors prueba.py)
    # ===============================================================
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=th.nn.ReLU,
        ortho_init=False,
    )

    lr_schedule =get_linear_fn(7e-4, 1e-4, end_fraction=1.0)

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=log_dir,

        learning_rate=lr_schedule,
        n_steps=1024,         # 8 envs -> 8192 steps/update
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

    # ===============================================================
    # Callbacks: best model + early stop
    # ===============================================================
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

    # ===============================================================
    # Curriculum loop (carry VecNormalize stats across stages)
    # ===============================================================
    obs_rms, ret_rms = None, None

    for i, stage in enumerate(curriculum, start=1):
        print(f"\n{'='*80}")
        print(f"CURRICULUM STAGE {i}/{len(curriculum)}: {stage['name']}")
        print(f"Dates: {stage['start']} -> {stage['end']} | Timesteps: {stage['timesteps']:,}")
        print(f"{'='*80}")

        bound = frame_bound_from_dates(gold_df, stage["start"], stage["end"], window_size)

        stage_env = make_norm_env(
            gold_df, usd_df, window_size, bound,
            n_envs=n_envs, train=True, silence=True
        )

        # Carry normalization stats forward (VERY IMPORTANT for curriculum)
        if obs_rms is not None:
            stage_env.obs_rms = obs_rms
            stage_env.ret_rms = ret_rms

        # Also keep test env aligned with latest obs_rms
        if obs_rms is not None:
            test_env.obs_rms = obs_rms

        model.set_env(stage_env)

        # Train this stage with evaluation
        model.learn(total_timesteps=stage["timesteps"], callback=eval_callback)

        # Save stage checkpoint
        stage_ckpt = os.path.join(current_dir, f"ppo_curriculum_{stage['name']}")
        model.save(stage_ckpt)
        print(f"Saved stage checkpoint: {stage_ckpt}")

        # Update stats to carry to next stage
        obs_rms, ret_rms = stage_env.obs_rms, stage_env.ret_rms
        stage_env.close()

    # Final save
    final_path = os.path.join(current_dir, "ppo_curriculum_final")
    model.save(final_path)
    print(f"\n✅ Curriculum training finished. Final model saved to: {final_path}")
    print(f"✅ Best model snapshots (EvalCallback) under: {best_dir}")

    # Save final VecNormalize stats (useful for later test scripts)
    # Note: save stats from the *last* stage's carried values by attaching them to train_env before saving
    train_env.obs_rms = obs_rms
    train_env.ret_rms = ret_rms
    vec_path = os.path.join(current_dir, "vec_normalize_curriculum.pkl")
    train_env.save(vec_path)
    print(f"✅ VecNormalize stats saved to: {vec_path}")

    # Quick final evaluation episode
    print("Evaluating on Test Set...")
    test_env.obs_rms = obs_rms
    obs = test_env.reset()
    
    test_log = []
    step = 0
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        
        # VecEnv info is a list
        info0 = info[0] if isinstance(info, (list, tuple)) else info
        
        test_log.append({
            "step": step,
            "action": int(action[0]),
            "reward": float(reward[0]),
            "total_profit": info0.get("total_profit", np.nan),
            "gold_weight": info0.get("gold_weight", np.nan),
            "hedge_weight": info0.get("hedge_weight", np.nan),
            "drawdown": info0.get("drawdown", np.nan),
        })

        if step % 100 == 0:
            print(
                f"[STEP {step:05d}] "
                f"A={action[0]} | "
                f"R={reward[0]:+.5f} | "
                f"V={info0.get('total_profit', 0):.4f}"
            )
        
        step += 1
        if done[0] if isinstance(done, (list, np.ndarray)) else done:
            total_profit = info0.get("total_profit", 0.0)
            print("Episode finished.")
            break

    df_test = pd.DataFrame(test_log)
    trace_path = os.path.join(current_dir, "test_trace_curriculum.csv")
    df_test.to_csv(trace_path, index=False)

    print(f"✅ Test trace saved to {trace_path}")
    print(f"📌 Final test episode completed. total_profit={total_profit:.4f}")


if __name__ == "__main__":
    main()
