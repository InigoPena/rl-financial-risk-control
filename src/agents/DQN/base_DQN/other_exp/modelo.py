# train_dqn_curriculum.py
import os
import sys
import numpy as np
import pandas as pd

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

# ---------------------------------------------------------------------
# Path Setup (igual idea que tus scripts)
# ---------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))

# Ajusta esto a tu estructura real:
# - En tu train.py estabas subiendo "../../" y añadiendo "envs"
src_path = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_path)
sys.path.append(os.path.join(src_path, "envs"))

# Importa el env que estás usando en DQN ahora
from envs.gold_hedge_env2_patched import GoldHedgeEnv


# ---------------------------------------------------------------------
# Data (basado en tu train.py, pero robusto con features si faltan)
# ---------------------------------------------------------------------
def load_data():
    """
    Loads and aligns gold and usd data.
    Adds RETURN_1D and VOLATILITY if missing.
    """
    base_dir = os.path.dirname(src_path)  # Codigo/
    data_dir = os.path.join(base_dir, "data")

    gold_path = os.path.join(data_dir, "gold_data.csv")
    usd_path = os.path.join(data_dir, "usd_yfinance_2000_2025.csv")

    print(f"Loading data from:\n  {gold_path}\n  {usd_path}")

    gold_df = pd.read_csv(gold_path, parse_dates=["Date"], index_col="Date")
    usd_df = pd.read_csv(usd_path, parse_dates=["Date"], index_col="Date")

    gold_df.sort_index(inplace=True)
    usd_df.sort_index(inplace=True)

    # Ensure USD features exist
    if "RETURN_1D" not in usd_df.columns:
        usd_df["RETURN_1D"] = usd_df["Close"].pct_change()
    if "VOLATILITY" not in usd_df.columns:
        usd_df["VOLATILITY"] = usd_df["RETURN_1D"].rolling(window=20).std()

    # (Opcional pero seguro) si el env algún día pide estas columnas en gold también
    if "RETURN_1D" not in gold_df.columns and "Close" in gold_df.columns:
        gold_df["RETURN_1D"] = gold_df["Close"].pct_change()
    if "VOLATILITY" not in gold_df.columns and "RETURN_1D" in gold_df.columns:
        gold_df["VOLATILITY"] = gold_df["RETURN_1D"].rolling(window=20).std()

    gold_df.dropna(inplace=True)
    usd_df.dropna(inplace=True)

    common_index = gold_df.index.intersection(usd_df.index)
    gold_df = gold_df.loc[common_index]
    usd_df = usd_df.loc[common_index]

    print(f"Aligned Data Points: {len(gold_df)}")
    return gold_df, usd_df


# ---------------------------------------------------------------------
# Helpers (copiados conceptualmente de tu curriculum PPO)
# ---------------------------------------------------------------------
def date_to_idx(df, date_str: str) -> int:
    return max(df.index.searchsorted(pd.to_datetime(date_str)), 0)


def frame(df, start: str, end: str, window: int):
    s = max(date_to_idx(df, start), window)
    e = date_to_idx(df, end)
    return (s, e)


def make_env(gold_df, hedge_df, window_size, frame_bound, train: bool):
    env = make_vec_env(
        GoldHedgeEnv,
        n_envs=1,  # DQN mejor 1 env (más estable)
        env_kwargs=dict(
            gold_df=gold_df,
            hedge_df=hedge_df,
            window_size=window_size,
            frame_bound=frame_bound,
            render_mode=None,
        ),
    )
    return VecNormalize(
        env,
        norm_obs=True,
        norm_reward=train,  # train=True -> normaliza reward; test=False -> no
        training=train,
        clip_obs=10.0,
    )


def run_episode_and_log(model, vec_env, print_every=250):
    """
    Runs one full episode on vec_env (n_envs=1), returns dataframe log and final profit.
    """
    obs = vec_env.reset()
    test_log = []
    step = 0
    total_profit = None

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

        if step % print_every == 0:
            print(
                f"[STEP {step:05d}] A={action[0]} | R={reward[0]:+.5f} | "
                f"V={info.get('total_profit', 0):.4f} | DD={info.get('drawdown', np.nan):.4f}"
            )

        step += 1
        if dones[0]:
            total_profit = info.get("total_profit", 0.0)
            print("Episode finished.")
            break

    return pd.DataFrame(test_log), float(total_profit)


# ---------------------------------------------------------------------
# Hyperparams candidates (elige uno)
# ---------------------------------------------------------------------
DQN_CONFIGS = {
    # A) Más conservador / estable (suele ir bien en trading discreto)
    "A_stable": dict(
        learning_rate=5e-5,
        buffer_size=300_000,
        learning_starts=20_000,
        batch_size=128,
        gamma=0.995,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=20_000,
        exploration_fraction=0.30,
        exploration_final_eps=0.05,
        max_grad_norm=10.0,
    ),
    # B) Más agresivo (aprende antes, pero puede ser más ruidoso)
    "B_faster": dict(
        learning_rate=1e-4,
        buffer_size=200_000,
        learning_starts=10_000,
        batch_size=64,
        gamma=0.995,
        train_freq=4,
        gradient_steps=2,
        target_update_interval=10_000,
        exploration_fraction=0.20,
        exploration_final_eps=0.05,
        max_grad_norm=10.0,
    ),
}


def main():
    gold, hedge = load_data()
    window_size = 10

    # ===============================================================
    # CURRICULUM STAGES (MISMAS FECHAS que tu PPO curriculum)
    # ===============================================================
    curriculum = [
        # Stage 1 – Trending market regime
        {"name": "stage1_trend_2000_2010", "start": "2000-01-01", "end": "2010-12-31", "timesteps": 800_000},
        {"name": "stage1_trend_2024_2025", "start": "2024-01-01", "end": "2025-11-14", "timesteps": 400_000},
        # Stage 2 – Stable market regime
        {"name": "stage2_stable_2020_2024", "start": "2020-01-01", "end": "2024-12-31", "timesteps": 800_000},
        # Stage 3 – Volatile market regime
        {"name": "stage3_volatile_2010_2015", "start": "2010-01-01", "end": "2015-12-31", "timesteps": 1_200_000},
    ]

    test_period = ("2015-01-01", "2020-01-01")

    # ===============================================================
    # Choose hyperparams
    # ===============================================================
    cfg_name = "A_stable"   # <-- cambia a "B_faster" si quieres comparar
    cfg = DQN_CONFIGS[cfg_name]
    print(f"\n=== Using DQN config: {cfg_name} ===")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    # Logging dirs
    log_dir = os.path.join(current_dir, f"tensorboard_logs_dqn_curriculum_{cfg_name}")
    best_dir = os.path.join(current_dir, f"best_dqn_curriculum_{cfg_name}")
    eval_dir = os.path.join(current_dir, f"eval_logs_dqn_curriculum_{cfg_name}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # ===============================================================
    # Init env with first stage
    # ===============================================================
    first_bound = frame(gold, curriculum[0]["start"], curriculum[0]["end"], window_size)
    train_env = make_env(gold, hedge, window_size, first_bound, train=True)

    # Test env for callback (2015-2020)
    test_bound = frame(gold, *test_period, window_size)
    test_env = make_env(gold, hedge, window_size, test_bound, train=False)

    # Important: sync obs stats later
    # (EvalCallback uses test_env; we'll keep it and update obs_rms after each stage)
    eval_callback = EvalCallback(
        test_env,
        best_model_save_path=best_dir,
        log_path=eval_dir,
        eval_freq=100_000,     # en curriculum tiene sentido evaluar menos a menudo
        n_eval_episodes=1,
        deterministic=True,
        render=False,
    )

    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log=log_dir,
        **cfg,
    )

    # Keep normalization stats across stages
    obs_rms, ret_rms = None, None

    # ===============================================================
    # Curriculum loop
    # ===============================================================
    for stage in curriculum:
        print(f"\n=== Training {stage['name']} ({stage['start']} -> {stage['end']}) ===")

        bound = frame(gold, stage["start"], stage["end"], window_size)
        env = make_env(gold, hedge, window_size, bound, train=True)

        # carry normalization stats
        if obs_rms is not None:
            env.obs_rms = obs_rms
            env.ret_rms = ret_rms

        model.set_env(env)

        # Also ensure callback env sees updated obs stats (so eval isn't off)
        test_env.obs_rms = env.obs_rms

        model.learn(total_timesteps=stage["timesteps"], callback=eval_callback)

        obs_rms, ret_rms = env.obs_rms, env.ret_rms
        env.close()

    # Final save
    final_model_path = os.path.join(current_dir, f"dqn_curriculum_final_{cfg_name}")
    model.save(final_model_path)
    train_env.save(os.path.join(current_dir, f"vec_normalize_dqn_curriculum_{cfg_name}.pkl"))
    print(f"\nSaved final model to: {final_model_path}")

    # ===============================================================
    # Final TEST (2015–2020)
    # ===============================================================
    print("\n=== Final evaluation on test period (2015-2020) ===")
    test_env.obs_rms = obs_rms  # sync obs stats from last stage

    df_test, total_profit = run_episode_and_log(model, test_env, print_every=250)
    trace_path = os.path.join(current_dir, f"test_trace_dqn_curriculum_{cfg_name}.csv")
    df_test.to_csv(trace_path, index=False)

    print(f"Test trace saved to {trace_path}")
    print(f"Test Set Total Profit: {total_profit:.4f}")
    print("Final evaluation completed.")


if __name__ == "__main__":
    main()
