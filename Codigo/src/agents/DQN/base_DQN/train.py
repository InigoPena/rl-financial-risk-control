# train_dqn.py
import os
import sys
import pandas as pd
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

# ---------------------------------------------------------------------
# Path Setup (como tu train_ppo.py)
# ---------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_path)
sys.path.append(os.path.join(src_path, "envs"))

from envs.gold_hedge_env2_patched import GoldHedgeEnv


# ---------------------------------------------------------------------
# Data (copiado de tu train_ppo.py)
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
    return gold_df, usd_df


def run_episode_and_log(model, vec_env, print_every=100):
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
                f"[STEP {step:04d}] "
                f"A={action[0]} | R={reward[0]:+.5f} | "
                f"V={info.get('total_profit', 0):.4f}"
            )

        step += 1

        if dones[0]:
            total_profit = info.get("total_profit", 0.0)
            print("Episode finished.")
            break

    return pd.DataFrame(test_log), float(total_profit)


def main():
    # 1) Load data
    gold_df, usd_df = load_data()

    # 2) Split 70/30 (idéntico a tu PPO script)
    n_total = len(gold_df)
    train_split_idx = int(n_total * 0.70)

    print(f"Total samples: {n_total}")
    print(f"Training samples: {train_split_idx}")
    print(f"Testing samples: {n_total - train_split_idx}")

    # 3) Environments
    window_size = 10

    train_frame_bound = (window_size, train_split_idx)
    test_frame_bound = (train_split_idx, n_total)

    train_env_kwargs = {
        "gold_df": gold_df,
        "hedge_df": usd_df,
        "window_size": window_size,
        "frame_bound": train_frame_bound,
    }

    test_env_kwargs = {
        "gold_df": gold_df,
        "hedge_df": usd_df,
        "window_size": window_size,
        "frame_bound": test_frame_bound,
        "render_mode": None,
    }

    # DQN: mejor empezar con 1 env para estabilidad
    train_vec_env = make_vec_env(GoldHedgeEnv, n_envs=1, env_kwargs=train_env_kwargs)
    train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    test_vec_env = make_vec_env(GoldHedgeEnv, n_envs=1, env_kwargs=test_env_kwargs)
    test_vec_env = VecNormalize(test_vec_env, norm_obs=True, norm_reward=False, training=False, clip_obs=10.0)

    # 4) DQN hyperparams (robusto para trading discreto)
    log_dir = os.path.join(current_dir, "tensorboard_logs_dqn")
    os.makedirs(log_dir, exist_ok=True)

    # Eval callback: evalúa periódicamente en el set de test
    eval_callback = EvalCallback(
        test_vec_env,
        best_model_save_path=os.path.join(current_dir, "best_dqn"),
        log_path=os.path.join(current_dir, "eval_logs_dqn"),
        eval_freq=50_000,          # cada X steps (ajusta si quieres)
        n_eval_episodes=1,         # tu test es un episodio largo; 1 suele bastar
        deterministic=True,
        render=False,
    )

    model = DQN(
        policy="MlpPolicy",
        env=train_vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        # --- core DQN ---
        learning_rate=1e-4,
        buffer_size=200_000,
        learning_starts=10_000,
        batch_size=64,
        gamma=0.995,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10_000,
        # --- exploration ---
        exploration_fraction=0.20,
        exploration_final_eps=0.05,
        # --- stability ---
        max_grad_norm=10.0,
    )

    print("Starting DQN Training...")
    TOTAL_TIMESTEPS = 2_000_000  # punto de partida razonable para DQN
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    print("Training Finished.")

    # Save model + VecNormalize
    model_save_path = os.path.join(current_dir, "dqn_gold_hedge")
    model.save(model_save_path)
    train_vec_env.save(os.path.join(current_dir, "vec_normalize_dqn.pkl"))
    print(f"Model saved to {model_save_path}")

    # 5) Evaluate on test set (sync obs stats)
    test_vec_env.obs_rms = train_vec_env.obs_rms

    print("Evaluating on Test Set...")
    df_test, total_profit = run_episode_and_log(model, test_vec_env, print_every=200)

    trace_path = os.path.join(current_dir, "test_trace_dqn.csv")
    df_test.to_csv(trace_path, index=False)

    print(f"Test trace saved to {trace_path}")
    print(f"Test Set Total Profit: {total_profit:.4f}")


if __name__ == "__main__":
    main()
