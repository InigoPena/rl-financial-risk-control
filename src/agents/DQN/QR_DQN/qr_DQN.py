import os
import sys
import io
import pandas as pd
import numpy as np
import gymnasium as gym
import torch as th
from contextlib import redirect_stdout

# Algoritmo Distribucional (Base de Rainbow en SB3)
from sb3_contrib import QRDQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

# ---------------------------------------------------------------------
# 1. Wrapper para Silenciar el Entorno
# ---------------------------------------------------------------------
class SilentWrapper(gym.Wrapper):
    def step(self, action):
        with redirect_stdout(io.StringIO()):
            return self.env.step(action)
    def reset(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return self.env.reset(**kwargs)

# ---------------------------------------------------------------------
# 2. Configuración de Rutas e Importación
# ---------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(src_path)
sys.path.append(os.path.join(src_path, "envs"))

from envs.gold_hedge_env2 import GoldHedgeEnv

# ---------------------------------------------------------------------
# 3. Carga y Procesamiento de Datos
# ---------------------------------------------------------------------
def load_data():
    base_dir = os.path.dirname(src_path) 
    data_dir = os.path.join(base_dir, "data")
    gold_path = os.path.join(data_dir, "gold_data.csv")
    usd_path = os.path.join(data_dir, "usd_yfinance_2000_2025.csv")

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
    return gold_df.loc[common_index], usd_df.loc[common_index]

# ---------------------------------------------------------------------
# 4. Factory de Entornos
# ---------------------------------------------------------------------
def make_env(gold_df, hedge_df, window_size, frame_bound, silence: bool):
    def _wrap(env: gym.Env):
        return SilentWrapper(env) if silence else env
    return make_vec_env(
        GoldHedgeEnv,
        n_envs=1, 
        env_kwargs=dict(gold_df=gold_df, hedge_df=hedge_df, window_size=window_size, frame_bound=frame_bound),
        wrapper_class=_wrap,
    )

# ---------------------------------------------------------------------
# 5. Ejecución Principal
# ---------------------------------------------------------------------
def main():
    gold_df, usd_df = load_data()
    n_total = len(gold_df)
    train_split_idx = int(n_total * 0.70)
    window_size = 10

    train_env = make_env(gold_df, usd_df, window_size, (window_size, train_split_idx), silence=True)
    test_env = make_env(gold_df, usd_df, window_size, (train_split_idx, n_total), silence=True)

    # Normalización: Crucial para la convergencia en finanzas
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    test_env = VecNormalize(test_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    # Configuración QR-DQN (Estructura Distribucional tipo Rainbow)
    policy_kwargs = dict(n_quantiles=50, net_arch=[256, 256], activation_fn=th.nn.ReLU)

    model = QRDQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-4,
        buffer_size=150_000,
        learning_starts=10_000,
        batch_size=128,
        gamma=0.99,
        target_update_interval=5000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./tensorboard_rainbow/",
        verbose=1
    )

    # Entrenamiento con Callback de guardado
    eval_callback = EvalCallback(test_env, best_model_save_path="./best_model_rainbow/", eval_freq=25000)
    
    print("Iniciando entrenamiento...")
    model.learn(total_timesteps=800_000, callback=eval_callback)
    
    # --- EVALUACIÓN Y GENERACIÓN DE CSV ---
    print("\nEvaluando en el conjunto de Test y generando CSV...")
    test_env.obs_rms = train_env.obs_rms  # Sincronizar estadísticas de normalización
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

        if step % 200 == 0:
            print(f"[STEP {step:05d}] Acción={action[0]} | Profit={info.get('total_profit', 0):.4f}")

        step += 1
        if dones[0]: break

    # Guardar traza en CSV
    df_test = pd.DataFrame(test_log)
    trace_path = os.path.join(current_dir, "test_trace_qrdqn.csv")
    df_test.to_csv(trace_path, index=False)
    
    print(f"\nTrace guardado en: {trace_path}")
    print(f"Profit Final en Test: {test_log[-1]['total_profit']:.4f}")

    model.save("qrdqn_final_model")
    train_env.save("vec_normalize_qrdqn.pkl")

if __name__ == "__main__":
    main()