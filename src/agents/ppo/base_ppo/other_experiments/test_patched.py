import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# ---------------------------------------------------------------------
# Path Setup
# ---------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, '../../../'))
sys.path.append(src_path)
sys.path.append(os.path.join(src_path, 'envs'))

from envs.gold_hedge_env2 import GoldHedgeEnv

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

    return gold_df, usd_df

def render_portfolio_evolution(df_test, title="Evolución de la Cartera"):
    """
    weights_history: DataFrame with columns 'gold_weight', 'hedge_weight'
    """
    steps = df_test['step']
    
    oro = df_test['gold_weight'].values
    hedge = df_test['hedge_weight'].values
    cash = 1.0 - (oro + hedge) 

    plt.figure(figsize=(14, 8))
    
    # Creamos líneas separadas para cada activo
    plt.plot(steps, oro, label='Oro', color='#ffd700', linewidth=2, marker='o', markersize=3)
    plt.plot(steps, hedge, label='Hedge (USD)', color='#2ecc71', linewidth=2, marker='s', markersize=3)
    plt.plot(steps, cash, label='Cash', color='#3498db', linewidth=2, marker='^', markersize=3)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Días (Ticks)', fontsize=12)
    plt.ylabel('Peso en la Cartera (proporción)', fontsize=12)
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plot_path = os.path.join(current_dir, "portfolio_evolution.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.show()

def main():
    # 1) Load Data
    gold_df, usd_df = load_data()

    # 2) Split Data (Same as training to get the right test part)
    n_total = len(gold_df)
    train_split_idx = int(n_total * 0.70)
    window_size = 10
    
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

    # Load Normalization Stats
    stats_path = os.path.join(current_dir, "vec_normalize.pkl")
    if os.path.exists(stats_path):
        test_vec_env = VecNormalize.load(stats_path, test_vec_env)
        test_vec_env.training = False
        test_vec_env.norm_reward = False
        print("Loaded VecNormalize stats.")
    else:
        print("Warning: vec_normalize.pkl not found. Observations will not be normalized.")

    # Load Model
    model_path = os.path.join(current_dir, "ppo_gold_hedge.zip")
    if not os.path.exists(model_path):
        # Try best model
        model_path = os.path.join(current_dir, "best_model", "best_model.zip")
    
    if os.path.exists(model_path):
        model = PPO.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Error: Model not found at {model_path}")
        return

    # Evaluation Loop
    print("Evaluating on Test Set...")
    obs = test_vec_env.reset()
    test_log = []
    step = 0
    total_profit = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = test_vec_env.step(action)
        info = infos[0]

        test_log.append({
            "step": step,
            "action": int(action[0]),
            "reward": float(reward[0]),
            "total_profit": info.get("total_profit", np.nan),
            "gold_weight": info.get("gold_weight", 0.0),
            "hedge_weight": info.get("hedge_weight", 0.0),
            "drawdown": info.get("drawdown", np.nan),
        })

        if step % 100 == 0:
            print(f"[STEP {step:05d}] A={action[0]} | Profit={info.get('total_profit', 0):.4f}")

        step += 1
        if dones[0]:
            total_profit = info.get("total_profit", 0.0)
            break

    df_test = pd.DataFrame(test_log)
    trace_path = os.path.join(current_dir, "test_trace.csv")
    df_test.to_csv(trace_path, index=False)
    print(f"Test trace saved to {trace_path}")
    print(f"Test Set Total Profit: {total_profit:.4f}")

    # Plotting
    render_portfolio_evolution(df_test, title="PPO Agent Portfolio Evolution (Test Set)")

if __name__ == "__main__":
    main()
