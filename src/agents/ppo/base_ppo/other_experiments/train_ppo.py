import os
import sys
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

# Add the parent directory 'src' to sys.path to allow imports from 'envs'
# Current script: Codigo/src/agents/ppo/train_ppo.py
# Goal: Import from Codigo/src/envs
current_dir = os.path.dirname(os.path.abspath(__file__))
# up to agents, up to src
src_path = os.path.abspath(os.path.join(current_dir, '../../../'))
sys.path.append(src_path)
# Also add envs directly because gold_hedge_env does 'from trading_env import...'
sys.path.append(os.path.join(src_path, 'envs'))

# Now we can import envs
from envs.gold_hedge_env import GoldHedgeEnv

def load_data():
    """
    Loads and aligns gold and usd data.
    """
    # Base dir: Codigo/
    base_dir = os.path.dirname(src_path)
    data_dir = os.path.join(base_dir, 'data')
    
    gold_path = os.path.join(data_dir, 'gold_data.csv')
    # Using the extended CSV file with ~6000 rows
    usd_path = os.path.join(data_dir, 'usd_yfinance_2000_2025.csv')
    
    print(f"Loading data from:\n {gold_path}\n {usd_path}")

    # Load Gold Data
    gold_df = pd.read_csv(gold_path, parse_dates=['Date'], index_col='Date')
    
    # Load USD Data (CSV format)
    usd_df = pd.read_csv(usd_path, parse_dates=['Date'], index_col='Date')
    
    # Sort index to ensure chronological order
    usd_df.sort_index(inplace=True)
    
    # Feature Engineering for USD
    # Check if we need to calculate returns/volatility or if they exist
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

def main():
    # 1. Load Data
    gold_df, usd_df = load_data()
    
    # 2. Split Data 70/30
    n_total = len(gold_df)
    train_split_idx = int(n_total * 0.70)
    
    print(f"Total samples: {n_total}")
    print(f"Training samples: {train_split_idx}")
    print(f"Testing samples: {n_total - train_split_idx}")
    
    # 3. Create Environments
    window_size = 10
    
    # Training Environment (Indices: 0 to train_split_idx)
    train_frame_bound = (window_size, train_split_idx)
    
    env_kwargs = {
        'gold_df': gold_df,
        'hedge_df': usd_df,
        'window_size': window_size,
        'frame_bound': train_frame_bound
    }
    
    # Vectorized Environment + Normalization
    # Using 8 envs for faster training
    train_vec_env = make_vec_env(GoldHedgeEnv, n_envs=8, env_kwargs=env_kwargs)
    
    # Wrap with VecNormalize (Norm obs and reward)
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
    # Single env for testing
    test_vec_env = make_vec_env(GoldHedgeEnv, n_envs=1, env_kwargs=test_env_kwargs)
    # Norm obs only, don't norm reward during test for easier interpretation, dont train stats
    test_vec_env = VecNormalize(test_vec_env, norm_obs=True, norm_reward=False, training=False, clip_obs=10.)
    
    # 4. Setup PPO Training
    log_dir = os.path.join(current_dir, "tensorboard_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    model = PPO(
        "MlpPolicy", 
        train_vec_env, 
        verbose=1, 
        tensorboard_log=log_dir,
        learning_rate=0.0003,
        n_steps=2048, 
        batch_size=256
    )
    
    print("Starting PPO Training...")
    
    # Train for a small amount for verification, user can increase later
    # Using the user's value or a smaller one for quick test? 
    # User had 10,000,000. I will reduce it for the initial run to 100,000 to show it works, 
    # then the user can run it for longer. 
    # Actually, I'll stick to a smaller number for *my* verification, but write the file with a reasonable number
    # defined as a constant so they can change it easily.
    TOTAL_TIMESTEPS = 1000000 # Reduced for quick verification run 
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    
    print("Training Finished.")
    
    # Save the model and the normalization stats
    model_save_path = os.path.join(current_dir, "ppo_gold_hedge")
    model.save(model_save_path)
    train_vec_env.save(os.path.join(current_dir, "vec_normalize.pkl"))
    print(f"Model saved to {model_save_path}")

    # 5. Simple Evaluation Loop
    # Sync stats for testing
    test_vec_env.obs_rms = train_vec_env.obs_rms
    
    print("Evaluating on Test Set...")
    obs = test_vec_env.reset()
    done = False
    
    test_log = []
    step = 0
    
    # Run a full episode on the test env
    # Since it's a vector env, we need to handle the array outputs
    # We loop until the environment signals 'dones' (end of episode)
    
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
        
        # Only print every 100 steps to avoid clutter
        if step % 100 == 0:
            print(
                f"[STEP {step:03d}] "
                f"A={action[0]} | "
                f"R={reward[0]:+.5f} | "
                f"V={info.get('total_profit', 0):.4f} "
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
