# curriculum_stage_params_ppo_get_linear_fn.py
import os
import sys
import io
from contextlib import redirect_stdout
import pandas as pd
import numpy as np

import torch as th
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.utils import get_linear_fn, set_random_seed


# ---------------------------------------------------------------------
# Wrapper: silence noisy env prints
# ---------------------------------------------------------------------
class SilentWrapper(gym.Wrapper):
    def step(self, action):
        with redirect_stdout(io.StringIO()):
            return self.env.step(action)

    def reset(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return self.env.reset(**kwargs)


# ---------------------------------------------------------------------
# Path Setup
# ---------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../../../../"))
sys.path.append(src_path)
sys.path.append(os.path.join(src_path, "envs"))

from envs.gold_hedge_env2 import GoldHedgeEnv


# ---------------------------------------------------------------------
# Data
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

    idx = gold_df.index.intersection(usd_df.index)
    return gold_df.loc[idx], usd_df.loc[idx]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def date_to_idx(df, date_str):
    return max(int(df.index.searchsorted(pd.to_datetime(date_str))), 0)


def frame_bound_from_dates(df, start, end, window):
    s = max(date_to_idx(df, start), window)
    e = max(date_to_idx(df, end), s + 1)
    return (s, e)


# ---------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------
def make_norm_env(gold_df, hedge_df, window_size, frame_bound, n_envs, train, silence, seed):
    def _wrap(env):
        return SilentWrapper(env) if silence else env

    env = make_vec_env(
        GoldHedgeEnv,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=dict(
            gold_df=gold_df,
            hedge_df=hedge_df,
            window_size=window_size,
            frame_bound=frame_bound,
            render_mode=None,
        ),
        wrapper_class=_wrap,
    )

    env = VecNormalize(env, norm_obs=True, norm_reward=train, training=train)

    if not train:
        env.training = False
        env.norm_reward = False

    return env


# ---------------------------------------------------------------------
# PPO factory (uses get_linear_fn)
# ---------------------------------------------------------------------
def build_ppo(env, log_dir, policy_kwargs, stage_params, seed):
    lr_schedule = get_linear_fn(
    stage_params["lr_start"],
    stage_params["lr_end"],
    1.0
)
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        seed=seed,

        learning_rate=lr_schedule,
        n_steps=stage_params["n_steps"],
        batch_size=stage_params["batch_size"],
        n_epochs=stage_params["n_epochs"],
        gamma=stage_params["gamma"],
        gae_lambda=stage_params["gae_lambda"],
        clip_range=stage_params["clip_range"],
        ent_coef=stage_params["ent_coef"],
        vf_coef=stage_params["vf_coef"],
        max_grad_norm=stage_params["max_grad_norm"],

        policy_kwargs=policy_kwargs,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    SEED = 0
    set_random_seed(SEED)

    gold_df, usd_df = load_data()

    window_size = 10
    n_envs = 8

    curriculum = [
        {"name": "trend_2000_2010", "start": "2000-01-01", "end": "2010-12-31", "timesteps": 800_000},
        {"name": "trend_2024_2025", "start": "2024-01-01", "end": "2025-11-14", "timesteps": 400_000},
        {"name": "stable_2020_2024", "start": "2020-01-01", "end": "2024-12-31", "timesteps": 900_000},
        {"name": "volatile_2010_2015", "start": "2010-01-01", "end": "2015-12-31", "timesteps": 1_000_000},
    ]

    test_period = ("2015-01-01", "2020-01-01")

    STAGE_PARAMS = {
        "trend": dict(
            lr_start=7e-4, lr_end=2e-4,
            n_steps=2048, batch_size=512, n_epochs=10,
            gamma=0.997, gae_lambda=0.95,
            clip_range=0.12, ent_coef=0.005,
            vf_coef=0.5, max_grad_norm=0.5,
        ),
        "stable": dict(
            lr_start=5e-4, lr_end=2e-4,
            n_steps=1024, batch_size=512, n_epochs=10,
            gamma=0.995, gae_lambda=0.95,
            clip_range=0.10, ent_coef=0.01,
            vf_coef=0.5, max_grad_norm=0.5,
        ),
        "volatile": dict(
            lr_start=3e-4, lr_end=1e-4,
            n_steps=512, batch_size=512, n_epochs=10,
            gamma=0.990, gae_lambda=0.95,
            clip_range=0.08, ent_coef=0.02,
            vf_coef=0.5, max_grad_norm=0.5,
        ),
    }

    def pick_params(name):
        if "trend" in name:
            return STAGE_PARAMS["trend"]
        if "stable" in name:
            return STAGE_PARAMS["stable"]
        return STAGE_PARAMS["volatile"]

    log_dir = os.path.join(current_dir, "tensorboard_logs_curriculum_stageparams")
    best_dir = os.path.join(current_dir, "best_model_curriculum_stageparams")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    test_bound = frame_bound_from_dates(gold_df, *test_period, window_size)
    test_env = make_norm_env(
        gold_df, usd_df, window_size, test_bound,
        n_envs=1, train=False, silence=True, seed=SEED + 999
    )

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=th.nn.ReLU,
        ortho_init=False,
    )

    stop_cb = StopTrainingOnNoModelImprovement(10, 5, verbose=1)

    eval_cb = EvalCallback(
        test_env,
        best_model_save_path=best_dir,
        log_path=best_dir,
        eval_freq=50_000,
        n_eval_episodes=3,
        deterministic=True,
        callback_after_eval=stop_cb,
    )

    obs_rms, ret_rms = None, None
    carried_params = None
    model = None

    for i, stage in enumerate(curriculum, 1):
        params = pick_params(stage["name"])
        bound = frame_bound_from_dates(gold_df, stage["start"], stage["end"], window_size)

        env = make_norm_env(
            gold_df, usd_df, window_size, bound,
            n_envs=n_envs, train=True, silence=True, seed=SEED + i
        )

        if obs_rms is not None:
            env.obs_rms = obs_rms
            env.ret_rms = ret_rms
            test_env.obs_rms = obs_rms

        model = build_ppo(env, log_dir, policy_kwargs, params, SEED)

        if carried_params is not None:
            model.set_parameters(carried_params, exact_match=True)

        model.learn(stage["timesteps"], callback=eval_cb)

        carried_params = model.get_parameters()
        obs_rms, ret_rms = env.obs_rms, env.ret_rms
        env.close()

    model.save(os.path.join(current_dir, "ppo_curriculum_stageparams_final"))
    print("✅ Curriculum training finished successfully.")

    # ---------------------------------------------------------------------
    # Final Evaluation (Step-by-step Trace)
    # ---------------------------------------------------------------------
    print("\nEvaluating final model on Test Set...")
    test_env.obs_rms = obs_rms
    obs = test_env.reset()
    
    test_log = []
    step = 0
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        
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
        # Handle VecEnv 'done' which might be an array
        if done[0] if isinstance(done, (list, np.ndarray)) else done:
            total_profit = info0.get("total_profit", 0.0)
            print("Episode finished.")
            break

    df_test = pd.DataFrame(test_log)
    trace_path = os.path.join(current_dir, "test_trace_curriculum_stageparams.csv")
    df_test.to_csv(trace_path, index=False)

    print(f"✅ Test trace saved to {trace_path}")
    print(f"📌 Final test episode completed. total_profit={total_profit:.4f}")


if __name__ == "__main__":
    main()
