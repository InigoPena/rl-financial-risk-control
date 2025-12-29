import numpy as np
from trading_env import TradingEnv, Actions, Positions
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym
import os

class GoldHedgeEnv(TradingEnv):
    def __init__(self, gold_df, hedge_df, window_size, frame_bound, render_mode=None):
        self.frame_bound = frame_bound
        self.gold_df = gold_df
        self.hedge_df = hedge_df
        self.allocations = {
            0: [0.0, 0.0, 1.0],  # 100% Cash
            1: [0.25, 0.25, 0.5],# 25% Gold, 25% Hedge, 50% Cash
            2: [0.5, 0.5, 0.0],  # 50% Gold, 50% Hedge
            3: [0.75, 0.25, 0.0], # 75% Gold, 25% Hedge
            4: [1.0, 0.0, 0.0],  # 100% Gold
            5: [0.25, 0.75, 0.0], # 25% Gold, 75% Hedge
            6: [0.0, 1.0, 0.0]   # 100% Hedge
        }


        self.current_gold_w = 0.0
        self.current_hedge_w = 0.0

        super().__init__(gold_df, window_size, render_mode)

        self.trade_fee = 0.001  # Transaction fee of 0.1%
        self.action_space = gym.spaces.Discrete(len(self.allocations)) 
        
        self.state_history = []

    def _process_data(self):
        
        gold_feats = self.gold_df.loc[:, ['RETURN_1D', 'VOLATILITY']].to_numpy()
        hedge_feats = self.hedge_df.loc[:, ['RETURN_1D', 'VOLATILITY']].to_numpy()

        gold_prices = self.gold_df['Close'].to_numpy()
        hedge_prices = self.hedge_df['Close'].to_numpy()

        # [Gold_Ret, Gold_Vol, Hedge_Ret, Hedge_Vol]
        all_signal_features = np.column_stack((gold_feats, hedge_feats))

        # Frame bound
        start, end = self.frame_bound
        self.prices_gold = gold_prices[start-self.window_size:end]
        self.prices_hedge = hedge_prices[start-self.window_size:end]
        signal_features = all_signal_features[start-self.window_size:end]

        return self.prices_gold.astype(np.float32), signal_features.astype(np.float32)
    
    def _get_observation(self):

        obs = self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]

        # Wallet Drawdown
        if not hasattr(self, '_max_profit_so_far'): self._max_profit_so_far = 1.0
        self._max_profit_so_far = max(self._max_profit_so_far, self._total_profit)
        current_portfolio_dd = (self._max_profit_so_far - self._total_profit) / self._max_profit_so_far

        internal_state = np.full((self.window_size, 3), 
                                 [self.current_gold_w, self.current_hedge_w, current_portfolio_dd])
    
        return np.column_stack((obs, internal_state))
    
    def _update_profit(self, action):

        new_gold_w, new_hedge_w, _ = self.allocations[action]

        # Actual tick returns
        ret_gold = (self.prices_gold[self._current_tick] / self.prices_gold[self._current_tick-1]) - 1
        ret_hedge = (self.prices_hedge[self._current_tick] / self.prices_hedge[self._current_tick-1]) - 1

        # Wallet return
        portfolio_return = (self.current_gold_w * ret_gold) + (self.current_hedge_w * ret_hedge)

        if action != getattr(self, '_last_action', None):
            portfolio_return -= self.trade_fee
        
        self._total_profit *= (1 + portfolio_return)
        self.current_gold_w = new_gold_w
        self.current_hedge_w = new_hedge_w
        self._last_action = action
    
    def _calculate_reward(self, action):
        """
        Reward Function optimizez for Risk-Adjusted Returns and Effective Hedging
        """
        # Wallet actual return
        current_val = self._total_profit
        prev_val = getattr(self, '_prev_total_profit', 1.0)
        step_return = np.log(current_val / prev_val) if current_val > 0 else -10.0 # Penalización fuerte por pérdida total
        self._prev_total_profit = current_val

        if not hasattr(self, '_returns_history'):
            self._returns_history = []
            
        self._returns_history.append(step_return)

        # Drawdown Penalty
        if not hasattr(self, '_max_profit_so_far'): 
            self._max_profit_so_far = 1.0
        self._max_profit_so_far = max(self._max_profit_so_far, current_val)
        portfolio_dd = (self._max_profit_so_far - current_val) / self._max_profit_so_far
        dd_penalty = portfolio_dd ** 2 * 2.0

        # Volatility Penalty
        volatility_penalty = 0.0
        if len(self._returns_history) >= 5: # At least 5 data points
            recent_returns = self._returns_history[-20:]
            volatility = np.std(recent_returns)
            volatility_penalty = volatility * 0.5

        # Penalize LOSSES more than reward gains
        if step_return < 0:
            downside_penalty = abs(step_return) * 2.0
        else:
            downside_penalty = 0.0

        # SHARPE-like
        sharpe_bonus = 0.0
        if len(self._returns_history) >= 10:
            recent_returns = np.array(self._returns_history[-30:])
            mean_return = np.mean(recent_returns)
            std_return = np.std(recent_returns)
            if std_return > 0:
                sharpe_ratio = mean_return / std_return
                sharpe_bonus = sharpe_ratio * 0.1

        # BETA Penalty
        beta_penalty = 0.0
        if len(self._returns_history) >= 15:

            lookback = min(30, len(self._returns_history))
            recent_portfolio_returns = np.array(self._returns_history[-lookback:])
            
            start_idx = self._current_tick - lookback + 1
            recent_gold_returns = self.signal_features[start_idx:self._current_tick+1, 0]
            
            if len(recent_portfolio_returns) == len(recent_gold_returns) and len(recent_gold_returns) >= 15:
                # Beta = Cov(Portfolio, Gold) / Var(Gold)
                covariance = np.cov(recent_portfolio_returns, recent_gold_returns)[0, 1]
                gold_variance = np.var(recent_gold_returns)
                
                # Non-zero variance check
                if gold_variance > 1e-6:
                    beta = covariance / gold_variance
                    
                    if beta > 0.7:
                        beta_penalty = (beta - 0.7) * 1.5
                    elif beta > 0.5:
                        beta_penalty = (beta - 0.5) * 0.5

        # VaR Penalty
        var_penalty = 0.0
        if len(self._returns_history) >= 20:
            recent_returns = np.array(self._returns_history[-30:])
            
            var_95 = np.percentile(recent_returns, 5) # VaR at 95% confidence level

            if var_95 < -0.02:
                var_penalty = abs(var_95 + 0.02) * 10.0
            elif var_95 < -0.01:
                var_penalty = abs(var_95 + 0.01) * 3.0

        # Capital Preservation Bonus
        capital_preservation_bonus = 0.0

        if current_val >= 0.95:
            capital_preservation_bonus = 0.002
        elif current_val >= 0.90:
            capital_preservation_bonus = 0.001

        if current_val < 0.85:
            severe_loss_penalty = (0.85 - current_val) * 5.0
        else:
            severe_loss_penalty = 0.0

        reward = (
            step_return * 0.3  # Less sensitive to step returns
            - dd_penalty
            - volatility_penalty
            - downside_penalty
            - beta_penalty
            - var_penalty
            + sharpe_bonus
            + capital_preservation_bonus
            - severe_loss_penalty
        )

        return reward

    def step(self, action):
        self._truncated = False
        self._current_tick += 1

        # PRINT INFO
        print(f"\n{'='*60}")
        print(f"TICK {self._current_tick} / {self._end_tick}")
        print(f"{'='*60}")

        if self._current_tick == self._end_tick:
            self._truncated = True

        # PRINT INFO
        gold_w, hedge_w, cash_w = self.allocations[action]
        print(f"🎯 Acción: {action} → Oro: {gold_w*100:.1f}% | Hedge: {hedge_w*100:.1f}% | Cash: {cash_w*100:.1f}%")
        
        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(action)
        self.history_w.append([self.current_gold_w, self.current_hedge_w])

        # PRINT INFO
        gold_price = self.prices_gold[self._current_tick]
        hedge_price = self.prices_hedge[self._current_tick]
        print(f"💰 Precio Oro: ${gold_price:.2f} | Precio Hedge: ${hedge_price:.2f}")
        
        # PRINT INFO
        print(f"📈 Reward: {step_reward:.6f} | Total Reward: {self._total_reward:.6f}")
        print(f"💼 Valor Cartera: {self._total_profit:.4f} (x{self._total_profit:.2%})")
        
        # PRINT INFO
        portfolio_dd = (self._max_profit_so_far - self._total_profit) / self._max_profit_so_far
        print(f"📉 Drawdown: {portfolio_dd*100:.2f}%")

        trade = False
        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            trade = True

        if trade:
            self._position = self._position.opposite()
            self._last_trade_tick = self._current_tick
            print(f"🔄 Cambio de posición: {self._position}")

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        if self.render_mode == 'human':
            self._render_frame()

        return observation, step_reward, False, self._truncated, info
    
    def reset(self, seed=None, options=None):
        # Reset custom variables
        self._max_profit_so_far = 1.0
        self._prev_total_profit = 1.0
        self._returns_history = []
        self.current_gold_w = 0.0
        self.current_hedge_w = 0.0
        self.history_w = [] 

        return super().reset(seed=seed)

# Esta funcion y el bloque main son solo para pruebas rápidas del entorno

def render_portfolio_evolution(weights_history, title="Evolución de la Cartera"):
    """
    weights_history: Lista o array de listas con el formato [oro_w, hedge_w, cash_w]
    """
    weights_history = np.array(weights_history)
    steps = range(len(weights_history))
    
    oro = weights_history[:, 0]
    hedge = weights_history[:, 1]
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
    plt.show()

if __name__ == "__main__":

    # 1. Cargar los datos
    gold_data = pd.read_csv('Codigo/data/gold_data.csv', index_col=0)
    hedge_data = pd.read_csv('Codigo/data/treasury_data_safepolicy.csv', index_col=0)

    # 2. Convertir el índice a formato fecha (imprescindible)
    gold_data.index = pd.to_datetime(gold_data.index)
    hedge_data.index = pd.to_datetime(hedge_data.index)

    # 3. Seleccionar el rango de fechas para la simulación
    # Puedes usar el formato 'YYYY-MM-DD'
    fecha_inicio = '2020-01-01'
    fecha_fin    = '2020-02-03'

    gold_subset = gold_data.loc[fecha_inicio : fecha_fin].copy()
    hedge_subset = hedge_data.loc[fecha_inicio : fecha_fin].copy()

    # 4. Sincronizar (opcional pero recomendado)
    # Esto asegura que ambos tengan exactamente los mismos días por si falta alguno
    common_index = gold_subset.index.intersection(hedge_subset.index)
    gold_subset = gold_subset.loc[common_index]
    hedge_subset = hedge_subset.loc[common_index]

    # 2. Configurar el entorno
    n_days = len(gold_subset)
    window_size = 10
    frame_bound = (window_size, n_days)
    
    env = GoldHedgeEnv(
        gold_df=gold_subset, 
        hedge_df=hedge_subset, 
        window_size=window_size, 
        frame_bound=frame_bound
    )

    # 3. Ejecutar una simulación con acciones aleatorias
    # Esto sirve para ver si el código "fluye" bien antes de entrenar
    obs, info = env.reset()
    done = False
    
    print("Iniciando simulación de prueba...")
    
    while not done:
        # Elegimos una acción al azar (0 a 6)
        action = env.action_space.sample() 
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    print(f"Simulación terminada.")
    print(f"Profit final: {info['total_profit']:.2f}")

    # 4. Visualizar la evolución de los pesos
    render_portfolio_evolution(env.history_w, title="Prueba de Asignación Aleatoria")