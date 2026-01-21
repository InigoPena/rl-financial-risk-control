# Reinforcement Learning for Financial Risk Management

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![RL Framework](https://img.shields.io/badge/Framework-Stable--Baselines3-orange.svg)

**Sistema avanzado de gestión de riesgo financiero basado en Aprendizaje por Refuerzo (RL) para trading de oro y activo de cobertura.**


## Descripción del Proyecto

Este proyecto implementa múltiples algoritmos de **Reinforcement Learning** para optimizar estrategias de trading y gestión de riesgo en mercados financieros. El enfoque principal es el trading de oro (Gold) con cobertura mediante instrumentos como el USD y bonos del tesoro.

### Características Principales

- **Múltiples Algoritmos de RL**: Implementación de PPO, DQN, QR-DQN con variantes experimentales
- **Entornos Personalizados**: Entornos de trading basados en Gymnasium con métricas financieras realistas
- **Gestión de Portfolio**: Sistema de asignación dinámica de activos (Gold/Hedge/Cash)
- **Curriculum Learning**: Estrategias de aprendizaje progresivo para mejorar la convergencia
- **Análisis de Datos**: Procesamiento completo de datos históricos con indicadores técnicos

### Caso de Uso

El sistema aprende a tomar decisiones de trading óptimas considerando:
- 📊 **Volatilidad del mercado**
- 💰 **Costos de transacción** (0.1%-0.3%)
- 🎯 **Estrategias de cobertura** (hedging)
- 📈 **Retornos ajustados por riesgo**
- 🔄 **Rebalanceo dinámico de portafolio**

---

## 📂 Estructura del Proyecto

```
rl-financial-risk-control/
│
├── data/                              # Datos financieros históricos
│   ├── gold_data.csv                  # Precios históricos del oro (2015-2025)
│   ├── gold_data_1h.csv               # Datos intradiarios del oro
│   ├── usd_yfinance_2000_2025.csv     # Datos del USD como hedge
│   ├── treasury_data_safepolicy.csv   # Bonos del tesoro (SafePolicy)
│   └── *.parquet                      # Versiones optimizadas
│
├── src/                               # Código fuente principal
│   ├── envs/                          # Entornos de trading
│   │   ├── trading_env.py             # Clase base abstracta
│   │   ├── gold_hedge_env.py          # Entorno Gold + Hedge (v1)
│   │   ├── forex_env.py               # Entorno Forex
│   │   └── stocks_env.py              # Entorno de acciones
│   │
│   ├── agents/                        # Implementaciones de algoritmos RL
│   │   ├── DQN/                       # Deep Q-Network
│   │   │   ├── base_DQN/
│   │   │   │   └── definitive/        # Versión definitiva DQN
│   │   │   │       ├── prueba.py      # Script de entrenamiento
│   │   │   │       ├── best_model_dqn/
│   │   │   │       └── vec_normalize_dqn.pkl
│   │   │   ├── QR_DQN/                # Quantile Regression DQN
│   │   │   │   └── qr_DQN.py          # Implementación distribucional
│   │   │   └── leaky_Relu_prueba/     # Experimentos con activación
│   │   │
│   │   └── ppo/                       # Proximal Policy Optimization
│   │       ├── base_ppo/
│   │       │   └── ppo_definitive/    # Versión definitiva PPO
│   │       │       ├── script_final.py
│   │       │       ├── best_model/
│   │       │       └── vec_normalize.pkl
│   │       └── curriculum_learning/   # Aprendizaje por curriculum
│   │           └── curr_definitive/
│   │               └── prueba.py      # PPO con curriculum
│   │
│   └── data_processing/               # Procesamiento de datos
│       ├── dataExtraction.py          # Descarga datos de Yahoo Finance
│       ├── dataUSD.py                 # Procesamiento USD
│       ├── dataBonos.py               # Procesamiento bonos
│       ├── common_data.py             # Utilidades de alineación
│       └── data_check/                # Herramientas de inspección
│
├── tradenv/                           # Entorno virtual Python
│   └── ...                            # Dependencias instaladas
```

---

## Algoritmos Implementados

### 1. **PPO (Proximal Policy Optimization)**
- **Ubicación**: [`src/agents/ppo/base_ppo/ppo_definitive/`](src/agents/ppo/base_ppo/ppo_definitive/)
- **Características**:
  - Policy clipping (ε=0.2)
  - Learning rate schedule (1e-4 → 5e-6)
  - Arquitectura: [256, 256] con activación ReLU
  - 2M steps de entrenamiento
  - Early stopping basado en recompensa

### 2. **DQN (Deep Q-Network)**
- **Ubicación**: [`src/agents/DQN/base_DQN/definitive/`](src/agents/DQN/base_DQN/definitive/)
- **Características**:
  - Experience replay (150K buffer)
  - Target network (actualización cada 5K steps)
  - ε-greedy exploration (1.0 → 0.05)
  - Double DQN habilitado

### 3. **QR-DQN (Quantile Regression DQN)**
- **Ubicación**: [`src/agents/DQN/QR_DQN/`](src/agents/DQN/QR_DQN/)
- **Características**:
  - Distribución completa de Q-values (50 quantiles)
  - Mayor robustez ante outliers
  - Mejor estimación de incertidumbre
  - Arquitectura distribucional tipo Rainbow

### 4. **Curriculum Learning**
- **Ubicación**: [`src/agents/ppo/curriculum_learning/curr_definitive/`](src/agents/ppo/curriculum_learning/curr_definitive/)
- **Características**:
  - Entrenamiento progresivo por períodos temporales
  - Incremento gradual de dificultad (2008 crisis → 2020 COVID → Período completo)
  - Transfer learning entre fases

---

## Entorno de Trading

### `GoldHedgeEnv`

Entorno para trading de oro con cobertura dinámica, centrado en **gestión activa de riesgo**.

**Espacio de Observación**:
- `[window_size, 7]` :
  - **Gold Return (1D)**: Retorno diario del oro
  - **Gold Volatility**: Desviación estándar rolling (20 días)
  - **Hedge Return (1D)**: Retorno diario del activo de cobertura (USD/Treasury)
  - **Hedge Volatility**: Desviación estándar rolling (20 días)
  - **Current Gold Weight**: Peso actual en oro del portfolio
  - **Current Hedge Weight**: Peso actual en hedge del portfolio
  - **Portfolio Drawdown**: Pérdida desde máximo histórico (gestión de riesgo)

**Espacio de Acción** (7 estrategias discretas):
```python
0: [0.0, 0.0, 1.0]   # 100% Cash
1: [0.25, 0.25, 0.5] # Balanceado
2: [0.5, 0.5, 0.0]   # Equilibrado
3: [0.75, 0.25, 0.0] # Agresivo en Gold
4: [1.0, 0.0, 0.0]   # 100% Gold
5: [0.25, 0.75, 0.0] # Cobertura fuerte
6: [0.0, 1.0, 0.0]   # 100% Hedge
```

**Función de Recompensa**:

La función de recompensa está diseñada para optimizar **retornos ajustados por riesgo** con múltiples componentes:

```python
reward = (
    step_return × 0.3           # Retorno logarítmico ponderado (30%)
    - drawdown_penalty²         # Penalización cuadrática por drawdown
    - volatility_penalty        # Penalización por volatilidad excesiva
    - downside_penalty × 2.0    # Penalización 2x para pérdidas
    - beta_penalty              # Penalización si β > 0.5 respecto al oro
    - var_penalty               # Penalización por VaR(95%) < -2%
    + sharpe_bonus              # Bonificación por Sharpe Ratio alto
    + capital_preservation      # Bonificación si portfolio > 95%
    - severe_loss_penalty × 5.0 # Penalización severa si portfolio < 85%
)
```

**Filosofía de diseño**: La función prioriza **preservación de capital** y **retornos ajustados por riesgo** sobre rentabilidad bruta, alineándose con principios de gestión profesional de riesgos.

---


## 🚀 Uso

### 1. Entrenar un Modelo (PPO)

```bash
cd src/agents/ppo/base_ppo/ppo_definitive
python script_final.py
```

**Configuración principal**:
- Split: 70% train / 30% test
- Window size: 10 días
- Timesteps: 2,000,000
- Normalización de observaciones activada

### 2. Entrenar con DQN

```bash
cd src/agents/DQN/base_DQN/definitive
python prueba.py
```

### 3. Entrenar con QR-DQN (Distribucional)

```bash
cd src/agents/DQN/QR_DQN
python qr_DQN.py
```

### 4. Curriculum Learning

```bash
cd src/agents/ppo/curriculum_learning/curr_definitive
python prueba.py
```

### 5. Procesar Nuevos Datos

```bash
cd src/data_processing
python dataExtraction.py  # Descargar datos de oro
python download_usd_yfinance.py  # Descargar datos USD
python common_data.py  # Verificar alineación
```

---

## Datos

### Features Calculadas

- **RETURN_1D**: Retorno logarítmico diario
- **VOLATILITY**: Desviación estándar rolling (20 días)
- **ATR (Average True Range)**: Volatilidad absoluta
- **RSI**: Relative Strength Index
- **MACD**: Moving Average Convergence Divergence

---

## Experimentos

### Carpetas de Experimentos

- **`other_exp/`**: Experimentos descartados/exploración inicial
- **`definitive/`**: Versiones finales y optimizadas
- **`prueba_*`**: Scripts de testing y validación


### Métricas de Evaluación

- **Sharpe Ratio**: Retorno ajustado por riesgo
- **Maximum Drawdown**: Pérdida máxima desde peak
- **Win Rate**: Porcentaje de operaciones rentables
- **Total Return**: Rentabilidad acumulada
- **Volatility**: Desviación estándar de retornos

---

## Resultados

Los modelos entrenados se guardan en:
- **PPO**: `src/agents/ppo/base_ppo/ppo_definitive/best_model/`
- **DQN**: `src/agents/DQN/base_DQN/definitive/best_model_dqn/`

### Visualizar Training

```bash
tensorboard --logdir src/agents/ppo/base_ppo/ppo_definitive/tensorboard/
```
