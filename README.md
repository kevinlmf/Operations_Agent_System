# Operations Agent System · AI-Driven Inventory Optimization Platform

**Multi-Agent Decision System** powered by Reinforcement Learning and Operations Research, combining dynamic scenario evaluation, net benefit optimization, and hierarchical planning to deliver automated inventory management for supply chain operations.

---

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![JAX](https://img.shields.io/badge/JAX-Accelerated-orange.svg)](https://github.com/google/jax)

---

## Project Positioning

**AI-Driven Inventory Optimization** - A research platform that combines:

- **Operations Research**: Multi-level optimization (MIP/LP/DP), EOQ, Safety Stock, dynamic programming
- **AI/ML Engineering**: Multi-agent RL (DQN, SAC), deep learning (JAX/Flax), LSTM forecasting
- **Research Innovation**: Goal-oriented planning, net benefit optimization, regime-adaptive systems

## Key Features

### 1. Unified Method Interface

- **Traditional Methods**: EOQ, Safety Stock, (s,S) Policy
- **ML Methods**: LSTM, Transformer for demand forecasting
- **RL Methods**: DQN, Multi-Agent RL for adaptive decisions
- **Action Composition**: Unified `InventoryMethod` interface

### 2. Hierarchical Multi-Agent System

- **Strategic Agent (MIP)**: Facility location, network design
- **Tactical Agent (LP)**: Inventory allocation, capacity planning
- **Operational Agent (DP)**: Daily ordering, replenishment

### 3. Net Benefit Optimization

- **Objective**: Maximize Net Benefit = Revenue - Total Cost
- **Cost Components**: Holding, stockout, ordering, implementation
- **ROI Analysis**: Return on investment tracking

### 4. Dynamic Scenario Evaluation

- **Seasonality Detection**: Annual and weekly patterns
- **Trend Analysis**: Upward, downward, stable trends
- **Uncertainty Quantification**: CV coefficient for volatility
- **Monte Carlo Simulation**: Multi-scenario risk assessment

### 5. Risk-Adjusted Performance

- **Expected Net Benefit**: Mean across scenarios
- **Risk Measurement**: Standard deviation
- **Risk-Adjusted Metric**: Expected Return - 0.5 × Risk


## Installation

```bash
# Clone the repository
git clone https://github.com/kevinlmf/Operations_Agent_System
cd Operations_Agent_System

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run Dynamic Scenario Evaluation

Evaluate methods across dynamic scenarios with seasonality, trends, and uncertainty:

```bash
python evaluate_dynamic_scenarios.py
```

### Run Net Benefit Optimization

Find optimal method maximizing Net Benefit = Revenue - Total Cost:

```bash
python evaluate_net_benefit.py
```

### Optimize RL/DL to Beat Baseline

Automatically optimize RL/DL parameters to outperform traditional methods:

```bash
python optimize_for_dynamic.py
```

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Goal Layer                                                   │
│  Input: U(net_benefit, cost, risk, service_level)            │
│  Output: Goal directive (order quantities, reorder points)   │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│  Scenario Detector + Dynamic Evaluator                       │
│  Seasonality | Trend | Uncertainty Detection                 │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│  Net Benefit Optimizer                                        │
│  Maximize: Revenue - (Holding + Stockout + Ordering + Impl)  │
└────────────────────┬─────────────────────────────────────────┘
                     │
      ┌──────────────┼──────────────┐
      │              │              │
┌─────▼─────┐  ┌─────▼─────┐  ┌────▼──────┐
│ Strategic │  │ Tactical  │  │Operational│
│   Agent   │  │   Agent   │  │   Agent   │
│   (MIP)   │  │   (LP)    │  │   (DP)    │
└─────┬─────┘  └─────┬─────┘  └────┬──────┘
      │              │              │
      └──────────────┼──────────────┘
                     │
      ┌──────────────▼──────────────┐
      │  Method Selector             │
      │  Traditional | ML | RL       │
      │  EOQ | LSTM | DQN | SAC     │
      └──────────────────────────────┘
```

## Project Structure

```
Operations_Agent_System/
├── goal/                          # Goal definitions & interfaces
│   ├── interfaces.py             # InventoryMethod interface
│   ├── objective_selector.py     # Goal selection logic
│   └── mathematical_framework.py # Mathematical models
│
├── agent/                         # All agent implementations
│   ├── traditional/              # Traditional methods
│   │   ├── eoq.py               # Economic Order Quantity
│   │   ├── safety_stock.py      # Safety Stock method
│   │   └── s_S_policy.py        # (s,S) Policy
│   ├── ml_methods/               # ML methods
│   │   ├── lstm.py              # LSTM forecasting
│   │   └── transformer.py       # Transformer model
│   ├── rl_methods/               # RL methods
│   │   ├── dqn.py               # Deep Q-Network
│   │   └── multi_agent.py       # Multi-agent RL
│   ├── orchestrator/             # Multi-agent orchestration
│   │   ├── orchestrator.py      # Main orchestrator
│   │   └── definitions.py       # Context definitions
│   ├── environment/              # RL environment
│   │   └── env.py               # Gym-compatible environment
│   └── or_optimization/          # OR optimizers
│       ├── linear_programming.py
│       ├── mixed_integer_programming.py
│       └── dynamic_programming.py
│
├── evaluation/                    # Evaluation framework
│   ├── comparison/               # Method comparison
│   │   ├── evaluator.py         # Basic evaluator
│   │   ├── net_benefit_optimizer.py    # Net Benefit optimization
│   │   ├── dynamic_scenario_evaluator.py  # Dynamic evaluation
│   │   ├── dynamic_optimizer.py # RL/DL optimizer
│   │   └── parameter_optimizer.py # Parameter tuning
│   └── risk_management/          # Risk control
│       ├── anomaly_detector.py
│       └── contingency_planner.py
│
├── results/                       # Output folder (CSV, PNG)
│
├── evaluate_dynamic_scenarios.py  # Dynamic scenario evaluation
├── evaluate_net_benefit.py        # Net Benefit optimization
├── evaluate_system.py             # Basic system evaluation
├── optimize_for_dynamic.py        # RL/DL optimization
└── README.md
```


## Execution Flow

```
1. Data Generation / Loading
   ↓
2. Scenario Detection (Seasonality, Trend, Uncertainty)
   ↓
3. Method Training (Traditional, ML, RL)
   ↓
4. Net Benefit Evaluation
   ↓
5. Risk-Adjusted Comparison
   ↓
6. Optimal Method Selection
```

## Performance Optimization

- **JAX Acceleration**: All ML/RL methods use JAX/Flax
- **Fast Mode**: `fast_mode=True` reduces training time
- **JIT Compilation**: Key functions use `@jit` decorator

## Experimental Results

Based on comprehensive evaluations across multiple scenarios with 90-day test periods and 365-day training data:

### Net Benefit Analysis

| Method | Category | Net Benefit | ROI | Service Level | Forecast Acc |
|--------|----------|-------------|-----|---------------|--------------|
| **Safety Stock** | Traditional | **$78,452.64** | **7,745%** | **76.67%** | 81.03% |
| EOQ | Traditional | -$11,427.55 | -1,243% | 44.44% | 81.03% |
| LSTM | ML | -$59,156.53 | -945% | 1.11% | 78.87% |
| DQN | RL | -$74,092.00 | -670% | 1.11% | 82.28% |

### Cost Breakdown

| Method | Operational Cost | Implementation | Training | Inference | Maintenance | Total Cost |
|--------|------------------|----------------|----------|-----------|-------------|------------|
| Safety Stock | $15,498.36 | $1,000 | $0 | $9 | $900 | $17,407.36 |
| EOQ | $11,518.55 | $1,000 | $0 | $9 | $900 | $13,427.55 |
| LSTM | $49,611.53 | $5,000 | $2,000 | $45 | $4,500 | $61,156.53 |
| DQN | $54,002.00 | $8,000 | $5,000 | $90 | $9,000 | $76,092.00 |

### Dynamic Scenario Performance

Evaluation under dynamic conditions with seasonality, trends, and 20% uncertainty:

- **Seasonality Detection**: Successfully identified annual (365.25-day) and weekly (7-day) patterns
- **Trend Adaptation**: Detected upward/downward trends with 5% strength
- **Uncertainty Handling**: Monte Carlo simulation with 10 scenarios for risk assessment
- **Risk-Adjusted Metric**: Expected Return - 0.5 × Risk for conservative optimization

### Key Findings

1. **Traditional Methods Dominate**: Safety Stock achieves highest Net Benefit with proper service level management
2. **RL/ML Training Gap**: DQN and LSTM require more training episodes to compete with traditional baselines
3. **Cost-Benefit Trade-off**: Higher implementation costs of ML/RL methods not justified by current performance
4. **Forecast Accuracy**: All methods achieve ~80% forecast accuracy, but action decisions vary significantly
5. **Service Level Critical**: High service level (76.67%) directly correlates with Net Benefit maximization

## Future Work: Making ML/DL Beat Traditional Methods

The current results show traditional methods (Safety Stock) significantly outperforming ML/DL approaches. Here's the roadmap to close this gap:

### 1. Training Scale & Stability

- **Increase Episodes**: Scale from 10 → 500+ episodes for proper policy convergence
- **Curriculum Learning**: Start with stable demand, progressively add seasonality and uncertainty
- **Reward Shaping**: Design rewards that explicitly penalize stockouts and incentivize service levels
- **Pre-training**: Use imitation learning to warm-start RL agents from Safety Stock policy

### 2. State Representation & Feature Engineering

- **Temporal Features**: Add day-of-week, month-of-year, holiday indicators to capture seasonality
- **Trend Signals**: Include moving averages (7-day, 30-day) and momentum indicators
- **Inventory Context**: Encode days-of-supply, stockout history, and order pipeline status
- **Attention Mechanism**: Let LSTM/Transformer learn which historical patterns matter most

### 3. Action Space & Policy Design

- **Continuous Actions**: Replace discrete order quantities with continuous policy (SAC/PPO)
- **Action Bounds**: Constrain actions to feasible ranges based on EOQ/Safety Stock baselines
- **Hybrid Policy**: Combine OR-computed baseline with RL-learned adjustments (residual RL)
- **Multi-step Planning**: Use model-based RL to plan ahead during high-uncertainty periods

### 4. Objective Alignment

- **Service Level Constraint**: Add hard constraint for minimum 95% service level
- **Cost-Aware Reward**: Include holding/stockout costs directly in reward function
- **Risk-Sensitive RL**: Use CVaR or worst-case optimization for robust policies
- **Multi-Objective**: Pareto optimization balancing cost, service, and inventory turnover

### 5. Evaluation & Deployment

- **Longer Test Horizons**: Evaluate over 365+ days to capture full seasonal cycles
- **Out-of-Distribution Testing**: Test on demand patterns not seen during training
- **Ensemble Methods**: Combine predictions from LSTM + DQN + Traditional for robustness
- **Online Adaptation**: Implement continuous learning to adapt to demand distribution shifts



## License

This project is licensed under the MIT License.

## Disclaimer

 **Important**: This project is provided for educational, academic research, and learning purposes only. The inventory decisions generated by this system should be validated by domain experts before implementation in production environments.

---

May our lives keep optimizing, like finding balance in every step😊
