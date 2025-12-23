"""
Dynamic Scenarios Evaluation

Considers:
1. Seasonality
2. Trends  
3. Uncertainty
4. Dynamic programming multi-period decisions
5. Time series forecasting
6. Claude Agent Skills (Agentic AI with Tool Use)
"""

import numpy as np
import pandas as pd
import os

from evaluation.comparison.dynamic_scenario_evaluator import DynamicScenarioEvaluator, ScenarioCharacteristics
from evaluation.comparison.net_benefit_optimizer import NetBenefitOptimizer
from agent.traditional.eoq import EOQMethod
from agent.traditional.safety_stock import SafetyStockMethod
from agent.rl_methods.dqn import DQNInventoryMethod
from agent.ml_methods.lstm import LSTMInventoryMethod
from agent.claude_agent import ClaudeAgentInventoryMethod
from goal.interfaces import InventoryState


def generate_demand_data(days, mean=50, std=10, seasonality=True):
    """Generate synthetic demand data"""
    np.random.seed(42)
    base_demand = np.random.normal(mean, std, days)
    
    if seasonality:
        t = np.arange(days)
        seasonal = 10 * np.sin(2 * np.pi * t / 365.25)
        weekly = 5 * np.sin(2 * np.pi * t / 7)
        base_demand += seasonal + weekly
        
    return np.maximum(base_demand, 0).astype(int)


def main():
    print("🚀 Dynamic Scenarios Evaluation")
    print("=" * 70)

    # 1. Data Generation
    print("\n1️⃣ Generating Dynamic Test Data...")
    train_days = 365
    test_days = 90
    total_days = train_days + test_days
    
    full_demand = generate_demand_data(total_days)
    train_demand = full_demand[:train_days]
    test_demand = full_demand[train_days:]
    
    print(f"  Training data: {len(train_demand)} days")
    print(f"  Test data: {len(test_demand)} days")
    print(f"  Mean demand: {np.mean(full_demand):.1f}")

    # Create test states
    test_states = []
    current_inv = 100
    for i in range(len(test_demand)):
        state = InventoryState(
            inventory_level=current_inv,
            outstanding_orders=0,
            demand_history=train_demand[-30:],
            time_step=train_days + i
        )
        test_states.append(state)

    # 2. Setup Scenario
    print("\n2️⃣ Setting up Dynamic Scenario...")
    scenario = ScenarioCharacteristics(
        has_seasonality=True,
        has_trend=True,
        uncertainty_level=0.2,
        trend_strength=0.05,
        seasonality_amplitude=0.3,
        volatility=0.15
    )

    # 3. Initialize Methods
    print("\n3️⃣ Setting up Methods...")
    methods = {}

    # Traditional Method: EOQ
    print("  📊 Traditional Method: EOQ...")
    try:
        eoq = EOQMethod(holding_cost=2.0, ordering_cost=50.0, lead_time=1)
        eoq.fit(train_demand)
        methods['EOQ'] = eoq
        print(f"    ✅ EOQ fitted")
    except Exception as e:
        print(f"    ❌ EOQ failed: {e}")

    # Traditional Method: Safety Stock
    print("  📊 Traditional Method: Safety Stock...")
    try:
        safety_stock = SafetyStockMethod(service_level=0.95, method='normal')
        safety_stock.fit(train_demand)
        methods['Safety_Stock'] = safety_stock
        print(f"    ✅ Safety Stock fitted")
    except Exception as e:
        print(f"    ❌ Safety Stock failed: {e}")

    # Claude Agent Method (Agentic AI with Tool Use)
    print("  🤖 Claude Agent Method (Agentic AI)...")
    try:
        # Get API key from environment variable
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        claude_agent = ClaudeAgentInventoryMethod(
            api_key=api_key,
            model="claude-sonnet-4-20250514",
            holding_cost=2.0,
            ordering_cost=50.0,
            stockout_cost=10.0,
            lead_time=1,
            service_level=0.95,
            use_extended_thinking=True,
            max_tool_iterations=5,
            fallback_mode=True  # Use local tools if API unavailable
        )
        claude_agent.fit(train_demand)
        methods['Claude_Agent'] = claude_agent
        
        # Show agent status
        params = claude_agent.get_parameters()
        api_status = "API Connected" if params.get("api_available") else "Fallback Mode (Local Tools)"
        print(f"    ✅ Claude Agent initialized - {api_status}")
        print(f"       Demand Pattern: {params.get('demand_stats', {}).get('demand_pattern', 'analyzing...')}")
    except Exception as e:
        print(f"    ❌ Claude Agent failed: {e}")

    # ML Method: LSTM
    print("  🤖 ML Method: LSTM (Time Series)...")
    try:
        import jax
        devices = jax.devices()
        device_type = devices[0].device_kind if devices else "CPU"
        print(f"    🔧 JAX Device: {device_type}")
        
        lstm = LSTMInventoryMethod(
            sequence_length=30,
            hidden_size=32,
            num_layers=1,
            epochs=10,
            batch_size=64
        )
        lstm.fit(train_demand)
        methods['LSTM'] = lstm
        print(f"    ✅ LSTM trained")
    except Exception as e:
        print(f"    ❌ LSTM failed: {e}")

    # RL Method: DQN
    print("  🎮 RL Method: DQN...")
    try:
        import jax
        devices = jax.devices()
        device_type = devices[0].device_kind if devices else "CPU"
        print(f"    🔧 JAX Device: {device_type}")
        
        dqn = DQNInventoryMethod(
            state_dim=6,
            num_actions=21,
            hidden_sizes=(32, 32),
            learning_rate=0.001,
            memory_size=5000,
            batch_size=64
        )
        dqn.fit(train_demand)
        dqn.train_agent(num_episodes=10, fast_mode=True)
        methods['DQN'] = dqn
        print(f"    ✅ DQN trained")
    except Exception as e:
        print(f"    ❌ DQN failed: {e}")

    if not methods:
        print("\n❌ No methods available for evaluation")
        return

    # 4. Dynamic Scenario Evaluation
    print(f"\n4️⃣ Running Dynamic Scenario Analysis...")
    print(f"   Comparing {len(methods)} methods: {list(methods.keys())}")
    
    base_optimizer = NetBenefitOptimizer(
        unit_price=20.0,
        unit_cost=10.0,
        holding_cost=2.0,
        stockout_cost=10.0,
        ordering_cost=50.0,
        cost_constraint=None
    )
    
    dynamic_evaluator = DynamicScenarioEvaluator(
        base_optimizer=base_optimizer,
        scenario=scenario
    )
    
    results_df = dynamic_evaluator.compare_methods_dynamic(
        methods,
        train_demand,
        test_states,
        test_demand,
        num_scenarios=10
    )

    # 5. Reporting
    print("\n" + "=" * 70)
    print("📊 DYNAMIC SCENARIOS COMPARISON RESULTS")
    print("=" * 70)
    
    if results_df.empty:
        print("❌ No results available")
        return
    
    best = results_df.iloc[0]
    print(f"\n🏆 Best Method (Risk-Adjusted): {best['method_name']}")
    print(f"   Expected Net Benefit: ${best.get('expected_net_benefit', 0):,.2f}")
    print(f"   Risk: ${best.get('risk', 0):,.2f}")
    print(f"   Risk-Adjusted Net Benefit: ${best.get('risk_adjusted_net_benefit', 0):,.2f}")
    
    # Save results
    results_path = "results/dynamic_scenarios_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved to: {results_path}")
    
    # Create visualization
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Dynamic Scenarios Analysis', fontsize=16)
        
        methods_list = results_df['method_name'].tolist()
        
        ax1 = axes[0, 0]
        expected_nb = results_df['expected_net_benefit'].tolist()
        risk = results_df['risk'].tolist()
        ax1.scatter(risk, expected_nb, s=100, alpha=0.7)
        for i, method in enumerate(methods_list):
            ax1.annotate(method, (risk[i], expected_nb[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        ax1.set_xlabel('Risk ($)')
        ax1.set_ylabel('Expected Net Benefit ($)')
        ax1.set_title('Risk-Return Trade-off')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        risk_adj = results_df['risk_adjusted_net_benefit'].tolist()
        colors = ['green' if x > 0 else 'red' for x in risk_adj]
        ax2.barh(methods_list, risk_adj, color=colors)
        ax2.set_xlabel('Risk-Adjusted Net Benefit ($)')
        ax2.set_title('Risk-Adjusted Performance')
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        forecast_acc = results_df['forecast_accuracy'].tolist()
        ax3.barh(methods_list, forecast_acc, color='blue')
        ax3.set_xlabel('Forecast Accuracy')
        ax3.set_title('Time Series Forecast Accuracy')
        ax3.set_xlim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        uncertainty = results_df['uncertainty_level'].tolist()
        ax4.scatter(uncertainty, forecast_acc, s=100, alpha=0.7, color='purple')
        for i, method in enumerate(methods_list):
            ax4.annotate(method, (uncertainty[i], forecast_acc[i]), xytext=(5, 5),
                        textcoords='offset points', fontsize=8)
        ax4.set_xlabel('Uncertainty Level')
        ax4.set_ylabel('Forecast Accuracy')
        ax4.set_title('Uncertainty vs Forecast Accuracy')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = "results/dynamic_scenarios_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")

    print("\n" + "=" * 70)
    print("✅ Dynamic Scenarios Evaluation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
