"""
Evaluate Claude-Enhanced Multi-Agent System vs Traditional Methods

Compares:
1. Traditional OR Methods (MIP, LP, DP separately)
2. Traditional Orchestrated Pipeline (MIP -> LP -> DP)
3. Claude-Enhanced Agent Pipeline (MIP+AI -> LP+AI -> DP+AI)

Metrics:
- Total cost
- Total value
- Execution time
- Decision quality
- Risk-adjusted performance
"""

import numpy as np
import pandas as pd
import time
import os
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Import traditional components
from agent.or_agents import MIPAgent, LPAgent, DPAgent
from agent.orchestrator.orchestrator import OperationsOrchestrator
from agent.orchestrator.definitions import OperationsContext

# Import Claude-enhanced components
from agent.claude_enhanced import (
    ClaudeStrategicAgent,
    ClaudeTacticalAgent,
    ClaudeOperationalAgent,
    IntelligentOrchestrator
)


def generate_test_scenario(scenario_type: str = "medium") -> OperationsContext:
    """Generate test scenarios of different complexity"""
    
    if scenario_type == "small":
        n_facilities = 3
        n_customers = 5
        n_items = 5
    elif scenario_type == "medium":
        n_facilities = 5
        n_customers = 10
        n_items = 10
    else:  # large
        n_facilities = 10
        n_customers = 20
        n_items = 20
    
    np.random.seed(42)
    
    # Generate facility data
    fixed_costs = np.random.uniform(80, 150, n_facilities)
    capacities = np.random.uniform(150, 300, n_facilities)
    
    # Generate customer demands
    demands = np.random.uniform(30, 70, n_customers)
    
    # Generate transport costs (with some structure)
    transport_costs = np.random.uniform(1, 15, (n_facilities, n_customers))
    # Make some facilities naturally better for certain customers
    for i in range(n_facilities):
        nearby_customers = np.random.choice(n_customers, size=n_customers//3, replace=False)
        transport_costs[i, nearby_customers] *= 0.3
    
    # Generate inventory items
    item_values = np.random.uniform(40, 150, n_items).tolist()
    item_weights = np.random.randint(5, 40, n_items).tolist()
    
    return OperationsContext(
        potential_facilities_costs=fixed_costs,
        potential_facilities_capacities=capacities,
        customer_demands=demands,
        transport_costs_full=transport_costs,
        item_values=item_values,
        item_weights=item_weights
    )


def evaluate_traditional_pipeline(context: OperationsContext) -> Dict[str, Any]:
    """Evaluate traditional orchestrated pipeline"""
    
    start_time = time.time()
    
    orchestrator = OperationsOrchestrator()
    
    try:
        result_context = orchestrator.run_pipeline(context)
        
        execution_time = time.time() - start_time
        
        if result_context.is_complete():
            total_cost = (
                result_context.strategic_plan.total_fixed_cost + 
                result_context.tactical_plan.total_transport_cost
            )
            total_value = result_context.operational_plan.total_inventory_value
            
            return {
                "method": "Traditional_Pipeline",
                "success": True,
                "total_cost": total_cost,
                "total_value": total_value,
                "net_benefit": total_value - total_cost,
                "execution_time": execution_time,
                "open_facilities": result_context.strategic_plan.open_facilities_indices,
                "num_facilities": len(result_context.strategic_plan.open_facilities_indices),
                "tool_calls": 0,
                "has_reasoning": False
            }
        else:
            return {
                "method": "Traditional_Pipeline",
                "success": False,
                "error": "Pipeline incomplete"
            }
    except Exception as e:
        return {
            "method": "Traditional_Pipeline",
            "success": False,
            "error": str(e)
        }


def evaluate_claude_enhanced_pipeline(context: OperationsContext, 
                                      use_reasoning: bool = True) -> Dict[str, Any]:
    """Evaluate Claude-enhanced pipeline"""
    
    start_time = time.time()
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    orchestrator = IntelligentOrchestrator(api_key=api_key, use_reasoning=use_reasoning)
    
    method_name = "Claude_Enhanced" if use_reasoning else "Claude_Fallback"
    
    try:
        result_context = orchestrator.run_pipeline(context, use_reasoning=use_reasoning)
        
        execution_time = time.time() - start_time
        collab_result = orchestrator.get_collaboration_result()
        
        if result_context.is_complete():
            total_cost = (
                result_context.strategic_plan.total_fixed_cost + 
                result_context.tactical_plan.total_transport_cost
            )
            total_value = result_context.operational_plan.total_inventory_value
            
            return {
                "method": method_name,
                "success": True,
                "total_cost": total_cost,
                "total_value": total_value,
                "net_benefit": total_value - total_cost,
                "execution_time": execution_time,
                "open_facilities": result_context.strategic_plan.open_facilities_indices,
                "num_facilities": len(result_context.strategic_plan.open_facilities_indices),
                "tool_calls": collab_result.metrics.tool_calls_count if collab_result else 0,
                "has_reasoning": collab_result.metrics.reasoning_generated if collab_result else False,
                "reasoning_summary": collab_result.reasoning_summary[:500] if collab_result else ""
            }
        else:
            return {
                "method": method_name,
                "success": False,
                "error": "Pipeline incomplete"
            }
    except Exception as e:
        import traceback
        return {
            "method": method_name,
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def run_comparative_evaluation():
    """Run comprehensive comparison of all methods"""
    
    print("=" * 70)
    print("🔬 COMPARATIVE EVALUATION: Claude-Enhanced vs Traditional Methods")
    print("=" * 70)
    
    results = []
    
    # Test different scenarios
    scenarios = ["small", "medium"]  # Skip large for faster testing
    
    for scenario_type in scenarios:
        print(f"\n📊 Scenario: {scenario_type.upper()}")
        print("-" * 50)
        
        # Generate fresh context for each method
        context = generate_test_scenario(scenario_type)
        
        # 1. Traditional Pipeline
        print("\n1️⃣ Evaluating Traditional Pipeline...")
        trad_result = evaluate_traditional_pipeline(
            OperationsContext(
                potential_facilities_costs=context.potential_facilities_costs.copy(),
                potential_facilities_capacities=context.potential_facilities_capacities.copy(),
                customer_demands=context.customer_demands.copy(),
                transport_costs_full=context.transport_costs_full.copy(),
                item_values=context.item_values.copy(),
                item_weights=context.item_weights.copy()
            )
        )
        trad_result["scenario"] = scenario_type
        results.append(trad_result)
        
        if trad_result["success"]:
            print(f"   ✅ Cost: ${trad_result['total_cost']:.2f}, Value: ${trad_result['total_value']:.2f}")
            print(f"   ⏱️  Time: {trad_result['execution_time']:.2f}s")
        else:
            print(f"   ❌ Failed: {trad_result.get('error', 'Unknown')}")
        
        # 2. Claude-Enhanced (Fallback mode - no API)
        print("\n2️⃣ Evaluating Claude-Enhanced (Fallback mode)...")
        claude_fallback = evaluate_claude_enhanced_pipeline(
            OperationsContext(
                potential_facilities_costs=context.potential_facilities_costs.copy(),
                potential_facilities_capacities=context.potential_facilities_capacities.copy(),
                customer_demands=context.customer_demands.copy(),
                transport_costs_full=context.transport_costs_full.copy(),
                item_values=context.item_values.copy(),
                item_weights=context.item_weights.copy()
            ),
            use_reasoning=False
        )
        claude_fallback["scenario"] = scenario_type
        results.append(claude_fallback)
        
        if claude_fallback["success"]:
            print(f"   ✅ Cost: ${claude_fallback['total_cost']:.2f}, Value: ${claude_fallback['total_value']:.2f}")
            print(f"   ⏱️  Time: {claude_fallback['execution_time']:.2f}s")
            print(f"   🔧 Tool calls: {claude_fallback['tool_calls']}")
        else:
            print(f"   ❌ Failed: {claude_fallback.get('error', 'Unknown')}")
        
        # 3. Claude-Enhanced (Full reasoning if API available)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            print("\n3️⃣ Evaluating Claude-Enhanced (Full reasoning)...")
            claude_full = evaluate_claude_enhanced_pipeline(
                OperationsContext(
                    potential_facilities_costs=context.potential_facilities_costs.copy(),
                    potential_facilities_capacities=context.potential_facilities_capacities.copy(),
                    customer_demands=context.customer_demands.copy(),
                    transport_costs_full=context.transport_costs_full.copy(),
                    item_values=context.item_values.copy(),
                    item_weights=context.item_weights.copy()
                ),
                use_reasoning=True
            )
            claude_full["scenario"] = scenario_type
            results.append(claude_full)
            
            if claude_full["success"]:
                print(f"   ✅ Cost: ${claude_full['total_cost']:.2f}, Value: ${claude_full['total_value']:.2f}")
                print(f"   ⏱️  Time: {claude_full['execution_time']:.2f}s")
                print(f"   🔧 Tool calls: {claude_full['tool_calls']}")
                print(f"   🧠 Reasoning: {'Yes' if claude_full['has_reasoning'] else 'No'}")
            else:
                print(f"   ❌ Failed: {claude_full.get('error', 'Unknown')}")
        else:
            print("\n3️⃣ Skipping Full reasoning (No API key)")
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Filter successful results
    success_df = df[df['success'] == True].copy()
    
    if len(success_df) > 0:
        print("\n" + "=" * 70)
        print("📊 RESULTS SUMMARY")
        print("=" * 70)
        
        # Summary by method
        summary = success_df.groupby('method').agg({
            'total_cost': 'mean',
            'total_value': 'mean',
            'net_benefit': 'mean',
            'execution_time': 'mean',
            'tool_calls': 'mean'
        }).round(2)
        
        print("\n📈 Average Performance by Method:")
        print(summary.to_string())
        
        # Find best method
        best_method = success_df.loc[success_df['net_benefit'].idxmax()]
        print(f"\n🏆 Best Method (by Net Benefit): {best_method['method']}")
        print(f"   Net Benefit: ${best_method['net_benefit']:.2f}")
        
        # Comparison
        if len(success_df['method'].unique()) >= 2:
            print("\n📊 Method Comparison:")
            for method in success_df['method'].unique():
                method_data = success_df[success_df['method'] == method]
                avg_benefit = method_data['net_benefit'].mean()
                avg_time = method_data['execution_time'].mean()
                print(f"   {method}:")
                print(f"      Avg Net Benefit: ${avg_benefit:.2f}")
                print(f"      Avg Time: {avg_time:.2f}s")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    results_path = "results/agent_comparison_results.csv"
    df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved to: {results_path}")
    
    return df


def demonstrate_agent_skills():
    """Demonstrate the agent skills in action"""
    
    print("\n" + "=" * 70)
    print("🎯 AGENT SKILLS DEMONSTRATION")
    print("=" * 70)
    
    # Test individual agents
    print("\n1️⃣ Strategic Agent Skills:")
    strategic = ClaudeStrategicAgent()
    
    # Analyze costs
    analysis = strategic.skills.execute_tool("analyze_facility_costs", {
        "fixed_costs": [100, 120, 80],
        "transport_costs": [[2, 5, 8], [6, 3, 7], [9, 4, 2]],
        "demand": [50, 60, 40]
    })
    print(f"   Cost Analysis: {analysis.result}")
    
    print("\n2️⃣ Tactical Agent Skills:")
    tactical = ClaudeTacticalAgent()
    
    # Analyze supply-demand
    sd_analysis = tactical.skills.execute_tool("analyze_supply_demand", {
        "supply": [100, 150],
        "demand": [80, 70, 60]
    })
    print(f"   Supply-Demand: {sd_analysis.result}")
    
    print("\n3️⃣ Operational Agent Skills:")
    operational = ClaudeOperationalAgent()
    
    # Analyze items
    item_analysis = operational.skills.execute_tool("analyze_inventory_items", {
        "values": [60, 100, 120],
        "weights": [10, 20, 30]
    })
    print(f"   Item Analysis: {item_analysis.result}")


if __name__ == "__main__":
    print("🚀 Starting Evaluation...")
    
    # First demonstrate agent skills
    demonstrate_agent_skills()
    
    # Then run comparative evaluation
    results_df = run_comparative_evaluation()
    
    print("\n✅ Evaluation Complete!")

