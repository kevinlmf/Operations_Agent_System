"""
Net Benefit Optimizer - Cost-Benefit Analysis Framework

目标：找到最优的库存管理方法，最大化 Net Benefit = Revenue - Total Cost
约束：Total Cost <= Cost Constraint

对比方法：传统方法、ML、RL、DL
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from goal.interfaces import InventoryMethod, InventoryState
from evaluation.comparison.evaluator import EnhancedInventoryEvaluator


@dataclass
class MethodCosts:
    """方法的成本结构"""
    implementation_cost: float  # 实施成本（一次性）
    training_cost: float        # 训练成本（一次性）
    inference_cost_per_period: float  # 每期推理成本
    maintenance_cost_per_period: float  # 每期维护成本


@dataclass
class NetBenefitResult:
    """Net Benefit评估结果"""
    method_name: str
    method_category: str
    
    # Revenue metrics
    total_revenue: float
    units_sold: float
    
    # Cost breakdown
    operational_cost: float      # 运营成本（holding + stockout + ordering）
    implementation_cost: float   # 实施成本
    training_cost: float         # 训练成本
    inference_cost: float        # 推理成本
    maintenance_cost: float      # 维护成本
    total_cost: float            # 总成本
    
    # Benefit metrics
    net_benefit: float           # Net Benefit = Revenue - Total Cost
    roi: float                   # Return on Investment
    cost_benefit_ratio: float    # Revenue / Total Cost
    
    # Constraints
    meets_cost_constraint: bool   # 是否满足成本约束
    cost_constraint: float       # 成本约束值
    
    # Performance metrics
    service_level: float
    inventory_turnover: float
    forecast_accuracy: float


class NetBenefitOptimizer:
    """
    Net Benefit优化器
    
    目标函数：Maximize Net Benefit = Revenue - Total Cost
    约束条件：Total Cost <= Cost Constraint
    """
    
    def __init__(self,
                 unit_price: float = 20.0,
                 unit_cost: float = 10.0,
                 holding_cost: float = 2.0,
                 stockout_cost: float = 10.0,
                 ordering_cost: float = 50.0,
                 cost_constraint: Optional[float] = None,
                 periods_per_year: int = 365):
        """
        初始化Net Benefit优化器
        
        Args:
            unit_price: 单位售价
            unit_cost: 单位成本
            holding_cost: 持有成本（每单位每期）
            stockout_cost: 缺货成本（每单位每期）
            ordering_cost: 订货成本（每次）
            cost_constraint: 成本约束（如果为None则不设约束）
            periods_per_year: 每年期数（用于年化计算）
        """
        self.unit_price = unit_price
        self.unit_cost = unit_cost
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.ordering_cost = ordering_cost
        self.cost_constraint = cost_constraint
        self.periods_per_year = periods_per_year
        
        # 方法成本配置（可根据实际情况调整）
        self.method_costs = {
            'traditional': MethodCosts(
                implementation_cost=1000.0,      # 传统方法实施成本低
                training_cost=0.0,               # 无需训练
                inference_cost_per_period=0.1,   # 推理成本低
                maintenance_cost_per_period=10.0  # 维护成本低
            ),
            'ml': MethodCosts(
                implementation_cost=5000.0,      # ML实施成本中等
                training_cost=2000.0,            # 需要训练
                inference_cost_per_period=0.5,   # 推理成本中等
                maintenance_cost_per_period=50.0  # 维护成本中等
            ),
            'rl': MethodCosts(
                implementation_cost=8000.0,       # RL实施成本较高
                training_cost=5000.0,            # 训练成本高
                inference_cost_per_period=1.0,   # 推理成本较高
                maintenance_cost_per_period=100.0 # 维护成本较高
            ),
            'dl': MethodCosts(
                implementation_cost=10000.0,      # DL实施成本最高
                training_cost=8000.0,            # 训练成本最高
                inference_cost_per_period=2.0,   # 推理成本最高（GPU）
                maintenance_cost_per_period=150.0 # 维护成本最高
            )
        }
        
        # 使用EnhancedInventoryEvaluator进行基础评估
        self.base_evaluator = EnhancedInventoryEvaluator(
            holding_cost=holding_cost,
            stockout_cost=stockout_cost,
            ordering_cost=ordering_cost
        )
    
    def evaluate_method_net_benefit(self,
                                   method: InventoryMethod,
                                   test_states: List[InventoryState],
                                   true_demands: np.ndarray,
                                   num_periods: int,
                                   method_name: str = None) -> NetBenefitResult:
        """
        评估方法的Net Benefit
        
        Args:
            method: 库存管理方法
            test_states: 测试状态列表
            true_demands: 真实需求
            num_periods: 评估期数
            method_name: 方法名称
            
        Returns:
            NetBenefitResult
        """
        method_name = method_name or method.method_name
        method_category = method.category.value if hasattr(method, 'category') else 'unknown'
        
        # 获取方法成本配置
        method_cost_config = self.method_costs.get(method_category, self.method_costs['traditional'])
        
        # 1. 基础评估（获取运营成本和实际销售数据）
        base_results = self.base_evaluator.evaluate_method_comprehensive(
            method, test_states, true_demands, method_name
        )
        
        # 2. 计算Revenue（使用实际销售量）
        # 从detailed_data中获取实际的销售数据
        detailed_data = base_results.get('detailed_data', {})
        if 'true_demands' in detailed_data:
            true_demands_array = detailed_data['true_demands']
            service_levels = detailed_data.get('service_levels', np.ones(len(true_demands_array)))
            
            # 实际销售量计算：从evaluator的模拟逻辑
            # 在evaluator中：units_sold = min(current_inventory, actual_demand)
            # 我们需要重建这个逻辑
            units_sold_list = []
            current_inv = test_states[0].inventory_level if test_states else 50.0
            
            # 获取lead_time
            lead_time = getattr(method, 'lead_time', 0)
            pending_orders = []  # (arrival_period, quantity)
            
            for i, (state, demand) in enumerate(zip(test_states, true_demands_array)):
                # 处理到货订单
                arriving = [qty for period, qty in pending_orders if period == i]
                for qty in arriving:
                    current_inv += qty
                pending_orders = [(p, q) for p, q in pending_orders if p != i]
                
                # 获取推荐动作
                try:
                    action = method.recommend_action(state)
                    order_qty = action.order_quantity if hasattr(action, 'order_quantity') else 0
                except:
                    order_qty = 0
                
                # 下单（考虑lead_time）
                if order_qty > 0:
                    pending_orders.append((i + lead_time, order_qty))
                
                # 满足需求（实际销售量）
                sold = min(current_inv, demand)
                units_sold_list.append(sold)
                
                # 更新库存
                current_inv = max(0, current_inv - demand)
            
            units_sold = np.sum(units_sold_list)
        else:
            # 回退方法：使用service_level估算
            # 如果service_level=1，销售量=需求；否则使用保守估计
            avg_service_level = base_results.get('service_level', 0.9)
            units_sold = np.sum(true_demands) * avg_service_level
        
        total_revenue = units_sold * self.unit_price
        
        # 3. 计算运营成本（total_cost已经是所有periods的总和，不需要再乘以num_periods）
        # 但需要检查total_cost的单位
        operational_cost = base_results['total_cost']  # 已经是总成本，不是每期成本
        
        # 4. 计算实施和训练成本（一次性）
        implementation_cost = method_cost_config.implementation_cost
        training_cost = method_cost_config.training_cost
        
        # 5. 计算推理和维护成本（每期）
        inference_cost = method_cost_config.inference_cost_per_period * num_periods
        maintenance_cost = method_cost_config.maintenance_cost_per_period * num_periods
        
        # 6. 总成本
        total_cost = (operational_cost + implementation_cost + training_cost + 
                     inference_cost + maintenance_cost)
        
        # 7. 计算Net Benefit
        net_benefit = total_revenue - total_cost
        
        # 8. 计算ROI
        total_investment = implementation_cost + training_cost
        roi = ((net_benefit - total_investment) / total_investment * 100) if total_investment > 0 else 0
        
        # 9. Cost-Benefit Ratio
        cost_benefit_ratio = total_revenue / total_cost if total_cost > 0 else 0
        
        # 10. 检查成本约束
        meets_cost_constraint = True
        if self.cost_constraint is not None:
            meets_cost_constraint = total_cost <= self.cost_constraint
        
        return NetBenefitResult(
            method_name=method_name,
            method_category=method_category,
            total_revenue=total_revenue,
            units_sold=units_sold,
            operational_cost=operational_cost,
            implementation_cost=implementation_cost,
            training_cost=training_cost,
            inference_cost=inference_cost,
            maintenance_cost=maintenance_cost,
            total_cost=total_cost,
            net_benefit=net_benefit,
            roi=roi,
            cost_benefit_ratio=cost_benefit_ratio,
            meets_cost_constraint=meets_cost_constraint,
            cost_constraint=self.cost_constraint or float('inf'),
            service_level=base_results.get('service_level', 0.0),
            inventory_turnover=base_results.get('inventory_turnover', 0.0),
            forecast_accuracy=base_results.get('forecast_accuracy', 0.0)
        )
    
    def compare_methods_net_benefit(self,
                                   methods: Dict[str, InventoryMethod],
                                   test_states: List[InventoryState],
                                   true_demands: np.ndarray,
                                   num_periods: Optional[int] = None) -> pd.DataFrame:
        """
        对比所有方法的Net Benefit
        
        Args:
            methods: 方法字典 {name: method}
            test_states: 测试状态列表
            true_demands: 真实需求
            num_periods: 评估期数（如果None则使用test_states长度）
            
        Returns:
            DataFrame包含所有方法的Net Benefit对比结果
        """
        if num_periods is None:
            num_periods = len(test_states)
        
        results = []
        
        for method_name, method in methods.items():
            try:
                result = self.evaluate_method_net_benefit(
                    method, test_states, true_demands, num_periods, method_name
                )
                results.append({
                    'method_name': result.method_name,
                    'method_category': result.method_category,
                    'total_revenue': result.total_revenue,
                    'units_sold': result.units_sold,
                    'operational_cost': result.operational_cost,
                    'implementation_cost': result.implementation_cost,
                    'training_cost': result.training_cost,
                    'inference_cost': result.inference_cost,
                    'maintenance_cost': result.maintenance_cost,
                    'total_cost': result.total_cost,
                    'net_benefit': result.net_benefit,
                    'roi': result.roi,
                    'cost_benefit_ratio': result.cost_benefit_ratio,
                    'meets_cost_constraint': result.meets_cost_constraint,
                    'service_level': result.service_level,
                    'inventory_turnover': result.inventory_turnover,
                    'forecast_accuracy': result.forecast_accuracy
                })
            except Exception as e:
                print(f"❌ 评估 {method_name} 失败: {e}")
                continue
        
        df = pd.DataFrame(results)
        
        # 如果有成本约束，过滤不满足约束的方法
        if self.cost_constraint is not None:
            df = df[df['meets_cost_constraint'] == True]
        
        # 按Net Benefit排序
        df = df.sort_values('net_benefit', ascending=False)
        
        return df
    
    def find_optimal_method(self,
                           methods: Dict[str, InventoryMethod],
                           test_states: List[InventoryState],
                           true_demands: np.ndarray,
                           num_periods: Optional[int] = None) -> Tuple[str, NetBenefitResult]:
        """
        找到最优方法（Net Benefit最大）
        
        Returns:
            (最优方法名, NetBenefitResult)
        """
        df = self.compare_methods_net_benefit(methods, test_states, true_demands, num_periods)
        
        if df.empty:
            raise ValueError("没有满足约束的方法")
        
        best_method_name = df.iloc[0]['method_name']
        best_result = self.evaluate_method_net_benefit(
            methods[best_method_name], test_states, true_demands, 
            num_periods or len(test_states), best_method_name
        )
        
        return best_method_name, best_result
    
    def print_comparison_summary(self, df: pd.DataFrame):
        """打印对比摘要"""
        if df.empty:
            print("❌ 没有可用的对比结果")
            return
        
        print("\n" + "=" * 80)
        print("📊 NET BENEFIT 对比分析")
        print("=" * 80)
        
        print(f"\n目标: Maximize Net Benefit = Revenue - Total Cost")
        if self.cost_constraint:
            print(f"约束: Total Cost <= {self.cost_constraint:,.2f}")
        
        print(f"\n{'排名':<4} {'方法':<20} {'Net Benefit':>15} {'总成本':>15} {'ROI':>10} {'C/B比率':>10}")
        print("-" * 80)
        
        for i, (_, row) in enumerate(df.iterrows(), 1):
            print(f"{i:<4} {row['method_name']:<20} "
                  f"{row['net_benefit']:>15,.2f} "
                  f"{row['total_cost']:>15,.2f} "
                  f"{row['roi']:>10.1f}% "
                  f"{row['cost_benefit_ratio']:>10.2f}")
        
        # 最优方法
        best = df.iloc[0]
        print(f"\n🏆 最优方法: {best['method_name']} ({best['method_category']})")
        print(f"   Net Benefit: ${best['net_benefit']:,.2f}")
        print(f"   总成本: ${best['total_cost']:,.2f}")
        print(f"   ROI: {best['roi']:.1f}%")
        print(f"   Cost-Benefit Ratio: {best['cost_benefit_ratio']:.2f}")
        
        # 成本分解
        print(f"\n💰 成本分解 (最优方法):")
        print(f"   运营成本: ${best['operational_cost']:,.2f}")
        print(f"   实施成本: ${best['implementation_cost']:,.2f}")
        print(f"   训练成本: ${best['training_cost']:,.2f}")
        print(f"   推理成本: ${best['inference_cost']:,.2f}")
        print(f"   维护成本: ${best['maintenance_cost']:,.2f}")

