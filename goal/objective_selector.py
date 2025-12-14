"""
Objective-Based Algorithm Selector

根据库存优化目标自动选择最适合的算法
"""

from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass


class InventoryObjective(Enum):
    """库存优化目标"""
    COST_MINIMIZATION = "cost_minimization"           # 成本最小化
    SERVICE_LEVEL_MAXIMIZATION = "service_level_max"  # 服务水平最大化
    PROFIT_MAXIMIZATION = "profit_maximization"       # 利润最大化
    TURNOVER_MAXIMIZATION = "turnover_maximization"   # 周转率最大化
    RISK_MINIMIZATION = "risk_minimization"           # 风险最小化
    CASH_FLOW_OPTIMIZATION = "cash_flow_optimization" # 现金流优化
    MULTI_OBJECTIVE = "multi_objective"               # 多目标优化


@dataclass
class ProblemCharacteristics:
    """问题特征"""
    objective: InventoryObjective
    has_fixed_order_cost: bool = False
    has_quantity_discount: bool = False
    demand_stability: str = "stable"  # stable, seasonal, random, dynamic
    data_volume: str = "medium"  # low, medium, high
    single_period: bool = False
    multi_product: bool = False
    constraints: List[str] = None


class ObjectiveBasedSelector:
    """
    基于目标的算法选择器
    
    根据优化目标和问题特征，推荐最适合的算法
    """
    
    def __init__(self):
        self.algorithm_mapping = self._build_mapping()
    
    def _build_mapping(self) -> Dict[InventoryObjective, Dict[str, List[str]]]:
        """构建目标-算法映射"""
        return {
            InventoryObjective.COST_MINIMIZATION: {
                "simple": ["EOQ", "ConvexOptimization"],
                "fixed_cost": ["(s,S)Policy", "NonConvexOptimization"],
                "discount": ["NonConvexOptimization", "GeneticAlgorithm"],
                "dynamic": ["DQN", "PPO"]
            },
            InventoryObjective.SERVICE_LEVEL_MAXIMIZATION: {
                "known_distribution": ["SafetyStock", "Newsvendor"],
                "uncertain": ["LSTM", "SafetyStock", "LSTM+SafetyStock"],
                "dynamic": ["DQN", "PPO"]
            },
            InventoryObjective.PROFIT_MAXIMIZATION: {
                "single_period": ["Newsvendor"],
                "multi_period": ["DQN", "PPO"],
                "price_dynamic": ["DQN", "ReinforcementLearning"]
            },
            InventoryObjective.TURNOVER_MAXIMIZATION: {
                "stable": ["EOQ", "JITOptimization"],
                "volatile": ["JITOptimization", "DQN"]
            },
            InventoryObjective.RISK_MINIMIZATION: {
                "demand_risk": ["SafetyStock", "LSTM+SafetyStock"],
                "supply_risk": ["RobustOptimization", "MultiEchelon+RL"]
            },
            InventoryObjective.CASH_FLOW_OPTIMIZATION: {
                "reduce_inventory": ["AggressiveStrategy", "JITOptimization"],
                "optimize_timing": ["JITOptimization", "DQN"]
            },
            InventoryObjective.MULTI_OBJECTIVE: {
                "few_objectives": ["WeightedOptimization", "NonConvexOptimization"],
                "many_objectives": ["GeneticAlgorithm", "ParetoOptimization"]
            }
        }
    
    def recommend_algorithm(
        self,
        objective: InventoryObjective,
        characteristics: ProblemCharacteristics
    ) -> List[Dict[str, Any]]:
        """
        推荐算法
        
        Args:
            objective: 优化目标
            characteristics: 问题特征
        
        Returns:
            推荐的算法列表（按优先级排序）
        """
        recommendations = []
        
        if objective == InventoryObjective.COST_MINIMIZATION:
            if characteristics.has_fixed_order_cost:
                recommendations.append({
                    "algorithm": "NonConvexOptimization",
                    "module": "src.optimization.nonconvex_optimizer",
                    "class": "NonConvexInventoryOptimizer",
                    "method": "optimize_with_fixed_order_cost",
                    "priority": 1,
                    "reason": "处理固定订货成本"
                })
                recommendations.append({
                    "algorithm": "(s,S)Policy",
                    "module": "src.methods.traditional.s_S_policy",
                    "class": "SSPolicyMethod",
                    "priority": 2,
                    "reason": "经典双参数策略"
                })
            elif characteristics.has_quantity_discount:
                recommendations.append({
                    "algorithm": "NonConvexOptimization",
                    "module": "src.optimization.nonconvex_optimizer",
                    "class": "NonConvexInventoryOptimizer",
                    "method": "optimize_with_quantity_discount",
                    "priority": 1,
                    "reason": "处理批量折扣"
                })
            else:
                recommendations.append({
                    "algorithm": "ConvexOptimization",
                    "module": "src.optimization.convex_optimizer",
                    "class": "ConvexInventoryOptimizer",
                    "method": "optimize_linear_cost",
                    "priority": 1,
                    "reason": "线性成本，保证全局最优"
                })
                recommendations.append({
                    "algorithm": "EOQ",
                    "module": "src.methods.traditional.eoq",
                    "class": "EOQMethod",
                    "priority": 2,
                    "reason": "简单快速"
                })
        
        elif objective == InventoryObjective.SERVICE_LEVEL_MAXIMIZATION:
            if characteristics.demand_stability == "dynamic":
                recommendations.append({
                    "algorithm": "DQN",
                    "module": "src.methods.rl_methods.dqn",
                    "class": "DQNInventoryMethod",
                    "priority": 1,
                    "reason": "动态环境自适应"
                })
            elif characteristics.demand_stability in ["seasonal", "random"]:
                recommendations.append({
                    "algorithm": "LSTM+SafetyStock",
                    "module": ["src.methods.ml_methods.lstm", "src.methods.traditional.safety_stock"],
                    "class": ["LSTMInventoryMethod", "SafetyStockMethod"],
                    "priority": 1,
                    "reason": "预测需求+安全库存"
                })
            else:
                recommendations.append({
                    "algorithm": "SafetyStock",
                    "module": "src.methods.traditional.safety_stock",
                    "class": "SafetyStockMethod",
                    "priority": 1,
                    "reason": "简单有效"
                })
        
        elif objective == InventoryObjective.PROFIT_MAXIMIZATION:
            if characteristics.single_period:
                recommendations.append({
                    "algorithm": "Newsvendor",
                    "module": "src.methods.traditional.newsvendor",
                    "class": "NewsvendorMethod",
                    "priority": 1,
                    "reason": "经典单周期利润模型"
                })
            else:
                recommendations.append({
                    "algorithm": "DQN",
                    "module": "src.methods.rl_methods.dqn",
                    "class": "DQNInventoryMethod",
                    "priority": 1,
                    "reason": "多周期利润优化"
                })
        
        elif objective == InventoryObjective.TURNOVER_MAXIMIZATION:
            recommendations.append({
                "algorithm": "JITOptimization",
                "module": "src.cost_optimization.jit_optimizer",
                "class": "JITOptimizer",
                "priority": 1,
                "reason": "及时订货，提高周转率"
            })
            recommendations.append({
                "algorithm": "AggressiveStrategy",
                "module": "src.cost_optimization.inventory_optimizer",
                "class": "DynamicInventoryOptimizer",
                "priority": 2,
                "reason": "激进策略，最低库存"
            })
        
        elif objective == InventoryObjective.RISK_MINIMIZATION:
            recommendations.append({
                "algorithm": "SafetyStock+LSTM",
                "module": ["src.methods.traditional.safety_stock", "src.methods.ml_methods.lstm"],
                "class": ["SafetyStockMethod", "LSTMInventoryMethod"],
                "priority": 1,
                "reason": "预测+高安全库存"
            })
        
        elif objective == InventoryObjective.CASH_FLOW_OPTIMIZATION:
            recommendations.append({
                "algorithm": "AggressiveStrategy",
                "module": "src.cost_optimization.inventory_optimizer",
                "class": "DynamicInventoryOptimizer",
                "priority": 1,
                "reason": "激进策略，减少库存占用"
            })
            recommendations.append({
                "algorithm": "JITOptimization",
                "module": "src.cost_optimization.jit_optimizer",
                "class": "JITOptimizer",
                "priority": 2,
                "reason": "及时订货，优化现金流"
            })
        
        elif objective == InventoryObjective.MULTI_OBJECTIVE:
            recommendations.append({
                "algorithm": "NonConvexOptimization",
                "module": "src.optimization.nonconvex_optimizer",
                "class": "NonConvexInventoryOptimizer",
                "method": "optimize_multi_objective",
                "priority": 1,
                "reason": "多目标优化"
            })
        
        return recommendations
    
    def get_algorithm_info(self, algorithm_name: str) -> Dict[str, Any]:
        """获取算法详细信息"""
        info_map = {
            "EOQ": {
                "description": "经济订货批量模型",
                "complexity": "低",
                "data_requirement": "低",
                "training_time": "无",
                "best_for": "需求稳定，成本最小化"
            },
            "SafetyStock": {
                "description": "安全库存模型",
                "complexity": "低",
                "data_requirement": "中",
                "training_time": "无",
                "best_for": "服务水平最大化，风险最小化"
            },
            "LSTM": {
                "description": "长短期记忆网络",
                "complexity": "高",
                "data_requirement": "高",
                "training_time": "长",
                "best_for": "需求预测，时间序列模式"
            },
            "DQN": {
                "description": "深度Q网络",
                "complexity": "很高",
                "data_requirement": "很高",
                "training_time": "很长",
                "best_for": "动态环境，自适应学习"
            },
            "ConvexOptimization": {
                "description": "凸优化",
                "complexity": "中",
                "data_requirement": "低",
                "training_time": "短",
                "best_for": "线性成本，保证最优"
            },
            "NonConvexOptimization": {
                "description": "非凸优化",
                "complexity": "高",
                "data_requirement": "中",
                "training_time": "长",
                "best_for": "固定成本，批量折扣"
            }
        }
        return info_map.get(algorithm_name, {})


# 使用示例
if __name__ == "__main__":
    selector = ObjectiveBasedSelector()
    
    # 示例1: 成本最小化，有固定成本
    characteristics = ProblemCharacteristics(
        objective=InventoryObjective.COST_MINIMIZATION,
        has_fixed_order_cost=True,
        demand_stability="stable"
    )
    
    recommendations = selector.recommend_algorithm(
        InventoryObjective.COST_MINIMIZATION,
        characteristics
    )
    
    print("推荐算法:")
    for rec in recommendations:
        print(f"  {rec['algorithm']}: {rec['reason']}")


