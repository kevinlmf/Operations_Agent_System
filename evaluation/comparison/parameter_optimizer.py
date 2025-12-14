"""
Parameter Optimizer - 自动寻找最佳参数

为每种方法优化参数，最大化Net Benefit
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from scipy.optimize import minimize, differential_evolution
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluation.comparison.net_benefit_optimizer import NetBenefitOptimizer
from goal.interfaces import InventoryMethod, InventoryState


class ParameterOptimizer:
    """参数优化器 - 为每种方法找到最佳参数"""
    
    def __init__(self,
                 optimizer: NetBenefitOptimizer,
                 test_states: List[InventoryState],
                 test_demands: np.ndarray):
        """
        初始化参数优化器
        
        Args:
            optimizer: NetBenefitOptimizer实例
            test_states: 测试状态
            test_demands: 测试需求
        """
        self.optimizer = optimizer
        self.test_states = test_states
        self.test_demands = test_demands
        self.num_periods = len(test_states)
    
    def optimize_eoq(self, 
                     demand_history: np.ndarray,
                     holding_cost_range: Tuple[float, float] = (0.5, 5.0),
                     ordering_cost_range: Tuple[float, float] = (10.0, 200.0)) -> Dict[str, Any]:
        """
        优化EOQ参数
        
        Returns:
            最佳参数和结果
        """
        from methods.traditional.eoq import EOQMethod
        
        def objective(params):
            h_cost, o_cost = params
            try:
                eoq = EOQMethod(holding_cost=h_cost, ordering_cost=o_cost, lead_time=1)
                eoq.fit(demand_history)
                
                result = self.optimizer.evaluate_method_net_benefit(
                    eoq, self.test_states, self.test_demands, self.num_periods, "EOQ_optimized"
                )
                
                # 最大化Net Benefit = 最小化负Net Benefit
                return -result.net_benefit
            except:
                return 1e10  # 惩罚无效参数
        
        # 使用差分进化算法寻找全局最优
        bounds = [holding_cost_range, ordering_cost_range]
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=50,
            popsize=10,
            atol=1e-6
        )
        
        best_h_cost, best_o_cost = result.x
        
        # 创建最优模型
        best_eoq = EOQMethod(holding_cost=best_h_cost, ordering_cost=best_o_cost, lead_time=1)
        best_eoq.fit(demand_history)
        
        best_result = self.optimizer.evaluate_method_net_benefit(
            best_eoq, self.test_states, self.test_demands, self.num_periods, "EOQ_optimized"
        )
        
        return {
            'method': best_eoq,
            'parameters': {
                'holding_cost': best_h_cost,
                'ordering_cost': best_o_cost,
                'lead_time': 1
            },
            'net_benefit': best_result.net_benefit,
            'result': best_result
        }
    
    def optimize_dqn(self,
                     demand_history: np.ndarray,
                     learning_rate_range: Tuple[float, float] = (0.0001, 0.01),
                     hidden_size_range: Tuple[int, int] = (16, 128)) -> Dict[str, Any]:
        """
        优化DQN参数
        
        Returns:
            最佳参数和结果
        """
        from methods.rl_methods.dqn import DQNInventoryMethod
        
        def objective(params):
            lr, hidden = params
            hidden = int(hidden)
            try:
                dqn = DQNInventoryMethod(
                    state_dim=6,
                    num_actions=21,
                    hidden_sizes=(hidden, hidden),
                    learning_rate=lr,
                    memory_size=5000,
                    batch_size=64
                )
                dqn.fit(demand_history)
                dqn.train_agent(num_episodes=5, fast_mode=True)  # 快速训练用于优化
                
                result = self.optimizer.evaluate_method_net_benefit(
                    dqn, self.test_states, self.test_demands, self.num_periods, "DQN_optimized"
                )
                
                return -result.net_benefit
            except:
                return 1e10
        
        bounds = [learning_rate_range, hidden_size_range]
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=20,  # 减少迭代次数（RL训练慢）
            popsize=5,
            atol=1e-4
        )
        
        best_lr, best_hidden = result.x
        best_hidden = int(best_hidden)
        
        # 创建最优模型（完整训练）
        best_dqn = DQNInventoryMethod(
            state_dim=6,
            num_actions=21,
            hidden_sizes=(best_hidden, best_hidden),
            learning_rate=best_lr,
            memory_size=5000,
            batch_size=64
        )
        best_dqn.fit(demand_history)
        best_dqn.train_agent(num_episodes=10, fast_mode=True)
        
        best_result = self.optimizer.evaluate_method_net_benefit(
            best_dqn, self.test_states, self.test_demands, self.num_periods, "DQN_optimized"
        )
        
        return {
            'method': best_dqn,
            'parameters': {
                'learning_rate': best_lr,
                'hidden_size': best_hidden,
                'num_actions': 21
            },
            'net_benefit': best_result.net_benefit,
            'result': best_result
        }
    
    def optimize_lstm(self,
                      demand_history: np.ndarray,
                      hidden_size_range: Tuple[int, int] = (16, 128),
                      epochs_range: Tuple[int, int] = (5, 30)) -> Dict[str, Any]:
        """
        优化LSTM参数
        
        Returns:
            最佳参数和结果
        """
        from methods.ml_methods.lstm import LSTMInventoryMethod
        
        def objective(params):
            hidden, epochs = params
            hidden = int(hidden)
            epochs = int(epochs)
            try:
                lstm = LSTMInventoryMethod(
                    sequence_length=30,
                    hidden_size=hidden,
                    num_layers=1,
                    epochs=epochs,
                    batch_size=64
                )
                lstm.fit(demand_history)
                
                result = self.optimizer.evaluate_method_net_benefit(
                    lstm, self.test_states, self.test_demands, self.num_periods, "LSTM_optimized"
                )
                
                return -result.net_benefit
            except:
                return 1e10
        
        bounds = [hidden_size_range, epochs_range]
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=20,
            popsize=5,
            atol=1e-4
        )
        
        best_hidden, best_epochs = result.x
        best_hidden = int(best_hidden)
        best_epochs = int(best_epochs)
        
        # 创建最优模型
        best_lstm = LSTMInventoryMethod(
            sequence_length=30,
            hidden_size=best_hidden,
            num_layers=1,
            epochs=best_epochs,
            batch_size=64
        )
        best_lstm.fit(demand_history)
        
        best_result = self.optimizer.evaluate_method_net_benefit(
            best_lstm, self.test_states, self.test_demands, self.num_periods, "LSTM_optimized"
        )
        
        return {
            'method': best_lstm,
            'parameters': {
                'hidden_size': best_hidden,
                'epochs': best_epochs,
                'num_layers': 1
            },
            'net_benefit': best_result.net_benefit,
            'result': best_result
        }




