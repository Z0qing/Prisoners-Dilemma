import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
from prisoners_dilemma_tournament import strategies, strategy_tournament, simulate_rounds

# 创建保存图片的目录
if not os.path.exists('parameter_study'):
    os.makedirs('parameter_study')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 策略名称列表
strategy_names = [s.__name__ for s in strategies]

# 全局收益矩阵（与第一个文件相同）
PAYOFF_MATRIX = {
    ('C', 'C'): (3, 3),
    ('C', 'D'): (0, 5),
    ('D', 'C'): (5, 0),
    ('D', 'D'): (1, 1)
}

def study_rounds_effect():
    """研究博弈轮数对策略表现的影响"""
    print("="*50)
    print("研究博弈轮数对策略表现的影响")
    print("="*50)
    
    # 定义不同的博弈轮数
    rounds_list = [5,10,15,20]
    
    # 存储每个策略在不同轮数下的平均得分
    strategy_scores = {name: [] for name in strategy_names}
    
    for rounds in rounds_list:
        print(f"\n正在运行 {rounds} 轮博弈...")
        _, total_scores = strategy_tournament(
            strategies,
            rounds=rounds,
            repetitions=5,  # 减少重复次数以加快速度
            noise_level=0.05
        )
        
        # 计算每个策略的平均每场得分
        n = len(strategies)
        for i, name in enumerate(strategy_names):
            avg_score = total_scores[i] / (2 * n)  # 平均每场得分
            strategy_scores[name].append(avg_score)
    
    # 可视化结果
    plt.figure(figsize=(12, 8))
    for name, scores in strategy_scores.items():
        plt.plot(rounds_list, scores, 'o-', label=name)
    
    plt.xlabel('博弈轮数', fontsize=12)
    plt.ylabel('平均每场得分', fontsize=12)
    plt.title('博弈轮数对策略表现的影响', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"parameter_study/rounds_effect_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"保存图表: {filename}")
    
    return strategy_scores

def study_repetitions_effect():
    """研究重复次数对结果稳定性的影响"""
    print("\n" + "="*50)
    print("研究重复次数对结果稳定性的影响")
    print("="*50)
    
    # 定义不同的重复次数
    repetitions_list = [1, 3, 5, 10, 20, 50]
    
    # 存储顶级策略的得分变化
    top_strategy = 'generous_tft'  # 选择一个典型策略
    top_strategy_index = strategy_names.index(top_strategy)
    strategy_scores = {name: [] for name in strategy_names}
    score_variances = []
    
    for reps in repetitions_list:
        print(f"\n正在运行 {reps} 次重复...")
        _, total_scores = strategy_tournament(
            strategies,
            rounds=200,
            repetitions=reps,
            noise_level=0.05
        )
        
        # 计算每个策略的平均每场得分
        n = len(strategies)
        for i, name in enumerate(strategy_names):
            avg_score = total_scores[i] / (2 * n)
            strategy_scores[name].append(avg_score)
        
        # 记录顶级策略得分的标准差
        score_variances.append(np.std([s for i, s in enumerate(total_scores) if strategy_names[i] == top_strategy]))
    
    # 可视化得分变化
    plt.figure(figsize=(12, 8))
    for name, scores in strategy_scores.items():
        plt.plot(repetitions_list, scores, 'o-', label=name)
    
    plt.xlabel('重复次数', fontsize=12)
    plt.ylabel('平均每场得分', fontsize=12)
    plt.title('重复次数对策略得分的影响', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"parameter_study/repetitions_effect_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"保存图表: {filename}")
    
    # 可视化方差变化
    plt.figure(figsize=(10, 6))
    plt.plot(repetitions_list, score_variances, 'ro-')
    plt.xlabel('重复次数', fontsize=12)
    plt.ylabel(f'{top_strategy}得分的标准差', fontsize=12)
    plt.title('重复次数对结果稳定性的影响', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    filename = f"parameter_study/repetitions_variance_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"保存方差图表: {filename}")
    
    return strategy_scores

def study_noise_effect():
    """研究噪声水平对策略表现的影响"""
    print("\n" + "="*50)
    print("研究噪声水平对策略表现的影响")
    print("="*50)
    
    # 定义不同的噪声水平
    noise_levels = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
    
    # 存储每个策略在不同噪声水平下的平均得分
    strategy_scores = {name: [] for name in strategy_names}
    
    for noise in noise_levels:
        print(f"\n正在运行噪声水平 {noise*100}%...")
        _, total_scores = strategy_tournament(
            strategies,
            rounds=200,
            repetitions=5,  # 减少重复次数以加快速度
            noise_level=noise
        )
        
        # 计算每个策略的平均每场得分
        n = len(strategies)
        for i, name in enumerate(strategy_names):
            avg_score = total_scores[i] / (2 * n)
            strategy_scores[name].append(avg_score)
    
    # 可视化结果
    plt.figure(figsize=(12, 8))
    for name, scores in strategy_scores.items():
        plt.plot(noise_levels, scores, 'o-', label=name)
    
    plt.xlabel('噪声水平', fontsize=12)
    plt.ylabel('平均每场得分', fontsize=12)
    plt.title('噪声水平对策略表现的影响', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"parameter_study/noise_effect_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"保存图表: {filename}")
    
    return strategy_scores

def study_forgiveness_effect():
    """研究宽容概率对策略表现的影响"""
    print("\n" + "="*50)
    print("研究宽容概率对策略表现的影响")
    print("="*50)
    
    # 定义不同的宽容概率
    forgiveness_probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # 存储宽容TFT策略在不同宽容概率下的平均得分
    generous_tft_scores = []
    
    for prob in forgiveness_probs:
        print(f"\n正在运行宽容概率 {prob}...")
        
        # 创建自定义宽容策略
        def custom_generous_tft(history, **kwargs):
            """自定义宽容策略"""
            if not history:
                return 'C'
            last_opponent_action = history[-1][1]
            
            if last_opponent_action == 'D':
                if np.random.random() < prob:
                    return 'C'
            return last_opponent_action
        
        custom_generous_tft.__name__ = f"generous_tft_{int(prob*100)}"
        
        # 替换策略列表中的宽容策略
        custom_strategies = strategies.copy()
        for i, s in enumerate(custom_strategies):
            if s.__name__ == 'generous_tft':
                custom_strategies[i] = custom_generous_tft
        
        # 运行锦标赛
        _, total_scores = strategy_tournament(
            custom_strategies,
            rounds=200,
            repetitions=5,
            noise_level=0.05
        )
        
        # 找到宽容策略的得分
        for i, s in enumerate(custom_strategies):
            if s.__name__.startswith('generous_tft'):
                avg_score = total_scores[i] / (2 * len(custom_strategies))
                generous_tft_scores.append(avg_score)
                break
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.plot(forgiveness_probs, generous_tft_scores, 'bo-')
    plt.xlabel('宽容概率', fontsize=12)
    plt.ylabel('平均每场得分', fontsize=12)
    plt.title('宽容概率对Generous TFT策略表现的影响', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"parameter_study/forgiveness_effect_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"保存图表: {filename}")
    
    return generous_tft_scores

def study_payoff_matrix_effect():
    """研究收益矩阵对策略表现的影响"""
    print("\n" + "="*50)
    print("研究收益矩阵对策略表现的影响")
    print("="*50)
    
    # 定义不同的收益矩阵
    payoff_matrices = [
        # 标准囚徒困境
        {
            ('C', 'C'): (3, 3),
            ('C', 'D'): (0, 5),
            ('D', 'C'): (5, 0),
            ('D', 'D'): (1, 1)
        },
        # 高背叛诱惑
        {
            ('C', 'C'): (3, 3),
            ('C', 'D'): (0, 7),
            ('D', 'C'): (7, 0),
            ('D', 'D'): (1, 1)
        },
        # 低惩罚
        {
            ('C', 'C'): (3, 3),
            ('C', 'D'): (0, 5),
            ('D', 'C'): (5, 0),
            ('D', 'D'): (2, 2)
        },
        # 高合作收益
        {
            ('C', 'C'): (5, 5),
            ('C', 'D'): (0, 3),
            ('D', 'C'): (3, 0),
            ('D', 'D'): (1, 1)
        }
    ]
    
    matrix_names = ["标准", "高背叛诱惑", "低惩罚", "高合作收益"]
    
    # 存储每个策略在不同收益矩阵下的平均得分
    strategy_scores = {name: [] for name in strategy_names}
    
    for i, matrix in enumerate(payoff_matrices):
        print(f"\n正在运行收益矩阵: {matrix_names[i]}...")
        
        # 修改全局收益矩阵
        global PAYOFF_MATRIX
        original_matrix = PAYOFF_MATRIX
        PAYOFF_MATRIX = matrix
        
        # 运行锦标赛
        _, total_scores = strategy_tournament(
            strategies,
            rounds=200,
            repetitions=5,
            noise_level=0.05
        )
        
        # 恢复原始收益矩阵
        PAYOFF_MATRIX = original_matrix
        
        # 计算每个策略的平均每场得分
        n = len(strategies)
        for j, name in enumerate(strategy_names):
            avg_score = total_scores[j] / (2 * n)
            strategy_scores[name].append(avg_score)
    
    # 可视化结果
    plt.figure(figsize=(14, 8))
    x = np.arange(len(matrix_names))
    width = 0.08  # 柱状图宽度
    
    for i, (name, scores) in enumerate(strategy_scores.items()):
        plt.bar(x + i*width, scores, width, label=name)
    
    plt.xlabel('收益矩阵类型', fontsize=12)
    plt.ylabel('平均每场得分', fontsize=12)
    plt.title('收益矩阵对策略表现的影响', fontsize=14)
    plt.xticks(x + width*len(strategies)/2, matrix_names)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"parameter_study/payoff_matrix_effect_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"保存图表: {filename}")
    
    return strategy_scores

if __name__ == "__main__":
    # 运行所有参数研究
    rounds_results = study_rounds_effect()
    reps_results = study_repetitions_effect()
    noise_results = study_noise_effect()
    forgiveness_results = study_forgiveness_effect()
    payoff_results = study_payoff_matrix_effect()
    
    print("\n所有参数研究完成！结果保存在 parameter_study 目录")