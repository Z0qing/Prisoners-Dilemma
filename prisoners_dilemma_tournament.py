import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os
import datetime

# 创建保存图片的目录
if not os.path.exists('results'):
    os.makedirs('results')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 囚徒困境收益矩阵
PAYOFF_MATRIX = {
    ('C', 'C'): (3, 3),
    ('C', 'D'): (0, 5),
    ('D', 'C'): (5, 0),
    ('D', 'D'): (1, 1)
}

# ======================== 策略定义 ========================
def always_cooperate(history, **kwargs):
    """始终合作"""
    return 'C'

def always_defect(history, **kwargs):
    """始终背叛"""
    return 'D'

def random_strategy(history, cooperation_prob=0.5, **kwargs):
    """随机策略"""
    return 'C' if np.random.random() < cooperation_prob else 'D'

def tit_for_tat(history, **kwargs):
    """以牙还牙"""
    if not history:
        return 'C'
    return history[-1][1]  # 返回对手上一轮的动作

def generous_tft(history, forgiveness_prob=0.2, **kwargs):
    """宽容的以牙还牙"""
    if not history:
        return 'C'
    last_opponent_action = history[-1][1]
    
    if last_opponent_action == 'D':
        if np.random.random() < forgiveness_prob:
            return 'C'
    return last_opponent_action

def tit_for_two_tats(history, **kwargs):
    """以二报还一报"""
    if not history:
        return 'C'
    
    # 检查最近两次动作
    if len(history) < 2:
        return 'C'
    
    # 如果对手最近两次都背叛
    if history[-1][1] == 'D' and history[-2][1] == 'D':
        return 'D'
    return 'C'

def grudger(history, **kwargs):
    """记仇者（冷酷触发）"""
    if not history:
        return 'C'
    
    # 检查是否有任何背叛
    if any(a2 == 'D' for _, a2 in history):
        return 'D'
    return 'C'

def wsls(history, **kwargs):
    """赢保持-输改变"""
    if not history:
        return 'C'
    last_act, last_opponent_act = history[-1]
    last_payoff = PAYOFF_MATRIX[(last_act, last_opponent_act)][0]
    
    # 判断是否"获胜"
    if last_payoff >= PAYOFF_MATRIX[(last_opponent_act, last_act)][1]:
        return last_act
    else:
        return 'D' if last_act == 'C' else 'C'

def adaptive_tft(history, window=5, **kwargs):
    """自适应TFT"""
    if not history:
        return 'C'
    
    # 计算对手近期合作率
    recent = history[-window:] if len(history) >= window else history
    if recent:
        coop_rate = sum(1 for _, a2 in recent if a2 == 'C') / len(recent)
    else:
        coop_rate = 1.0
    
    # 根据合作率调整报复概率
    if history[-1][1] == 'D':
        return 'D' if np.random.random() > coop_rate else 'C'
    return 'C'

# 定义策略列表（全局可访问）
strategies = [
    always_cooperate,
    always_defect,
    random_strategy,
    tit_for_tat,
    generous_tft,
    tit_for_two_tats,
    grudger,
    wsls,
    adaptive_tft
]

# ======================== 博弈模拟系统 ========================
def simulate_rounds(strategy1, strategy2, rounds=200, noise_level=0.05, **kwargs):
    """模拟多轮博弈"""
    history = []
    score1, score2 = 0, 0
    
    for _ in range(rounds):
        # 获取双方策略决策
        act1 = strategy1(history, **kwargs)
        # 反转历史使策略2看到正确视角
        reversed_history = [(b, a) for a, b in history]
        act2 = strategy2(reversed_history, **kwargs)
        
        # 添加噪声
        if noise_level > 0:
            if np.random.random() < noise_level:
                act1 = 'D' if act1 == 'C' else 'C'
            if np.random.random() < noise_level:
                act2 = 'D' if act2 == 'C' else 'C'
        
        # 记录动作并计算收益
        payoff = PAYOFF_MATRIX[(act1, act2)]
        history.append((act1, act2))
        score1 += payoff[0]
        score2 += payoff[1]
    
    return score1, score2, history

# ======================== 策略锦标赛 ========================
def strategy_tournament(strategies, rounds=200, repetitions=10, noise_level=0.05):
    """
    运行策略锦标赛
    返回:
    - 得分矩阵 (n x n x 2): 每个策略对每个对手的得分
    - 总得分排名
    """
    n = len(strategies)
    # 初始化得分矩阵 [策略A, 策略B, (A得分, B得分)]
    score_matrix = np.zeros((n, n, 2))
    
    print(f"开始策略锦标赛: {n}种策略 x {repetitions}次重复 x {rounds}轮/次")
    
    for i, s1 in enumerate(strategies):
        for j, s2 in enumerate(strategies):
            total_s1_score = 0
            total_s2_score = 0
            
            # 多次重复取平均值
            for _ in range(repetitions):
                s1_score, s2_score, _ = simulate_rounds(
                    s1, s2, 
                    rounds=rounds,
                    noise_level=noise_level
                )
                total_s1_score += s1_score
                total_s2_score += s2_score
            
            # 计算平均得分
            avg_s1_score = total_s1_score / repetitions
            avg_s2_score = total_s2_score / repetitions
            score_matrix[i, j] = [avg_s1_score, avg_s2_score]
    
    # 计算每个策略的总得分 (对抗所有对手)
    total_scores = np.zeros(n)
    for i in range(n):
        # 包含对抗所有对手的得分（包括自己）
        total_scores[i] = np.sum(score_matrix[i, :, 0]) + np.sum(score_matrix[:, i, 1])
    
    return score_matrix, total_scores

# ======================== 结果可视化 ========================
def visualize_results(strategies, score_matrix, total_scores, prefix=""):
    """可视化锦标赛结果并保存图片"""
    n = len(strategies)
    strategy_names = [s.__name__ for s in strategies]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 创建热力图展示得分矩阵
    plt.figure(figsize=(12, 10))
    
    # 提取策略A的得分（当A作为行策略时）
    player1_scores = score_matrix[:, :, 0]
    
    sns.heatmap(
        player1_scores, 
        annot=True, fmt=".0f",
        xticklabels=strategy_names,
        yticklabels=strategy_names,
        cmap="YlGnBu"
    )
    plt.title("策略锦标赛得分矩阵\n(值表示行策略对抗列策略的平均得分)", fontsize=14)
    plt.xlabel("对手策略", fontsize=12)
    plt.ylabel("主策略", fontsize=12)
    plt.tight_layout()
    
    # 保存热力图
    heatmap_filename = f"results/{prefix}heatmap_{timestamp}.png"
    plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"保存热力图: {heatmap_filename}")
    
    # 2. 创建总得分排名条形图
    plt.figure(figsize=(12, 6))
    
    # 按得分排序
    sorted_indices = np.argsort(total_scores)[::-1]
    sorted_scores = total_scores[sorted_indices]
    sorted_names = [strategy_names[i] for i in sorted_indices]
    
    bars = plt.bar(sorted_names, sorted_scores, color='skyblue')
    plt.ylabel('总得分', fontsize=12)
    plt.title('策略总得分排名', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # 在条形上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.0f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3点垂直偏移
                     textcoords="offset points",
                     ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存条形图
    barplot_filename = f"results/{prefix}barplot_{timestamp}.png"
    plt.savefig(barplot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"保存条形图: {barplot_filename}")
    
    # 3. 打印详细排名表
    print("\n策略总得分排名:")
    table_data = []
    for i in sorted_indices:
        avg_score = total_scores[i] / (2 * n)  # 平均每场得分
        table_data.append([
            strategy_names[i], 
            int(total_scores[i]),
            f"{avg_score:.1f}"
        ])
    
    print(tabulate(
        table_data, 
        headers=['策略', '总得分', '平均每场得分'],
        tablefmt='grid',
        numalign="right"
    ))
    
    return heatmap_filename, barplot_filename

# ======================== 单场对抗分析 ========================
def analyze_single_match(strategy1, strategy2, rounds=20, noise_level=0.0, prefix=""):
    """分析单场对抗并打印详细结果"""
    print(f"\n单场对抗分析: {strategy1.__name__} vs {strategy2.__name__}")
    s1_score, s2_score, history = simulate_rounds(
        strategy1, strategy2, 
        rounds=rounds,
        noise_level=noise_level
    )
    
    # 计算合作率
    s1_coop = sum(1 for a1, _ in history if a1 == 'C') / len(history)
    s2_coop = sum(1 for _, a2 in history if a2 == 'C') / len(history)
    
    print(f"最终得分: {strategy1.__name__}: {s1_score}, {strategy2.__name__}: {s2_score}")
    print(f"合作率: {strategy1.__name__}: {s1_coop:.2f}, {strategy2.__name__}: {s2_coop:.2f}")
    
    print("\n轮次历史:")
    for i, (a1, a2) in enumerate(history):
        payoff = PAYOFF_MATRIX[(a1, a2)]
        print(f"轮次 {i+1:2d}: {a1} vs {a2} -> 得分: {payoff[0]} vs {payoff[1]}")
    
    # 可视化得分变化
    plt.figure(figsize=(12, 6))
    
    # 计算累积得分
    s1_cumulative = np.cumsum([PAYOFF_MATRIX[(a1, a2)][0] for a1, a2 in history])
    s2_cumulative = np.cumsum([PAYOFF_MATRIX[(a1, a2)][1] for a1, a2 in history])
    
    plt.plot(s1_cumulative, label=f"{strategy1.__name__} (得分: {s1_score})", linewidth=2)
    plt.plot(s2_cumulative, label=f"{strategy2.__name__} (得分: {s2_score})", linewidth=2)
    
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('累积得分', fontsize=12)
    plt.title(f'{strategy1.__name__} vs {strategy2.__name__} 对抗得分变化', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/{prefix}{strategy1.__name__}_vs_{strategy2.__name__}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"保存对抗图表: {filename}")
    
    return s1_score, s2_score, history, filename

# ======================== 主程序 ========================
if __name__ == "__main__":
    # 创建实验标识前缀
    experiment_id = "tournament_"
    
    # 运行单场对抗分析
    print("="*50)
    print("开始单场对抗分析示例")
    print("="*50)
    
    # 示例1: 以牙还牙 vs 始终背叛
    analyze_single_match(tit_for_tat, always_defect, rounds=20, noise_level=0.0, prefix=experiment_id)
    
    # 示例2: 宽容TFT vs 以牙还牙
    analyze_single_match(generous_tft, tit_for_tat, rounds=20, noise_level=0.1, prefix=experiment_id)
    
    # 示例3: 记仇者 vs 随机策略
    analyze_single_match(grudger, random_strategy, rounds=20, noise_level=0.0, prefix=experiment_id)
    
    # 运行策略锦标赛
    print("\n" + "="*50)
    print("开始策略锦标赛")
    print("="*50)
    
    score_matrix, total_scores = strategy_tournament(
        strategies,
        rounds=200,      # 每场博弈轮数
        repetitions=10,  # 每对策略重复次数
        noise_level=0.05  # 添加5%的噪声
    )
    
    # 可视化结果并保存图片
    visualize_results(strategies, score_matrix, total_scores, prefix=experiment_id)