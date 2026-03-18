import matplotlib.pyplot as plt
import numpy as np

def plot_optimization_results(optimizer, init_points):
    """
    接收贝叶斯优化器对象，提取历史数据并绘制可视化图表
    """
    # 1. 提取历史数据
    iterations = range(1, len(optimizer.res) + 1)
    targets = [res['target'] for res in optimizer.res]
    radius = [res['params']['radius'] for res in optimizer.res]
    gap = [res['params']['gap'] for res in optimizer.res]

    # 提取当前历史最优值的变化轨迹 (用于画收敛曲线)
    running_max = np.maximum.accumulate(targets)

    # 2. 设置画布 (1行2列，两个子图)
    fig = plt.figure(figsize=(14, 6))
    plt.style.use('seaborn-v0_8-whitegrid') # 设置优雅的图表风格

    # ==========================================
    # 子图 1: FoM 收敛曲线 (Convergence Curve)
    # ==========================================
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(iterations, targets, 'o-', alpha=0.5, label='Current FoM (Sampled)')
    ax1.plot(iterations, running_max, 'r*-', linewidth=2, label='Best FoM Found')
    
    # 画一条垂直虚线，区分“随机探索阶段”和“贝叶斯寻优阶段”
    if len(iterations) > init_points:
        ax1.axvline(x=init_points + 0.5, color='gray', linestyle='--', label='End of Random Init')
    
    ax1.set_title('Bayesian Optimization Convergence', fontsize=14)
    ax1.set_xlabel('Iteration Number', fontsize=12)
    ax1.set_ylabel('Figure of Merit (FoM)', fontsize=12)
    ax1.legend()

    # ==========================================
    # 子图 2: 3D 设计空间散点图 (3D Parameter Space)
    # ==========================================
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # 颜色映射：根据采样的先后顺序变色，看算法是怎么“移动”的
    scatter = ax2.scatter(radius, gap, targets, 
                          c=iterations, cmap='viridis', 
                          s=60, alpha=0.8, edgecolor='k')
    
    # 标记出全局最优点
    best_idx = np.argmax(targets)
    ax2.scatter(radius[best_idx], gap[best_idx], targets[best_idx], 
                color='red', s=150, marker='*', label='Global Best')

    ax2.set_title('Design Space Exploration Trajectory', fontsize=14)
    ax2.set_xlabel('Radius (um)', fontsize=12)
    ax2.set_ylabel('Gap (nm)', fontsize=12)
    ax2.set_zlabel('FoM Target', fontsize=12)
    
    # 添加颜色条
    cbar = fig.colorbar(scatter, ax=ax2, pad=0.1)
    cbar.set_label('Iteration Number')
    ax2.legend()

    # 3. 渲染并展示图表
    plt.tight_layout()
    plt.show()
    