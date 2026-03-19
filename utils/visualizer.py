# utils/visualizer.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_botorch_results(csv_path='data/optimization_results.csv'):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("[Visualizer] 未找到结果文件。")
        return

    # 按照是否通过所有物理约束划分为合格与不合格样本
    valid_df = df[df['Is_Valid'] == True]
    invalid_df = df[df['Is_Valid'] == False]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.style.use('seaborn-v0_8-whitegrid')

    # 图1: 物理设计空间的采样分布 (Radius vs Gap)
    ax1.scatter(invalid_df['Radius (um)'], invalid_df['Gap (nm)'], 
                c='lightgray', alpha=0.6, label='Invalid Constraints', s=40)
    if not valid_df.empty:
        ax1.scatter(valid_df['Radius (um)'], valid_df['Gap (nm)'], 
                    c='blue', edgecolor='black', label='Valid Candidates', s=70)
    ax1.set_xlabel('Microring Radius (μm)')
    ax1.set_ylabel('Coupling Gap (nm)')
    ax1.set_title('Design Space Exploration Trajectory')
    ax1.legend()

    # 图2: 帕累托前沿分析 (Radius vs Modulation Efficiency)
    # 目标：Radius 越小越好（靠左），Efficiency 越大越好（靠上）
    ax2.scatter(invalid_df['Radius (um)'], invalid_df['Efficiency'], 
                c='lightgray', alpha=0.6, label='Invalid', s=40)
    if not valid_df.empty:
        sc = ax2.scatter(valid_df['Radius (um)'], valid_df['Efficiency'], 
                         c=valid_df['ER (dB)'], cmap='viridis', edgecolor='black', 
                         s=90, label='Valid (Color=ER)')
        cbar = plt.colorbar(sc, ax=ax2)
        cbar.set_label('Extinction Ratio (dB)')
        
        # 启发式评价：标出“效率与半径性价比”最高的点
        best_idx = (valid_df['Efficiency'] / valid_df['Radius (um)']).idxmax()
        best_r = valid_df.loc[best_idx, 'Radius (um)']
        best_eff = valid_df.loc[best_idx, 'Efficiency']
        ax2.plot(best_r, best_eff, 'r*', markersize=18, label='Optimal Trade-off')

    ax2.set_xlabel('Microring Radius (μm)')
    ax2.set_ylabel('Modulation Efficiency (Relative)')
    ax2.set_title('Multi-Objective Pareto Analysis')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('data/pareto_front_analysis.png', dpi=300)
    print("\n[Visualizer] 高维帕累托前沿图表已生成并保存至 data/pareto_front_analysis.png")
    plt.show()