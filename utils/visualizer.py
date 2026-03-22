# utils/visualizer.py
"""
Phase 2 全维度可视化模块
适配 6D 跑道型微环 + 3 目标 (Efficiency, -R, -Nd) 的帕累托优化框架
生成 6 面板发表级图表:
  Panel 1: 设计空间 Radius vs Gap (颜色=Lc)
  Panel 2: 帕累托投影 Radius vs Efficiency (颜色=ER)
  Panel 3: 帕累托投影 Radius vs Nd (颜色=fEO)
  Panel 4: 跑道维度 Lc vs Gap (颜色=Q)
  Panel 5: 约束诊断 Q vs fEO (约束框可视化)
  Panel 6: 综合指标雷达 (Top-3 流片候选)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np


def plot_botorch_results(csv_path='data/optimization_results.csv'):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("[Visualizer] 未找到结果文件。")
        return

    valid_df = df[df['Is_Valid'] == True]
    invalid_df = df[df['Is_Valid'] == False]

    # ==========================================
    # 全局画布配置
    # ==========================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('MRM Racetrack Inverse Design — 6D Pareto Optimization Dashboard',
                 fontsize=16, fontweight='bold', y=0.98)

    # 统一无效点样式
    inv_kw = dict(c='#D3D3D3', alpha=0.35, s=25, marker='x', linewidths=0.8)
    # 统一有效点边框
    val_edge = dict(edgecolor='black', linewidths=0.6)

    # ==========================================
    # Panel 1: 设计空间 Radius vs Gap, 颜色=Lc
    # ==========================================
    ax = axes[0, 0]
    ax.scatter(invalid_df['Radius (um)'], invalid_df['Gap (nm)'], label='Invalid', **inv_kw)
    if not valid_df.empty:
        sc1 = ax.scatter(valid_df['Radius (um)'], valid_df['Gap (nm)'],
                         c=valid_df['Lc (um)'], cmap='plasma', s=65, **val_edge)
        cb1 = plt.colorbar(sc1, ax=ax, pad=0.02)
        cb1.set_label('$L_c$ (μm)', fontsize=9)
    # DRC 安全线
    ax.axhline(y=150, color='red', ls='--', lw=1.2, alpha=0.7, label='DRC gap=150nm')
    ax.set_xlabel('Radius $R$ (μm)')
    ax.set_ylabel('Coupling Gap (nm)')
    ax.set_title('① Design Space ($R$ vs Gap)', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')

    # ==========================================
    # Panel 2: 帕累托投影 Radius vs Efficiency, 颜色=ER
    # ==========================================
    ax = axes[0, 1]
    ax.scatter(invalid_df['Radius (um)'], invalid_df['Efficiency (1/V.cm)'],
               label='Invalid', **inv_kw)
    if not valid_df.empty:
        sc2 = ax.scatter(valid_df['Radius (um)'], valid_df['Efficiency (1/V.cm)'],
                         c=valid_df['ER (dB)'], cmap='viridis', s=75, **val_edge)
        cb2 = plt.colorbar(sc2, ax=ax, pad=0.02)
        cb2.set_label('ER (dB)', fontsize=9)
        # 标注 Knee Point (效率/半径性价比最高)
        score = valid_df['Efficiency (1/V.cm)'] / valid_df['Radius (um)']
        best_idx = score.idxmax()
        ax.plot(valid_df.loc[best_idx, 'Radius (um)'],
                valid_df.loc[best_idx, 'Efficiency (1/V.cm)'],
                'r*', markersize=18, zorder=10, label='Knee Point')
    ax.set_xlabel('Radius $R$ (μm)')
    ax.set_ylabel('Modulation Efficiency $\\eta$ (1/V·cm)')
    ax.set_title('② Pareto: $R$ vs Efficiency', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')

    # ==========================================
    # Panel 3: 帕累托投影 Radius vs Nd, 颜色=fEO
    # ==========================================
    ax = axes[0, 2]
    ax.scatter(invalid_df['Radius (um)'], invalid_df['Nd (cm^-3)'],
               label='Invalid', **inv_kw)
    if not valid_df.empty:
        sc3 = ax.scatter(valid_df['Radius (um)'], valid_df['Nd (cm^-3)'],
                         c=valid_df['f_EO (GHz)'], cmap='coolwarm', s=65, **val_edge)
        cb3 = plt.colorbar(sc3, ax=ax, pad=0.02)
        cb3.set_label('$f_{EO}$ (GHz)', fontsize=9)
    ax.set_xlabel('Radius $R$ (μm)')
    ax.set_ylabel('Doping $N_d$ (cm$^{-3}$)')
    ax.set_yscale('log')
    ax.set_title('③ Pareto: $R$ vs $N_d$ (3rd Obj)', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')

    # ==========================================
    # Panel 4: 跑道维度 Lc vs Gap, 颜色=Q
    # ==========================================
    ax = axes[1, 0]
    ax.scatter(invalid_df['Lc (um)'], invalid_df['Gap (nm)'],
               label='Invalid', **inv_kw)
    if not valid_df.empty:
        sc4 = ax.scatter(valid_df['Lc (um)'], valid_df['Gap (nm)'],
                         c=valid_df['Q Factor'], cmap='YlOrRd', s=65, **val_edge)
        cb4 = plt.colorbar(sc4, ax=ax, pad=0.02)
        cb4.set_label('Q Factor', fontsize=9)
    ax.axhline(y=150, color='red', ls='--', lw=1.2, alpha=0.7, label='DRC gap=150nm')
    ax.set_xlabel('Coupling Length $L_c$ (μm)')
    ax.set_ylabel('Coupling Gap (nm)')
    ax.set_title('④ Racetrack: $L_c$ vs Gap', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')

    # ==========================================
    # Panel 5: 约束诊断 Q vs fEO (约束框可视化)
    # ==========================================
    ax = axes[1, 1]
    ax.scatter(invalid_df['Q Factor'], invalid_df['f_EO (GHz)'],
               label='Invalid', **inv_kw)
    if not valid_df.empty:
        sc5 = ax.scatter(valid_df['Q Factor'], valid_df['f_EO (GHz)'],
                         c=valid_df['ER (dB)'], cmap='viridis', s=65, **val_edge)
        cb5 = plt.colorbar(sc5, ax=ax, pad=0.02)
        cb5.set_label('ER (dB)', fontsize=9)
    # 约束边界框
    ax.axvline(x=4000, color='orangered', ls='--', lw=1.5, alpha=0.8, label='$Q_{min}$=4000')
    ax.axvline(x=10000, color='orangered', ls='--', lw=1.5, alpha=0.8, label='$Q_{max}$=10000')
    ax.axhline(y=20, color='dodgerblue', ls='--', lw=1.5, alpha=0.8, label='$f_{EO,min}$=20 GHz')
    # 填充可行域
    q_range = ax.get_xlim()
    feo_range = ax.get_ylim()
    ax.fill_between([4000, 10000], 20, feo_range[1] if feo_range[1] > 20 else 50,
                    alpha=0.08, color='green', label='Feasible Zone')
    ax.set_xlabel('Q Factor')
    ax.set_ylabel('$f_{EO}$ (GHz)')
    ax.set_title('⑤ Constraint Diagnostic ($Q$ vs $f_{EO}$)', fontsize=11)
    ax.legend(fontsize=7, loc='upper right', ncol=2)

    # ==========================================
    # Panel 6: 综合性能仪表 — Top-3 候选对比柱状图
    # ==========================================
    ax = axes[1, 2]
    if not valid_df.empty and len(valid_df) >= 1:
        # 选择 3 类代表性候选
        candidates = {}

        # 候选 1: 面积最小 (R 最小)
        candidates['Min Area'] = valid_df.loc[valid_df['Radius (um)'].idxmin()]

        # 候选 2: 效率最高
        candidates['Max Eff'] = valid_df.loc[valid_df['Efficiency (1/V.cm)'].idxmax()]

        # 候选 3: 掺杂最低 (Nd 最小, 可靠性最高)
        candidates['Low $N_d$'] = valid_df.loc[valid_df['Nd (cm^-3)'].idxmin()]

        # 归一化指标柱状图
        metrics = ['ER (dB)', 'Q Factor', 'f_EO (GHz)', 'Efficiency (1/V.cm)']
        display_names = ['ER (dB)', 'Q/1000', '$f_{EO}$ (GHz)', 'η (1/V·cm)']
        n_metrics = len(metrics)
        x_pos = np.arange(n_metrics)
        bar_width = 0.25
        colors = ['#2196F3', '#FF9800', '#4CAF50']

        for i, (label, cand) in enumerate(candidates.items()):
            values = []
            for m in metrics:
                v = cand[m]
                if m == 'Q Factor':
                    v = v / 1000  # 缩放
                values.append(v)
            ax.bar(x_pos + i * bar_width, values, bar_width,
                   label=label, color=colors[i], alpha=0.85, edgecolor='black', linewidth=0.5)

        ax.set_xticks(x_pos + bar_width)
        ax.set_xticklabels(display_names, fontsize=9)
        ax.set_ylabel('Value')
        ax.set_title('⑥ Top-3 Candidate Comparison', fontsize=11)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No Valid\nCandidates', ha='center', va='center',
                fontsize=16, color='red', transform=ax.transAxes)
        ax.set_title('⑥ Top-3 Candidates (N/A)', fontsize=11)

    # ==========================================
    # 全局布局微调
    # ==========================================
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = 'data/pareto_front_analysis.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n[Visualizer] 6D 帕累托优化全景仪表盘已保存至 {out_path}")
    plt.show()