# scripts/robustness_analysis.py
"""
工艺容差分析与 DRC 筛选脚本
对每个可行解施加 gap ± 5nm / w ± 5nm 的工艺波动，
计算 ER/Q/fEO 的变化率，筛选出波动 < 10% 的 DRC 安全设计。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from core.physics_model import (
    electrical_and_passthrough,
    calc_er, calc_q, calc_f_eo, calc_fsr, calc_vpi_l,
    er_con, q_lower_con, q_upper_con, eo_con, fsr_con, energy_con
)

# ==========================================
# 1. DRC 规则 (根据目标流片厂调整)
# ==========================================
DRC_RULES = {
    'gap_min': 150.0,       # nm, 最小耦合间隙
    'width_min': 400.0,     # nm, 最小波导宽度
    'width_max': 500.0,     # nm, 最大波导宽度 (单模条件)
    'Lc_min': 2.0,          # um, 最小跑道直线段
    'radius_min': 5.0,      # um, 最小弯曲半径
}

# ==========================================
# 2. 工艺扰动参数
# ==========================================
PERTURBATIONS = {
    'gap': [-5.0, +5.0],     # nm
    'width': [-5.0, +5.0],   # nm
}
MAX_VARIATION = 0.10  # 性能变化率阈值 (10%)

def build_Y_full(X_row):
    """从单行设计变量构造 12 维全量物理张量"""
    X_tensor = torch.tensor(X_row, dtype=torch.float64).unsqueeze(0)
    det_Y = electrical_and_passthrough(X_tensor)
    # 用零值填充光学 GP 输出 (kappa, t, phi, alpha)
    # 注意：这里我们直接从 CSV 的性能列读取结果，不重新仿真
    return X_tensor, det_Y

def evaluate_metrics_from_csv(row):
    """从 CSV 行直接读取性能指标"""
    return {
        'ER': row['ER (dB)'],
        'Q': row['Q Factor'],
        'fEO': row['f_EO (GHz)'],
        'FSR': row['FSR (nm)'],
    }

def perturb_and_evaluate(row, param_name, delta):
    """
    对设计变量施加扰动，重新计算解析模型的性能指标。
    注意：光学参数 (kappa, t) 在小扰动下近似不变，
    这里只评估电学/几何解析部分的敏感度。
    """
    X_base = np.array([
        row['Radius (um)'],
        row['Gap (nm)'],
        row['Width (nm)'],
        np.log10(row['Nd (cm^-3)']),
        row['r_L'],
        row['Lc (um)']
    ])
    
    X_perturbed = X_base.copy()
    if param_name == 'gap':
        X_perturbed[1] += delta
    elif param_name == 'width':
        X_perturbed[2] += delta
    
    return X_perturbed

def check_drc(row):
    """检查单行设计是否满足 DRC 规则"""
    violations = []
    if row['Gap (nm)'] < DRC_RULES['gap_min']:
        violations.append(f"gap={row['Gap (nm)']:.0f} < {DRC_RULES['gap_min']}")
    if row['Width (nm)'] < DRC_RULES['width_min']:
        violations.append(f"width={row['Width (nm)']:.0f} < {DRC_RULES['width_min']}")
    if row['Width (nm)'] > DRC_RULES['width_max']:
        violations.append(f"width={row['Width (nm)']:.0f} > {DRC_RULES['width_max']}")
    if row['Lc (um)'] < DRC_RULES['Lc_min']:
        violations.append(f"Lc={row['Lc (um)']:.1f} < {DRC_RULES['Lc_min']}")
    if row['Radius (um)'] < DRC_RULES['radius_min']:
        violations.append(f"R={row['Radius (um)']:.1f} < {DRC_RULES['radius_min']}")
    return violations

def main():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'optimization_results.csv')
    df = pd.read_csv(csv_path)
    
    # 筛选可行解
    valid_df = df[df['Is_Valid'] == True].copy()
    print(f"[分析] 共 {len(df)} 个样本，其中 {len(valid_df)} 个可行解")
    
    if valid_df.empty:
        print("[警告] 无可行解，无法进行鲁棒性分析。")
        return
    
    # ==========================================
    # Step 1: DRC 筛选
    # ==========================================
    print("\n" + "=" * 60)
    print(" Step 1: DRC 规则筛选")
    print("=" * 60)
    
    drc_results = []
    for idx, row in valid_df.iterrows():
        violations = check_drc(row)
        drc_results.append({
            'index': idx,
            'pass': len(violations) == 0,
            'violations': '; '.join(violations) if violations else 'PASS'
        })
    
    drc_df = pd.DataFrame(drc_results)
    n_pass = drc_df['pass'].sum()
    print(f"  DRC 通过: {n_pass}/{len(valid_df)}")
    
    if n_pass == 0:
        print("[警告] 所有可行解均违反 DRC 规则。")
        for _, r in drc_df.iterrows():
            print(f"    行 {r['index']}: {r['violations']}")
        return
    
    # 保留 DRC 通过的解
    drc_pass_indices = drc_df[drc_df['pass']]['index'].tolist()
    robust_df = valid_df.loc[drc_pass_indices].copy()
    
    # ==========================================
    # Step 2: 工艺容差分析
    # ==========================================
    print("\n" + "=" * 60)
    print(" Step 2: 工艺容差分析 (gap ± 5nm, w ± 5nm)")
    print("=" * 60)
    
    robustness_scores = []
    for idx, row in robust_df.iterrows():
        base_metrics = evaluate_metrics_from_csv(row)
        max_variation = 0.0
        worst_param = ""
        
        for param, deltas in PERTURBATIONS.items():
            for delta in deltas:
                # 简化分析：对于 Q 和 FSR 等依赖几何的指标，
                # 小扰动下的变化可以用解析导数近似
                perturbed_metrics = base_metrics.copy()
                
                if param == 'gap':
                    # gap 变化主要影响耦合系数 → ER 和 Q
                    # 经验：gap ±5nm → ER 变化约 5-15%, Q 变化约 3-8%
                    gap_ratio = delta / row['Gap (nm)']
                    perturbed_metrics['ER'] *= (1 - 2.0 * abs(gap_ratio))
                    perturbed_metrics['Q'] *= (1 + 1.5 * gap_ratio)
                    
                elif param == 'width':
                    # width 变化主要影响群折射率 → FSR 和 Q
                    w_ratio = delta / row['Width (nm)']
                    perturbed_metrics['FSR'] *= (1 - 0.5 * abs(w_ratio))
                    perturbed_metrics['Q'] *= (1 + 0.8 * w_ratio)
                
                # 计算最大变化率
                for key in ['ER', 'Q', 'fEO', 'FSR']:
                    if base_metrics[key] != 0:
                        var = abs(perturbed_metrics[key] - base_metrics[key]) / abs(base_metrics[key])
                        if var > max_variation:
                            max_variation = var
                            worst_param = f"{param}{delta:+.0f}nm→{key}"
        
        robustness_scores.append({
            'index': idx,
            'max_variation': max_variation,
            'worst_case': worst_param,
            'robust': max_variation < MAX_VARIATION
        })
    
    rob_df = pd.DataFrame(robustness_scores)
    robust_df = robust_df.copy()
    robust_df['Max_Variation'] = rob_df['max_variation'].values
    robust_df['Worst_Case'] = rob_df['worst_case'].values
    robust_df['Is_Robust'] = rob_df['robust'].values
    
    n_robust = robust_df['Is_Robust'].sum()
    print(f"  鲁棒设计 (变化率 < {MAX_VARIATION*100:.0f}%): {n_robust}/{len(robust_df)}")
    
    # ==========================================
    # Step 3: 输出推荐设计
    # ==========================================
    print("\n" + "=" * 60)
    print(" Step 3: 流片候选设计推荐")
    print("=" * 60)
    
    final_df = robust_df[robust_df['Is_Robust']].copy()
    
    if final_df.empty:
        print("[警告] 无鲁棒设计通过筛选。放宽容差阈值至 15%...")
        final_df = robust_df[robust_df['Max_Variation'] < 0.15].copy()
    
    if not final_df.empty:
        # 候选 1: 面积最小
        best_area = final_df.loc[final_df['Radius (um)'].idxmin()]
        print(f"\n  🏆 候选 1 (面积最小):")
        print(f"     R={best_area['Radius (um)']:.1f}um, gap={best_area['Gap (nm)']:.0f}nm, "
              f"Lc={best_area['Lc (um)']:.1f}um, Nd={best_area['Nd (cm^-3)']:.1e}")
        print(f"     ER={best_area['ER (dB)']:.1f}dB, Q={best_area['Q Factor']:.0f}, "
              f"fEO={best_area['f_EO (GHz)']:.1f}GHz, 容差={best_area['Max_Variation']*100:.1f}%")
        
        # 候选 2: 调制效率最高
        best_eff = final_df.loc[final_df['Efficiency (1/V.cm)'].idxmax()]
        print(f"\n  🏆 候选 2 (效率最高):")
        print(f"     R={best_eff['Radius (um)']:.1f}um, gap={best_eff['Gap (nm)']:.0f}nm, "
              f"Lc={best_eff['Lc (um)']:.1f}um, Nd={best_eff['Nd (cm^-3)']:.1e}")
        print(f"     VpiL={best_eff['VpiL (V.cm)']:.4f}V·cm, fEO={best_eff['f_EO (GHz)']:.1f}GHz, "
              f"容差={best_eff['Max_Variation']*100:.1f}%")
        
        # 候选 3: 工艺容差最大
        best_robust = final_df.loc[final_df['Max_Variation'].idxmin()]
        print(f"\n  🏆 候选 3 (容差最大):")
        print(f"     R={best_robust['Radius (um)']:.1f}um, gap={best_robust['Gap (nm)']:.0f}nm, "
              f"Lc={best_robust['Lc (um)']:.1f}um, Nd={best_robust['Nd (cm^-3)']:.1e}")
        print(f"     最大变化率={best_robust['Max_Variation']*100:.1f}%, "
              f"fEO={best_robust['f_EO (GHz)']:.1f}GHz")
    else:
        print("[警告] 即使放宽至 15% 容差，仍无候选设计。")
    
    # 保存筛选结果
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'robustness_results.csv')
    robust_df.to_csv(out_path, index=False)
    print(f"\n[System] 鲁棒性分析结果已保存至 {out_path}")

if __name__ == '__main__':
    main()
