import yaml
import torch
import numpy as np
import pandas as pd
from utils.visualizer import plot_botorch_results
from core.evaluator import MRMEvaluator
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import GenericMCMultiOutputObjective
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.sampling.get_sampler import GetSampler
from botorch.sampling.normal import SobolQMCNormalSampler

from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior

from core.physics_model import (
    electrical_and_passthrough, 
    er_con, q_lower_con, q_upper_con, rc_con, fsr_con,
    obj_efficiency, obj_radius
)

# 强制全局使用双精度浮点数，保障临界耦合与矩阵计算的数值稳定性
torch.set_default_dtype(torch.float64)

def load_config(config_path='configs/mrm_config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def evaluate_optical_physics(X_tensor, evaluator):
    """桥接函数：将 BoTorch 推荐的张量坐标送入 Lumerical，并将其结果组装回张量"""
    Y_list = []
    # 遍历批次中的每一个采样点
    for i in range(X_tensor.shape[0]):
        r = X_tensor[i, 0].item()
        g = X_tensor[i, 1].item()
        w = X_tensor[i, 2].item()
        
        print(f"     [FDTD] 正在仿真样本 {i+1}/{X_tensor.shape[0]}: R={r:.2f} um, gap={g:.1f} nm, w={w:.1f} nm...")
        kappa, t_mag, phi, alpha_pass = evaluator.run_physical_simulation(r, g, w)
        Y_list.append([kappa, t_mag, phi, alpha_pass])
        
    return torch.tensor(Y_list, dtype=torch.float64)

# ==========================================
# 核心修复 1: 完美融合 GP 与解析公式的自定义模型
# ==========================================
class CombinedPosterior(Posterior):
    def __init__(self, gp_posterior, det_Y):
        self.gp_posterior = gp_posterior
        self.det_Y = det_Y
        
    @property
    def device(self):
        return self.gp_posterior.device
        
    @property
    def dtype(self):
        return self.gp_posterior.dtype

    # 明确告诉 BoTorch 采样器噪声张量的形状
    @property
    def base_sample_shape(self):
        return self.mean.shape

    # 告诉采样器批处理维度的范围
    @property
    def batch_range(self):
        return self.gp_posterior.batch_range

    # 【本次新增】告诉采样器批处理维度的具体形状
    @property
    def batch_shape(self):
        return self.gp_posterior.batch_shape
        
    # 告诉超体积计算模块，扩展后的张量形状是什么
    def _extended_shape(self, sample_shape=torch.Size()):
        return sample_shape + self.base_sample_shape
        
    # 实现最新版 BoTorch 强制要求的底层噪声采样接口
    def rsample_from_base_samples(self, sample_shape, base_samples):
        # 1. 核心切割：底层的 Sobol 采样器会生成 11 维的基底噪声，
        # 但我们的光学 GP 只需要前 4 维的噪声，后面的电学公式不需要噪声。
        gp_base_samples = base_samples[..., :self.gp_posterior.mean.shape[-1]]
        
        # 2. 获取 GP 的蒙特卡洛采样结果 (自动保留梯度)
        gp_samples = self.gp_posterior.rsample_from_base_samples(sample_shape, gp_base_samples)
        
        # 3. 扩展解析张量的形状以匹配 GP 的采样批次
        expanded_det = self.det_Y.expand(*gp_samples.shape[:-1], self.det_Y.shape[-1])
        
        # 4. 将光学黑盒与电学白盒完美拼接
        return torch.cat([gp_samples, expanded_det], dim=-1)
        
    # 兼容常规采样接口
    def rsample(self, sample_shape=torch.Size(), base_samples=None):
        if base_samples is not None:
            return self.rsample_from_base_samples(sample_shape, base_samples)
        gp_samples = self.gp_posterior.rsample(sample_shape)
        expanded_det = self.det_Y.expand(*gp_samples.shape[:-1], self.det_Y.shape[-1])
        return torch.cat([gp_samples, expanded_det], dim=-1)
        
    @property
    def mean(self):
        return torch.cat([self.gp_posterior.mean, self.det_Y], dim=-1)
        
    @property
    def variance(self):
        det_var = torch.zeros_like(self.det_Y)
        return torch.cat([self.gp_posterior.variance, det_var], dim=-1)

# 向 BoTorch 底层注册我们的自定义后验采样规则
@GetSampler.register(CombinedPosterior)
def _get_sampler_combined(posterior, sample_shape, **kwargs):
    # 强制系统使用高质量的拟蒙特卡洛 (QMC) 采样器处理我们的灰盒模型
    return SobolQMCNormalSampler(sample_shape=sample_shape)

class CombinedModel(Model):
    def __init__(self, gp_model):
        super().__init__()
        self.gp_model = gp_model
        
    @property
    def num_outputs(self):
        # 4 (光学代理) + 7 (电学与几何透传) = 11维特征输出
        return self.gp_model.num_outputs + 7 
        
    def posterior(self, X, observation_noise=False, posterior_transform=None):
        # 1. 物理经验公式前向传播计算 (自动挂载 Autograd 梯度链)
        det_Y = electrical_and_passthrough(X)
        # 2. 调用 GP 获取后验分布
        gp_post = self.gp_model.posterior(X, observation_noise=observation_noise)
        # 3. 封装并返回联合分布
        return CombinedPosterior(gp_post, det_Y)


def get_fitted_model(train_X, train_Y_opt, bounds):
    """训练光学 GP 并与电学/透传模型组装"""
    # 核心修复 2: 引入 Normalize 和 Standardize 解决量纲悬殊引发的警告与崩溃
    gp_optical = SingleTaskGP(
        train_X, 
        train_Y_opt,
        input_transform=Normalize(d=5, bounds=bounds),
        outcome_transform=Standardize(m=4)
    )
    mll = ExactMarginalLogLikelihood(gp_optical.likelihood, gp_optical)
    fit_gpytorch_mll(mll)
    
    # 封装联合模型
    model = CombinedModel(gp_optical)
    return model

def main():
    print("[System] 启动 MRM 逆向设计 DSE 引擎 (全自动真机闭环版)...")
    config = load_config()
    evaluator = MRMEvaluator() # <--- 新增这行，唤醒 Lumerical
    
    # 1. 解析设计空间边界 (支持自动兼容 YAML 中的科学计数法)
    b_dict = config['bounds']
    bounds = torch.tensor([
        [float(v) for v in b_dict['radius']], 
        [float(v) for v in b_dict['gap']], 
        [float(v) for v in b_dict['width']], 
        [float(v) for v in b_dict['Nd']], 
        [float(v) for v in b_dict['rL']]
    ], dtype=torch.float64).T
    
    # 2. 初始化 LHS 采样 (此处简单使用随机采样代替)
    init_points = config['optimization']['init_points']
    train_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(init_points, 5)
    print(f"\n[System] 开始执行 {init_points} 个初始 LHS 采样点的真机仿真 (这可能需要几分钟)...")
    train_Y_opt = evaluate_optical_physics(train_X, evaluator)
    
    # 3. 贝叶斯优化主循环
    n_iter = config['optimization']['n_iter']
    print(f"\n[BO] 开始迭代寻优，共 {n_iter} 轮...")
    
    for i in range(n_iter):
        # 3.1 拟合/更新代理模型
        model = get_fitted_model(train_X, train_Y_opt, bounds)
        
        # 3.2 定义系统目标函数，并使用 BoTorch 专用的多目标包装器
        def obj_callable(Y, X=None):
            return torch.stack([obj_efficiency(Y), obj_radius(Y)], dim=-1)
            
        mo_objective = GenericMCMultiOutputObjective(obj_callable)
            
        # 3.3 构建采集函数 qLogNEHVI (采用对数版本，解决数值下溢问题)
        # 设定帕累托前沿参考点 (基线要求: η_m 下限，-R 下限)
        ref_point = torch.tensor([0.0, -30.0])
        
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point, 
            X_baseline=train_X,
            objective=mo_objective,
            # 将 fsr_con 加入输出约束列表！
            constraints=[er_con, q_lower_con, q_upper_con, rc_con, fsr_con] 
        )
        
        # 3.4 优化采集函数 (自动求导寻找下一个最佳采样点)
        print(f"  -> Iter {i+1}/{n_iter}: 正在优化多目标物理采集函数...")
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=config['optimization']['batch_size'],
            num_restarts=5,
            raw_samples=128
            # 删除了 nonlinear_inequality_constraints 行，彻底解决 ic_generator 报错
        )
        
        new_x = candidates.detach()
        print(f"     推荐候选点: R={new_x[0,0]:.2f} um, gap={new_x[0,1]:.1f} nm, Nd={10**new_x[0,3]:.1e}")
        
        # 3.5 获取真实的物理反馈 (此处用 Mock 数据发生器替代)
        new_y_opt = evaluate_optical_physics(new_x, evaluator)
        
        # 3.6 更新全局数据集
        train_X = torch.cat([train_X, new_x])
        train_Y_opt = torch.cat([train_Y_opt, new_y_opt])

    print("\n[System] 逆向设计优化圆满完成！")
    evaluator.shutdown() # 释放 Lumerical 进程

    # ==========================================
    # 数据持久化与指标逆向解析
    # ==========================================
    print("[System] 正在提取所有迭代张量并计算物理边界...")
    with torch.no_grad():
        # 重构 11 维全量物理张量
        det_Y = electrical_and_passthrough(train_X)
        Y_full = torch.cat([train_Y_opt, det_Y], dim=-1)
        
        # 逆向解析出真实的系统性能指标
        from core.physics_model import calc_er, calc_q, calc_fsr, calc_rc
        er_vals = calc_er(Y_full)
        q_vals = calc_q(Y_full)
        fsr_vals = calc_fsr(Y_full)
        rc_vals = calc_rc(Y_full)
        eff_vals = obj_efficiency(Y_full)
        
        # 严苛的刚性约束全量检验
        valid_mask = (er_con(Y_full) <= 0) & \
                     (q_lower_con(Y_full) <= 0) & \
                     (q_upper_con(Y_full) <= 0) & \
                     (rc_con(Y_full) <= 0) & \
                     (fsr_con(Y_full) <= 0)

    # 组装为格式化的 DataFrame
    df = pd.DataFrame({
        'Radius (um)': train_X[:, 0].numpy(),
        'Gap (nm)': train_X[:, 1].numpy(),
        'Width (nm)': train_X[:, 2].numpy(),
        'Nd (cm^-3)': torch.pow(10.0, train_X[:, 3]).numpy(),
        'r_L': train_X[:, 4].numpy(),
        'ER (dB)': er_vals.numpy(),
        'Q Factor': q_vals.numpy(),
        'FSR (nm)': fsr_vals.numpy() * 1e9,
        'f_RC (GHz)': rc_vals.numpy() / 1e9,
        'Efficiency': eff_vals.numpy(),
        'Is_Valid': valid_mask.numpy()
    })
    
    out_path = 'data/optimization_results.csv'
    df.to_csv(out_path, index=False)
    print(f"[System] 物理指标解算完成！原始数据集已保存至 {out_path}")
    
    # 打印最终验收结论
    valid_df = df[df['Is_Valid'] == True]
    if not valid_df.empty:
        best_idx = (valid_df['Efficiency'] / valid_df['Radius (um)']).idxmax()
        best_pt = valid_df.loc[best_idx]
        print("\n" + "="*45)
        print(" 🎯 发现符合全部刚性物理约束的综合最佳结构：")
        print("="*45)
        print(f" [几何参数] 半径 R     = {best_pt['Radius (um)']:.3f} μm")
        print(f" [几何参数] 间距 gap   = {best_pt['Gap (nm)']:.1f} nm")
        print(f" [电学参数] 掺杂 Nd    = {best_pt['Nd (cm^-3)']:.2e} cm⁻³")
        print("-" * 45)
        print(f" [系统指标] 消光比 ER  = {best_pt['ER (dB)']:.2f} dB (>=10)")
        print(f" [系统指标] 品质因数 Q  = {best_pt['Q Factor']:.0f} (9700~10000)")
        print(f" [系统指标] RC带宽     = {best_pt['f_RC (GHz)']:.1f} GHz (>=20)")
        print(f" [系统指标] FSR        = {best_pt['FSR (nm)']:.2f} nm (>=6.4)")
        print("="*45 + "\n")
    else:
        print("\n[警告] 经严格检验，所有探索点均未能同时满足 ER/Q/FSR/RC 全部刚性约束。需要调整 YAML 边界或增加迭代轮数。\n")

    # 触发渲染层
    plot_botorch_results(out_path)

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning) 
    main()