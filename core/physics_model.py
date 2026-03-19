# core/physics_model.py
import torch
import numpy as np

# 强制全局使用双精度浮点数，防止计算图在临界耦合点崩溃
torch.set_default_dtype(torch.float64)

# ==========================================
# 1. 索引定义 (与 ModelListGP 的输出对齐)
# ==========================================
# 光学 GP 输出索引 (0-3)
IDX_KAPPA = 0
IDX_T = 1
IDX_PHI = 2
IDX_ALPHA_PASS = 3

# 电学确定性模型透传输出索引 (4-10)
IDX_R = 4
IDX_W = 5
IDX_RL = 6
IDX_DN = 7
IDX_DALPHA = 8
IDX_CJ = 9
IDX_RS = 10

# ==========================================
# 2. 电学经验公式与变量透传层
# ==========================================
def electrical_and_passthrough(X):
    """
    接收设计变量 X [R, gap, w, Nd, rL] (5维)
    输出 [R, w, rL, dn, dalpha, Cj, Rs] (7维)
    """
    R = X[..., 0]
    w = X[..., 2]
    Nd = X[..., 3]
    rL = X[..., 4]
    
    # 电学经验常数 (Mock值，后续可通过 CHARGE 离线拟合替换)
    k1, k2, k3, k4 = 1.2e-18, 5.2e-5, 2.8e-10, 1.5e-20
    
    Cj = k1 * torch.sqrt(Nd)
    Rs = k2 * (w / Nd)
    dn = -k3 * torch.pow(Nd, 0.8)
    dalpha = k4 * Nd
    
    return torch.stack([R, w, rL, dn, dalpha, Cj, Rs], dim=-1)

def mock_interpolate_ng(w):
    """简单的群折射率插值Mock函数"""
    return 4.0 - (w - 400.0) * 1e-4

# ==========================================
# 3. 灰盒约束函数 (传给 qNEHVI，要求 <= 0)
# ==========================================
def compute_a(Y):
    """计算单程场振幅衰减系数 (内部辅助函数)"""
    alpha_pass = Y[..., IDX_ALPHA_PASS]
    dalpha = Y[..., IDX_DALPHA]
    R = Y[..., IDX_R]
    rL = Y[..., IDX_RL]
    
    Ld = rL * 2 * np.pi * R
    alpha_total = alpha_pass + (Ld / (2 * np.pi * R)) * dalpha
    # 显式加入 1e-4 将 um 转换为 cm
    return 10 ** (-alpha_total * np.pi * R * 1e-4 / 20)

def er_con(Y):
    """消光比约束: ER >= 10 -> 10 - ER <= 0"""
    a = compute_a(Y)
    t_mag = Y[..., IDX_T]
    
    numerator = torch.clamp(torch.abs(a - t_mag), min=1e-12)
    denominator = torch.clamp(1 - a * t_mag, min=1e-12)
    er = -20 * torch.log10(numerator / denominator)
    return 10.0 - er

def q_lower_con(Y):
    """Q值下限: Q >= 9700 -> 9700 - Q <= 0"""
    a = compute_a(Y)
    t_mag = Y[..., IDX_T]
    R = Y[..., IDX_R]
    w = Y[..., IDX_W]
    
    n_g = mock_interpolate_ng(w)
    lambda0 = 1550e-9
    
    denominator = torch.clamp(1 - a * t_mag, min=1e-12)
    Q = (np.pi * n_g * 2 * np.pi * R * 1e-6 / lambda0) * torch.sqrt(a * t_mag) / denominator
    return 9700.0 - Q

def q_upper_con(Y):
    """Q值上限: Q <= 10000 -> Q - 10000 <= 0"""
    a = compute_a(Y)
    t_mag = Y[..., IDX_T]
    R = Y[..., IDX_R]
    w = Y[..., IDX_W]
    
    n_g = mock_interpolate_ng(w)
    lambda0 = 1550e-9
    
    denominator = torch.clamp(1 - a * t_mag, min=1e-12)
    Q = (np.pi * n_g * 2 * np.pi * R * 1e-6 / lambda0) * torch.sqrt(a * t_mag) / denominator
    return Q - 10000.0

def rc_con(Y):
    """RC带宽约束: fRC >= 20GHz -> 20e9 - fRC <= 0"""
    Cj = Y[..., IDX_CJ]
    Rs = Y[..., IDX_RS]
    f_rc = 1.0 / (2 * np.pi * Rs * Cj)
    return 20e9 - f_rc

# ==========================================
# 4. 白盒约束 (统一改为接收 Y，要求 <= 0)
# ==========================================
def fsr_con(Y):
    """FSR约束: FSR >= 6.4nm -> 6.4e-9 - FSR <= 0"""
    R = Y[..., IDX_R]  # 直接从透传的 Y 中获取半径 R
    w = Y[..., IDX_W]  # 直接从透传的 Y 中获取波导宽度 w
    n_g = mock_interpolate_ng(w)
    lambda0 = 1550e-9
    
    fsr = lambda0**2 / (n_g * 2 * np.pi * R * 1e-6)
    return 6.4e-9 - fsr  # 返回 <= 0 表示满足约束

# ==========================================
# 5. 目标函数 (最大化调制效率)
# ==========================================
def obj_efficiency(Y):
    """计算调制效率 η_m"""
    dn = Y[..., IDX_DN]
    rL = Y[..., IDX_RL]
    w = Y[..., IDX_W]
    n_g = mock_interpolate_ng(w)
    
    lambda0 = 1550.0 # nm
    V_op = 1.0
    
    eta_m = torch.abs(dn / (n_g * V_op)) * rL * lambda0
    return eta_m

def obj_radius(Y):
    """最小化半径 R (通过最大化 -R 实现)"""
    return -Y[..., IDX_R]