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
    Nd_log = X[..., 3]
    rL = X[..., 4]
    
    Nd = 10.0 ** Nd_log
    
    # 电学经验常数 (真实物理基准量级校准)
    # k1 适配电容 Cj: 设定目标 Cj ~ 50 fF (5e-14 F) at Nd=1e18
    # k1 * sqrt(1e18) = 5e-14 -> k1 = 5.0e-23
    k1 = 5.0e-23
    # k2 适配串联电阻 Rs: Rs 与掺杂成反比, 设定目标 Rs ~ 100 Ohm at w=450, Nd=1e18
    # k2 * (450 / 1e18) = 100 -> k2 = 2.2e17
    k2 = 2.2e17
    # k3 适配折射率有效变化 (等效 1V bias 下的 modal dn)
    # 设 k3 = 4.0e-19, 使得 Nd=1e18 时 dn 约 1e-4, 对应 VpiL 约 0.77 V.cm
    k3 = 4.0e-19
    # k4 适配静态传输损耗 dalpha (对应 dB/cm，背景掺杂吸收)
    # 设 k4 = 2.0e-17, 使得 Nd=1e18 时 dalpha 约 20 dB/cm
    k4 = 2.0e-17
    
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
    # 圆周单程长度为 2 * pi * R * 1e-4 cm
    return 10 ** (-alpha_total * 2 * np.pi * R * 1e-4 / 20)

def calc_er(Y):
    """计算消光比 ER (dB)"""
    a = compute_a(Y)
    t_mag = Y[..., IDX_T]
    
    numerator = torch.clamp(torch.abs(a - t_mag), min=1e-12)
    denominator = torch.clamp(1 - a * t_mag, min=1e-12)
    return -20 * torch.log10(numerator / denominator)

def er_con(Y):
    """消光比约束: ER >= 10 -> (10 - ER)/10 <= 0"""
    return (10.0 - calc_er(Y)) / 10.0

def calc_q(Y):
    """计算品质因数 Q"""
    a = compute_a(Y)
    t_mag = Y[..., IDX_T]
    R = Y[..., IDX_R]
    w = Y[..., IDX_W]
    
    n_g = mock_interpolate_ng(w)
    lambda0 = 1550e-9
    
    denominator = torch.clamp(1 - a * t_mag, min=1e-12)
    return (np.pi * n_g * 2 * np.pi * R * 1e-6 / lambda0) * torch.sqrt(a * t_mag) / denominator

def q_lower_con(Y):
    """Q值下限: Q >= 9700 -> (9700 - Q)/9700 <= 0"""
    return (9700.0 - calc_q(Y)) / 9700.0

def q_upper_con(Y):
    """Q值上限: Q <= 10000 -> (Q - 10000)/10000 <= 0"""
    return (calc_q(Y) - 10000.0) / 10000.0

def calc_rc(Y):
    """计算RC带宽 fRC (Hz)"""
    Cj = Y[..., IDX_CJ]
    Rs = Y[..., IDX_RS]
    return 1.0 / (2 * np.pi * Rs * Cj)

def rc_con(Y):
    """RC带宽约束: fRC >= 20GHz -> (20e9 - fRC)/20e9 <= 0"""
    return (20e9 - calc_rc(Y)) / 20e9

def energy_con(Y):
    """能量守恒约束: kappa^2 + t^2 <= 1.0"""
    kappa = Y[..., IDX_KAPPA]
    t_mag = Y[..., IDX_T]
    return kappa**2 + t_mag**2 - 1.0

# ==========================================
# 4. 白盒约束 (统一改为接收 Y，要求 <= 0)
# ==========================================
def calc_fsr(Y):
    """计算 FSR (m)"""
    R = Y[..., IDX_R]
    w = Y[..., IDX_W]
    n_g = mock_interpolate_ng(w)
    lambda0 = 1550e-9
    
    return lambda0**2 / (n_g * 2 * np.pi * R * 1e-6)

def fsr_con(Y):
    """FSR约束: FSR >= 6.4nm -> (6.4e-9 - FSR)/6.4e-9 <= 0"""
    return (6.4e-9 - calc_fsr(Y)) / 6.4e-9

# ==========================================
# 5. 目标函数 (最大化调制效率)
# ==========================================
def calc_vpi_l(Y):
    """计算 Vpi * L, 典型单位 V.cm"""
    dn = torch.abs(Y[..., IDX_DN])
    lambda0 = 1550e-7 # cm
    return lambda0 / (2 * torch.clamp(dn, min=1e-12)) # V.cm

def obj_efficiency(Y):
    """计算调制效率 η_m = 1 / (Vpi * L) (越大越好)"""
    return 1.0 / calc_vpi_l(Y)

def obj_radius(Y):
    """最小化半径 R (通过最大化 -R 实现)"""
    return -Y[..., IDX_R]