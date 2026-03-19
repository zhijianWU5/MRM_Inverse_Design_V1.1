import sys
import os
import numpy as np

# ==========================================
# 基于 where 命令搜索结果的最终路径配置
# ==========================================
LUMERICAL_HOME = r"D:\Program Files\Lumerical\v232" 
LUMERICAL_BIN = os.path.join(LUMERICAL_HOME, "bin")
LUMERICAL_API = os.path.join(LUMERICAL_HOME, "api", "python")

# 核心修复：加载 DLL 环境
if os.path.exists(LUMERICAL_BIN):
    try:
        os.add_dll_directory(LUMERICAL_BIN)
        # print(f"[成功] 已加载 DLL 目录: {LUMERICAL_BIN}")
    except AttributeError:
        os.environ["PATH"] += os.pathsep + LUMERICAL_BIN

# 将 API 路径加入系统路径
if LUMERICAL_API not in sys.path:
    sys.path.append(LUMERICAL_API)

try:
    import lumapi
except ImportError as e:
    print(f"[致命错误] 依然无法导入 lumapi。报错: {e}")
    sys.exit(1)

# ==========================================
# 2. 物理引擎接口类
# ==========================================
class LumericalEngine:
    def __init__(self, fsp_file_path):
        """初始化真实的 Lumerical FDTD 引擎"""
        print("[Engine] 正在后台启动 Lumerical FDTD (静默模式)...")
        self.fdtd = lumapi.FDTD(hide=True) 
        
        if os.path.exists(fsp_file_path):
            self.fdtd.load(fsp_file_path)
            print(f"[Engine] 成功加载 FDTD DC 模板: {fsp_file_path}")
        else:
            raise FileNotFoundError(f"找不到 FDTD 模板文件: {fsp_file_path}")

    def evaluate_mrm(self, radius, gap, width):
        """物理层重构：传入几何参数，提取底层光学 S 参数"""
        self.fdtd.switchtolayout()

        # 动态注入几何参数
        self.fdtd.setnamed('MRM_Structure', 'radius', radius * 1e-6)
        self.fdtd.setnamed('MRM_Structure', 'gap', gap * 1e-9)
        self.fdtd.setnamed('MRM_Structure', 'width', width * 1e-9)

        # 执行仿真 (此时只需要不到1分钟)
        self.fdtd.run()

        # 提取直通端 (Through) 透射率
        T_through_data = self.fdtd.getresult('through_port', 'T')
        T_through = np.abs(T_through_data['T'][0]) # 提取中心波长处的透射功率
        
        # 提取交叉端 (Cross) 透射率
        T_cross_data = self.fdtd.getresult('cross_port', 'T')
        T_cross = np.abs(T_cross_data['T'][0])

        # 解析灰盒基础参数：t 和 kappa 的幅值
        t_mag = float(np.sqrt(T_through))
        kappa = float(np.sqrt(T_cross))
        
        # 对于本科毕设的解析灰盒模型：
        # 1. 附加相位 phi_t 对 Q 和 ER 极值的影响可忽略，这里直接传 0
        # 2. 3D FDTD 提取的传输损耗噪音大，采用业界典型的硅波导损耗 2.0 dB/cm
        phi = 0.0 
        alpha_pass = 2.0 

        return kappa, t_mag, phi, alpha_pass

    def close(self):
        """释放 License 并关闭进程"""
        self.fdtd.close()
        print("[Engine] Lumerical FDTD 进程已安全关闭。")