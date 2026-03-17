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
            print(f"[Engine] 成功加载 FDTD 模板: {fsp_file_path}")
        else:
            raise FileNotFoundError(f"找不到 FDTD 模板文件: {fsp_file_path}")

    def evaluate_mrm(self, radius, gap):
        """核心动作：修改参数 -> 运行仿真 -> 提取数据"""
        self.fdtd.switchtolayout()

        # 遵循师兄文档的 set 操作
        self.fdtd.setnamed('MRM_Structure', 'radius', radius * 1e-6)
        self.fdtd.setnamed('MRM_Structure', 'gap', gap * 1e-9)

        self.fdtd.run()

        # 提取结果并转化为 FoM 指标
        T_result = self.fdtd.getresult('through_port', 'T')
        T_val = np.abs(T_result['T'])
        T_db = 10 * np.log10(T_val)

        IL = -np.max(T_db)         
        ER = np.max(T_db) - np.min(T_db)

        return float(ER), float(IL)

    def close(self):
        """释放 License 并关闭进程"""
        self.fdtd.close()
        print("[Engine] Lumerical FDTD 进程已安全关闭。")