# core/evaluator.py
import os
from core.lum_interface import LumericalEngine

class MRMEvaluator:
    # 默认加载我们刚刚生成的新模板
    def __init__(self, fsp_file_path='data/mrm_dc_template.fsp'):
        current_dir = os.path.dirname(os.path.abspath(__file__)) 
        project_root = os.path.dirname(current_dir)             
        
        full_fsp_path = os.path.join(project_root, fsp_file_path)
        
        # 实例化 Lumerical 引擎
        self.engine = LumericalEngine(fsp_file_path=full_fsp_path)

    def run_physical_simulation(self, radius, gap, width):
        """调用 Lumerical 引擎并返回 [kappa, t_mag, phi, alpha_pass]"""
        kappa, t, phi, alpha = self.engine.evaluate_mrm(radius, gap, width)
        return kappa, t, phi, alpha
        
    def shutdown(self):
        self.engine.close()