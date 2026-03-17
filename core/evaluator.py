import os  # 新增这一行
import numpy as np
from core.lum_interface import LumericalEngine

class MRMEvaluator:
    def __init__(self, config, fsp_file_path='data/mrm_base_template.fsp'):
            self.weights = config['weights']
            
            # 获取当前文件的绝对路径，并据此推导项目根目录
            current_dir = os.path.dirname(os.path.abspath(__file__)) # core 目录
            project_root = os.path.dirname(current_dir)             # 项目根目录
            
            full_fsp_path = os.path.join(project_root, fsp_file_path)
            
            # 实例化 Lumerical 引擎
            self.engine = LumericalEngine(fsp_file_path=full_fsp_path)

    def calculate_fom(self, er, il):
        """目标函数维持不变：FoM = w1*ER - w2*IL"""
        w_er = self.weights['w_er']
        w_il = self.weights['w_il']
        fom = (w_er * er) - (w_il * il)
        return fom

    def run_physical_simulation(self, radius, gap):
        """
        这不再是一个代理函数。这是真实的物理调用！
        """
        # 调用真实 FDTD 接口
        er, il = self.engine.evaluate_mrm(radius, gap)
        return er, il
        
    def shutdown(self):
        """提供一个安全关闭物理引擎的接口"""
        self.engine.close()