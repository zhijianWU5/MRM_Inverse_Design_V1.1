import yaml
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

from utils.logger import setup_logger
from utils.visualizer import plot_optimization_results
from core.evaluator import MRMEvaluator

def load_config(config_path='configs/mrm_config.yaml'):
    """读取 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # 1. 初始化系统基建 (日志)
    logger = setup_logger()
    logger.info("系统启动：开始构建 MRM 逆向设计 DSE 引擎...")
    
    # 2. 加载全局配置
    config = load_config()
    logger.info("成功加载系统参数配置。")
    
    # 3. 实例化物理评估器
    evaluator = MRMEvaluator(config)
    
    # 4. 定义供优化器调用的"黑盒目标函数"
    def objective_function(radius, gap):
        # 步骤 A: 真实的 3D FDTD 仿真！
        er, il = evaluator.run_physical_simulation(radius, gap)
        # 步骤 B: 计算 FoM 得分
        fom = evaluator.calculate_fom(er, il)
        return fom

    # 5. 配置贝叶斯优化器
    pbounds = config['bounds']
    # 将 yaml 中的 list 转换为 bayesian-optimization 需要的 tuple
    optimize_bounds = {
        'radius': tuple(pbounds['radius']), 
        'gap': tuple(pbounds['gap'])
    }
    
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=optimize_bounds,
        random_state=42,             # 固定随机种子，保证每次运行轨迹一致
        allow_duplicate_points=True
    )
    
    # 6. 执行优化循环
    init_points = config['optimization']['init_points']
    n_iter = config['optimization']['n_iter']
    kappa = config['optimization']['kappa']
    
    logger.info(f"-> 阶段一：开始随机探索 (Exploration)，采样点数: {init_points}")
    optimizer.maximize(init_points=init_points, n_iter=0)
    
    logger.info(f"-> 阶段二：开始贝叶斯智能寻优 (Active Learning)，迭代次数: {n_iter}")
    # 设定采集函数为 Expected Improvement (EI)
    utility = UtilityFunction(kind="ei", kappa=kappa, xi=0.0)
    optimizer.maximize(init_points=0, n_iter=n_iter, acquisition_function=utility)
    
    # 7. 输出最终寻优结果
    best_res = optimizer.max
    logger.info("===" * 15)
    logger.info(f"逆向设计优化圆满完成！")
    logger.info(f"最高 FoM 得分: {best_res['target']:.4f}")
    logger.info(f"最佳几何参数: 半径 = {best_res['params']['radius']:.2f} um, 间距 = {best_res['params']['gap']:.2f} nm")
    # 7. 优化结束后的善后工作 (在 logger.info 输出最佳参数之后添加)
    logger.info("正在关闭底层物理仿真引擎...")
    evaluator.shutdown()  # 【核心新增】释放 Lumerical License

    # 8. 触发可视化模块 (新增)
    logger.info("正在生成 DSE 寻优轨迹可视化图表...")
    plot_optimization_results(optimizer, init_points)


if __name__ == '__main__':
    main()