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
    
    # --- 5.1 设置高级代理模型参数 ---
    if 'gp_params' in config['optimization']:
        optimizer.set_gp_params(**config['optimization']['gp_params'])
    
    # 6. 执行优化循环
    init_points = config['optimization']['init_points']
    n_iter = config['optimization']['n_iter']
    # 兼容处理原来直接写 kappa 的情况
    kappa = config['optimization'].get('kappa', 2.576)
    
    logger.info(f"-> 阶段一：开始随机探索 (Exploration)，采样点数: {init_points}")
    optimizer.maximize(init_points=init_points, n_iter=0)
    
    logger.info(f"-> 阶段二：开始贝叶斯智能寻优 (Active Learning)，最大迭代次数: {n_iter}")
    
    # --- 6.1 配置采集函数 ---
    acq_conf = config['optimization'].get('acq_func', {'kind': 'ei', 'kappa': kappa, 'xi': 0.0})
    utility = UtilityFunction(kind=acq_conf.get('kind', 'ei'), 
                              kappa=acq_conf.get('kappa', kappa), 
                              xi=acq_conf.get('xi', 0.0))
    
    # --- 6.2 执行主动学习与 Early Stopping 判断 ---
    early_stop_conf = config['optimization'].get('early_stopping', {'enabled': False})
    if early_stop_conf.get('enabled', False):
        patience = early_stop_conf.get('patience', 5)
        tol = early_stop_conf.get('tol', 0.01)
        best_fom = -float('inf')
        no_improve_count = 0
        
        logger.info(f"已开启提前停止(Early Stopping)策略: patience={patience}, tol={tol}")
        
        for step in range(n_iter):
            # 增量执行，每次迭代1步
            optimizer.maximize(init_points=0, n_iter=1, acquisition_function=utility)
            
            # 记录并判断当前步的提升幅度
            current_best = optimizer.max['target']
            improvement = current_best - best_fom
            
            if improvement > tol:
                best_fom = current_best
                no_improve_count = 0  # 如果有显著提升，重置计数器
            else:
                no_improve_count += 1
                
            if no_improve_count >= patience:
                logger.info(f"连续 {patience} 次迭代未见显著提升 (阈值:{tol})，触发提前停止！")
                break
    else:
        # 如果未开启提前停止，一次性执行完毕
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