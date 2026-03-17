import sys
import os

# 将搜索范围扩大到 ANSYS Inc 全局目录
root_search_path = r"D:\Program Files\Lumerical\v232"

print(f"=== 正在全局扫描 ANSYS Inc 目录下的 lumapi.py ===")

found_api_path = None
# 全局递归搜索
for root, dirs, files in os.walk(root_search_path):
    if "lumapi.py" in files:
        found_api_path = root
        break

if found_api_path:
    print(f"\n[发现] 真正的 API 路径位于: {found_api_path}")
    
    # 根据发现的 API 路径，自动推导 bin 路径
    # 通常 bin 文件夹在 api 文件夹的上两级或平行位置
    # 我们尝试自动定位 bin
    bin_path = None
    potential_bin = os.path.abspath(os.path.join(found_api_path, "..", "..", "bin"))
    if os.path.exists(potential_bin):
        bin_path = potential_bin
    
    print(f"[发现] 推测的 BIN 路径位于: {bin_path}")

    if bin_path:
        os.add_dll_directory(bin_path)
        sys.path.append(found_api_path)
        try:
            import lumapi
            print("\n[成功] 核心链路已打通！尝试启动 FDTD...")
            fdtd = lumapi.FDTD(hide=False)
            print("FDTD 启动成功！")
            fdtd.close()
        except Exception as e:
            print(f"导入成功但启动失败: {e}")
else:
    print(f"\n[严重错误] 在 {root_search_path} 下依然找不到 lumapi.py。")
    print("请手动检查 D:\\Program Files\\ANSYS Inc\\ANSYS Optics 文件夹内部。")