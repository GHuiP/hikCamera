import sys
import os
import platform

# 显示系统信息
print("=== 系统信息 ===")
print(f"操作系统: {platform.platform()}")
print(f"Python版本: {platform.python_version()}")
print(f"Python路径: {sys.executable}")
print(f"系统架构: {platform.architecture()[0]}")

# 检查MVS SDK路径
MVS_PATH = "/opt/MVS/Samples/64/Python"
print(f"\n=== MVS SDK诊断 ===")
print(f"MVS SDK路径: {MVS_PATH}")

# 检查路径是否存在
if os.path.exists(MVS_PATH):
    print("✓ 路径存在")
    
    # 列出目录内容
    print(f"\n目录内容:")
    for item in os.listdir(MVS_PATH):
        item_path = os.path.join(MVS_PATH, item)
        if os.path.isfile(item_path):
            print(f"  文件: {item}")
        elif os.path.isdir(item_path):
            print(f"  目录: {item}")
    
    # 检查是否有MvCameraControl相关文件
    print(f"\n检查MvCameraControl相关文件:")
    for item in os.listdir(MVS_PATH):
        if "MvCameraControl" in item:
            print(f"  ✓ 找到: {item}")
    
    # 检查是否有.so文件
    print(f"\n检查动态链接库:")
    for root, dirs, files in os.walk("/opt/MVS/lib"):
        for file in files:
            if file.endswith(".so") and "MvCamera" in file:
                print(f"  ✓ 找到库文件: {os.path.join(root, file)}")
    
else:
    print("✗ 路径不存在")

# 检查PYTHONPATH
print(f"\n=== PYTHONPATH设置 ===")
print(f"当前sys.path:")
for path in sys.path:
    print(f"  {path}")

# 尝试直接导入
print(f"\n=== 尝试导入 ===")
sys.path.insert(0, MVS_PATH)

# 尝试不同的导入方式
try:
    import MvCameraControl_class as MvCamera
    print("✓ 成功导入 MvCameraControl_class")
except ImportError as e:
    print(f"✗ 导入 MvCameraControl_class 失败: {e}")

try:
    import MvCameraControl
    print("✓ 成功导入 MvCameraControl")
except ImportError as e:
    print(f"✗ 导入 MvCameraControl 失败: {e}")

# 检查是否有__init__.py文件
init_file = os.path.join(MVS_PATH, "__init__.py")
if os.path.exists(init_file):
    print(f"\n✓ 找到__init__.py文件")
    with open(init_file, 'r') as f:
        content = f.read()
        print(f"  内容: {content}")
else:
    print(f"\n✗ 未找到__init__.py文件")

# 检查依赖
print(f"\n=== 依赖检查 ===")
try:
    import ctypes
    print("✓ ctypes模块可用")
except ImportError:
    print("✗ ctypes模块不可用")

try:
    import numpy
    print("✓ numpy模块可用")
except ImportError:
    print("✗ numpy模块不可用")

# 提示
print(f"\n=== 建议 ===")
print("1. 确认MVS SDK已完整安装")
print("2. 检查是否使用了正确的Python版本(SDK通常支持Python 3.6-3.9)")
print("3. 尝试以管理员权限运行")
print("4. 检查/opt/MVS/setup.sh是否已执行")