# -- coding: utf-8 --

import sys
import numpy as np

try:
    # 尝试导入CUDA相关库
    import torch
    import torch.cuda as cuda
    
    print("=== CUDA 环境测试 ===")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {cuda.is_available()}")
    
    if cuda.is_available():
        print(f"CUDA 设备数: {cuda.device_count()}")
        for i in range(cuda.device_count()):
            print(f"设备 {i}: {cuda.get_device_name(i)}")
        
        # 尝试执行简单的CUDA操作
        device = torch.device("cuda:0")
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        y = torch.tensor([4.0, 5.0, 6.0], device=device)
        z = x + y
        print(f"CUDA 计算结果: {z}")
        print("CUDA 环境测试通过!")
    else:
        print("CUDA 不可用")
        
except ImportError as e:
    print(f"导入库失败: {e}")
    print("请安装PyTorch和CUDA驱动")
except Exception as e:
    print(f"测试过程中发生错误: {e}")
    import traceback
    traceback.print_exc()