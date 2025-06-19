# check_env.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 屏蔽一些无关的 TensorFlow 日志

try:
    print("--- 正在尝试导入核心库 ---")
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    
    print("\n[✔] NumPy 导入成功！")
    print("[✔] TensorFlow 核心导入成功！")
    print("[✔] tensorflow.keras 导入成功！")
    print(f"\nTensorFlow 版本: {tf.__version__}")
    print(f"Keras 版本: {tf.keras.__version__}")
    
    print("\n--- 正在检查 GPU ---")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[✔] 成功找到 {len(gpus)} 个 GPU 设备: {gpus}")
    else:
        print("[i] 未找到 GPU，将使用 CPU。")
        
    print("\n✅✅✅ 环境自检通过！你的 Python 环境和依赖库安装正确！✅✅✅")

except ImportError as e:
    print(f"\n❌❌❌ 环境自检失败！出现导入错误：❌❌❌")
    print(f"错误信息: {e}")
    print("\n问题可能在于：")
    print("1. 你没有在正确的虚拟环境中运行此脚本。")
    print("2. pip install -r requirements.txt 安装过程中可能出现了错误。")

except Exception as e:
    print(f"\n❌❌❌ 环境自检失败！出现未知错误：❌❌❌")
    print(f"错误信息: {e}")