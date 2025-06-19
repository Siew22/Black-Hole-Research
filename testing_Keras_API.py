# test_keras3_compatibility.py
import tensorflow as tf
print(f"TensorFlow版本: {tf.__version__}")
print(f"Keras版本: {tf.keras.__version__}")
print("=" * 50)

# 测试基础导入
print("1. 测试基础层导入...")
try:
    from tensorflow.keras.layers import (
        Input, Dense, Reshape, Flatten, Dropout, BatchNormalization,
        LeakyReLU, Conv2DTranspose, Conv2D, Conv1D, UpSampling1D,
        Conv3DTranspose, UpSampling3D, Layer, Conv3D
    )
    print("✓ 基础层导入成功")
except ImportError as e:
    print(f"✗ 基础层导入失败: {e}")

# 测试concatenate vs Concatenate
print("\n2. 测试连接层...")
try:
    from tensorflow.keras.layers import concatenate
    print("✓ concatenate函数可用")
except ImportError:
    print("⚠ concatenate函数不可用")

try:
    from tensorflow.keras.layers import Concatenate
    print("✓ Concatenate层可用")
except ImportError:
    print("✗ Concatenate层不可用")

# 测试模型和优化器
print("\n3. 测试模型和优化器...")
try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import regularizers
    print("✓ 模型和优化器导入成功")
except ImportError as e:
    print(f"✗ 模型和优化器导入失败: {e}")

# 测试ImageDataGenerator
print("\n4. 测试ImageDataGenerator...")
try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    print("✓ ImageDataGenerator 可用")
except ImportError as e:
    print(f"⚠ ImageDataGenerator 不可用: {e}")
    # 测试新的数据API
    try:
        from tensorflow.keras.utils import image_dataset_from_directory
        print("✓ image_dataset_from_directory 可用（新API）")
    except ImportError as e2:
        print(f"✗ 新数据API也不可用: {e2}")

# 测试Keras Tuner
print("\n5. 测试Keras Tuner...")
try:
    import keras_tuner as kt
    print(f"✓ Keras Tuner 可用，版本: {kt.__version__}")
except ImportError as e:
    print(f"✗ Keras Tuner 不可用: {e}")

# 测试Wandb集成
print("\n6. 测试Wandb集成...")
try:
    import wandb
    print(f"✓ Wandb 可用，版本: {wandb.__version__}")
except ImportError as e:
    print(f"✗ Wandb 不可用: {e}")

try:
    from wandb.integration.keras import WandbMetricsLogger
    print("✓ WandbMetricsLogger 可用")
except ImportError as e:
    print(f"⚠ WandbMetricsLogger 不可用: {e}")
    
try:
    from wandb.integration.keras import WandbCallback
    print("✓ WandbCallback 可用")
except ImportError as e:
    print(f"⚠ WandbCallback 不可用: {e}")

# 测试实际功能
print("\n7. 测试实际功能...")
try:
    # 测试concatenate功能
    input1 = tf.keras.layers.Input(shape=(10,))
    input2 = tf.keras.layers.Input(shape=(10,))
    
    # 方法1：使用Concatenate层
    concat1 = tf.keras.layers.Concatenate()([input1, input2])
    print("✓ Concatenate层功能正常")
    
    # 方法2：使用concatenate函数
    try:
        concat2 = tf.keras.layers.concatenate([input1, input2])
        print("✓ concatenate函数功能正常")
    except:
        print("⚠ concatenate函数不可用，但Concatenate层可用")
        
except Exception as e:
    print(f"✗ 连接功能测试失败: {e}")

print("\n" + "=" * 50)
print("兼容性测试完成！")
print("\n建议的修复方案:")
print("1. 使用 Concatenate() 层而不是 concatenate() 函数")
print("2. 如果ImageDataGenerator不可用，使用 image_dataset_from_directory")
print("3. 如果WandbMetricsLogger不可用，使用 WandbCallback")