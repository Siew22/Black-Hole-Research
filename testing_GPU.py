# check_gpu.py
import tensorflow as tf
import torch
import sys

print("--- GPU Health Check ---")
print(f"Python Version: {sys.version}")
print("-" * 25)

# --- TensorFlow GPU Check ---
print("TensorFlow Version:", tf.__version__)
try:
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"✅ TensorFlow has found {len(gpu_devices)} GPU(s):")
        for i, device in enumerate(gpu_devices):
            print(f"  - GPU {i}: {device.name}")
            try:
                details = tf.config.experimental.get_device_details(device)
                print(f"    - Device Name: {details.get('device_name', 'N/A')}")
                print(f"    - Compute Capability: {details.get('compute_capability', 'N/A')}")
            except:
                 print("    - Could not get detailed device properties.")
    else:
        print("❌ TensorFlow: No GPU found. TensorFlow will run on CPU.")
except Exception as e:
    print(f"An error occurred during TensorFlow GPU check: {e}")

print("-" * 25)

# --- PyTorch GPU (CUDA) Check ---
print("PyTorch Version:", torch.__version__)
try:
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ PyTorch has found {gpu_count} CUDA-enabled GPU(s):")
        for i in range(gpu_count):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    - CUDA Version PyTorch is built with: {torch.version.cuda}")
    else:
        print("❌ PyTorch: No CUDA-enabled GPU found. PyTorch will run on CPU.")
except Exception as e:
    print(f"An error occurred during PyTorch GPU check: {e}")

print("-" * 25)
print("--- Health Check Complete ---")

# Optional: A small computation to see the GPU in action
if tf.config.list_physical_devices('GPU'):
    print("\nPerforming a small sample computation on GPU with TensorFlow...")
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print("TensorFlow sample computation successful. Result:")
            print(c.numpy())
    except Exception as e:
        print(f"TensorFlow sample computation failed: {e}")