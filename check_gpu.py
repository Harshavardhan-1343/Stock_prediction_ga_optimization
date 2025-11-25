"""
Comprehensive PyTorch GPU Check Script
"""

import sys
import subprocess

print("="*80)
print("🔍 PYTORCH GPU DIAGNOSTIC")
print("="*80)

# Check 1: NVIDIA Driver
print("\n1️⃣ Checking NVIDIA Driver...")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ NVIDIA Driver is installed")
        print(result.stdout)
    else:
        print("❌ nvidia-smi failed to run")
except FileNotFoundError:
    print("❌ nvidia-smi not found - NVIDIA drivers may not be installed")

# Check 2: PyTorch
print("\n2️⃣ Checking PyTorch...")
try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   cuDNN version: {torch.backends.cudnn.version()}")
    print(f"   cuDNN enabled: {torch.backends.cudnn.enabled}")
    
    if torch.cuda.is_available():
        print(f"\n3️⃣ CUDA Devices: {torch.cuda.device_count()} GPU(s) found")
        for i in range(torch.cuda.device_count()):
            print(f"\n   GPU {i}:")
            print(f"      Name: {torch.cuda.get_device_name(i)}")
            print(f"      Compute Capability: {torch.cuda.get_device_capability(i)}")
            print(f"      Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"      Current Device: {'Yes' if i == torch.cuda.current_device() else 'No'}")
        
        # Test GPU computation
        print("\n4️⃣ Testing GPU Computation...")
        device = torch.device('cuda:0')
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        import time
        start = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()  # Wait for GPU to finish
        end = time.time()
        
        print(f"✅ GPU computation successful!")
        print(f"   Matrix multiplication time: {(end-start)*1000:.2f} ms")
        print(f"   Result device: {z.device}")
        
        # Memory info
        print("\n5️⃣ GPU Memory:")
        print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"   Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
    else:
        print("\n❌ CUDA NOT AVAILABLE!")
        print("\n   Possible issues:")
        print("   1. PyTorch CPU version installed instead of CUDA version")
        print("   2. CUDA Toolkit not installed")
        print("   3. CUDA version mismatch")
        print("\n   Install PyTorch with CUDA:")
        print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("   (For CUDA 11.8)")
        
except ImportError:
    print("❌ PyTorch is not installed")
    print("   Install with: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*80)
print("📋 RECOMMENDATIONS")
print("="*80)

try:
    import torch
    if torch.cuda.is_available():
        print("\n✅ GPU is properly configured!")
        print("   Your system is ready for GPU-accelerated training")
    else:
        print("\n❌ GPU NOT DETECTED")
        print("\n   To install PyTorch with CUDA 11.8:")
        print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("\n   To install PyTorch with CUDA 12.1:")
        print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
except:
    pass

print("\n" + "="*80)