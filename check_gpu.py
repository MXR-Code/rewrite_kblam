import torch

def check_gpu():
    if torch.cuda.is_available():
        print(f"CUDA 是否可用: {torch.cuda.is_available()}")
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"GPU 数量: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i} 信息:")
            print(f"  名称: {torch.cuda.get_device_name(i)}")
            print(f"  总显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"  已使用显存: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  剩余显存: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    else:
        print("CUDA 不可用")

check_gpu()