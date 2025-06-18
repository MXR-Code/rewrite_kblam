import time
import warnings
import argparse
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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_args(args):
    for name, value in args.__dict__.items():
        print(f"{name} = {value}")
        if name == 'separate_query_head':
            assert isinstance(args.separate_query_head, bool)
        if name == 'debug':
            assert isinstance(args.debug, bool)
            if args.debug:
                warnings.warn("debugging, not training", UserWarning)
        if name == 'device' and args.device == "cpu":
            warnings.warn("using CPU, not CUDA", UserWarning)
        time.sleep(1)
        print()


def countdown(seconds):
    for i in range(seconds, 0, -1):
        print(i)
        time.sleep(1)
    print("start！")
