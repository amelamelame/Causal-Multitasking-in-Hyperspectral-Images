"""
utils/gpu_utils.py
VRAM monitoring + seed setting for RTX 3060 6GB.
"""
import os, random, gc
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device(cfg: dict) -> torch.device:
    """
    Activate RTX 3060 GPU properly.
    Prints GPU name + VRAM so you can confirm it's being used.
    """
    req = cfg["training"].get("device","cuda")
    if req == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
        props  = torch.cuda.get_device_properties(0)
        vram   = props.total_memory / 1e9
        print(f"\n  ✓ GPU activated: {props.name}")
        print(f"  ✓ VRAM available: {vram:.2f} GB")
        print(f"  ✓ CUDA version:   {torch.version.cuda}")
        if vram < 5.0:
            print("  ⚠  < 5 GB VRAM — reduce batch_size to 2 in config.yaml")
        # Enable TF32 for RTX 30xx — faster matmul with negligible precision loss
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        return device
    else:
        print("  ⚠  CUDA not available — running on CPU (slow!)")
        return torch.device("cpu")


class VRAMMonitor:
    def __init__(self, device: torch.device):
        self.device  = device
        self.is_cuda = device.type == "cuda"
        if self.is_cuda:
            self.total = torch.cuda.get_device_properties(0).total_memory / 1e9

    def print(self, label=""):
        if not self.is_cuda: return
        alloc = torch.cuda.memory_allocated(self.device)  / 1e9
        peak  = torch.cuda.max_memory_allocated(self.device) / 1e9
        bar   = "█" * int(alloc / self.total * 20)
        print(f"  [VRAM] {label:<20} {alloc:.2f}/{self.total:.1f} GB  |{bar:<20}|  peak={peak:.2f} GB")
        if alloc > self.total * 0.90:
            print("  ⚠  VRAM > 90% — risk of OOM. Reduce batch_size.")

    def clear(self):
        if self.is_cuda:
            gc.collect(); torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
