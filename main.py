import torch
import clip
import numpy as np

# GPUが使えるか
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

# CLIPのモデルリストが取得できるか
print(f"CLIP Models: {clip.available_models()}")