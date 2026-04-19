# config.py

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B/32"

# 検証したい画像とラベルの定義
TARGET_IMAGE = "data/inputs/Kot_Leon.jpg"
LABEL_SET = [
    "a photo of a cat",
    "a fluffy domestic predator",
    "a small tiger",
    "an animal that is not a dog",
    "🐱",
    "asdfghjkl",
    "a youkan",
]

OUTPUT_DIR = "./results"
