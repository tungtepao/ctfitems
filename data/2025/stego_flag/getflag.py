#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
题目文件:
    - secret_image.png: 包含隐藏特征的图片
    - multimodal_model.pth: 训练好的神经网络模型
    - stego_hint.txt: 解题提示文件

解题方法:
    1. 取图片左上角前20个像素的R值，计算R值mod10得到20维特征
    2. 将20维特征输入MLP模型(3层线性层+ReLU)
    3. 模型输出27个数值，取整后转为ASCII码
    4. ASCII码转换为字符得到完整flag
"""

import torch
import torch.nn as nn
from PIL import Image


class MultimodalMLP(nn.Module):
    """
    题目中的MLP模型结构
    - fc1: Linear(20, 64)  输入层: 20维特征 → 64维隐藏层
    - fc2: Linear(64, 32)  隐藏层: 64维 → 32维
    - fc3: Linear(32, 27)  输出层: 32维 → 27维(对应27个字符的flag)
    - 每层线性后接ReLU激活
    """
    def __init__(self):
        super(MultimodalMLP, self).__init__()
        self.fc1 = nn.Linear(20, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 27)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def extract_features(image_path):
    """
    从图片中提取20维特征
    修正：取第1列(x=0)前20行像素(y从0到19)的R值，计算R值mod10
    注意：stego_hint.txt中的"左上角前20个像素"应理解为第1列的前20个像素
    """
    img = Image.open(image_path).convert("RGB")
    features = []
    for y in range(20):  # 取第1列的前20行
        r, g, b = img.getpixel((0, y))  # x=0, y从0到19
        r_mod10 = r % 10
        features.append(r_mod10)
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)


def solve_ctf(model_path, image_path):
    """
    主解题函数
    """
    # 步骤1: 提取图片特征
    features = extract_features(image_path)

    # 步骤2: 加载模型并推理
    model = MultimodalMLP()
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        output = model(features)

    # 步骤3: 输出转ASCII码
    ascii_codes = [round(v.item()) for v in output[0]]
    flag = ''.join(chr(code) for code in ascii_codes)

    return features, ascii_codes, flag


if __name__ == "__main__":
    MODEL_PATH = "../0512/multimodal_model.pth"
    IMAGE_PATH = "../0512/secret_image.png"

    print("=" * 70)
    print(" " * 15 + "CTF题目解题 - Multimodal Steganography")
    print("=" * 70)

    try:
        features, ascii_codes, flag = solve_ctf(MODEL_PATH, IMAGE_PATH)

        print("\n[步骤1] 特征提取")
        print(f"  图片第1列前20个像素R值mod10: {list(features[0].numpy())}")

        print("\n[步骤2] 模型推理")
        print(f"  模型结构: fc1(20→64) + ReLU + fc2(64→32) + ReLU + fc3(32→27)")
        print(f"  输入特征维度: {features.shape}")
        print(f"  输出向量: {ascii_codes}")

        print("\n[步骤3] ASCII解码")
        print(f"  ASCII码: {ascii_codes}")
        print(f"  对应字符: {[chr(c) for c in ascii_codes]}")

        print("\n" + "=" * 70)
        print(f"  ✅ FLAG: {flag}")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
