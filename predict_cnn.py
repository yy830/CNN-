# cnn_predict.py
import torch
import numpy as np
import librosa
from model_cnn import CNNClassifier
import streamlit as st

# -*- coding: utf-8 -*-

def load_cnn_model(model_path):
    try:
        # 创建模型实例
        model = CNNClassifier()
        # 加载模型权重
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # 打印模型结构和状态字典的键，用于调试
        print("model structure:", model)
        print("State dictionary key:", state_dict.keys())
        
        model.load_state_dict(state_dict, strict=False)  # 使用 strict=False 允许部分加载
        model.eval()
        return model
    except Exception as e:
        print(f"Model loading failed.: {str(e)}")
        return None


def preprocess_audio_for_cnn(y, sr):
    try:
        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # 确保固定长度（与训练时相同）
        target_length = 50  # 应与训练时使用的长度相同
        if mfcc.shape[1] > target_length:
            mfcc = mfcc[:, :target_length]
        elif mfcc.shape[1] < target_length:
            pad_width = ((0, 0), (0, target_length - mfcc.shape[1]))
            mfcc = np.pad(mfcc, pad_width, mode='constant')
        
        # 转换为PyTorch张量并添加维度
        mfcc = mfcc.astype(np.float32)
        mfcc_tensor = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0)
        return mfcc_tensor
    except Exception as e:
        raise RuntimeError(f"Audio preprocessing failed:{str(e)}")

def predict_digit_cnn(model, mfcc_tensor):

    #使用 CNN 模型对单个语音样本进行预测。
    #输入：torch.Tensor，形状应为 (1, 1, 13, T)
    #返回：预测结果数字

    with torch.no_grad():
        output = model(mfcc_tensor)
        predicted = torch.argmax(output, dim=1).item()
    return predicted
