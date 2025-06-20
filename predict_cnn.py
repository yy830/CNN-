# cnn_predict.py
import torch
import numpy as np
import librosa
from model_cnn import CNNClassifier
import streamlit as st

# -*- coding: utf-8 -*-

def load_cnn_model(model_path):
    try:
        # ����ģ��ʵ��
        model = CNNClassifier()
        # ����ģ��Ȩ��
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # ��ӡģ�ͽṹ��״̬�ֵ�ļ������ڵ���
        print("model structure:", model)
        print("State dictionary key:", state_dict.keys())
        
        model.load_state_dict(state_dict, strict=False)  # ʹ�� strict=False �����ּ���
        model.eval()
        return model
    except Exception as e:
        print(f"Model loading failed.: {str(e)}")
        return None


def preprocess_audio_for_cnn(y, sr):
    try:
        # ��ȡMFCC����
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # ȷ���̶����ȣ���ѵ��ʱ��ͬ��
        target_length = 50  # Ӧ��ѵ��ʱʹ�õĳ�����ͬ
        if mfcc.shape[1] > target_length:
            mfcc = mfcc[:, :target_length]
        elif mfcc.shape[1] < target_length:
            pad_width = ((0, 0), (0, target_length - mfcc.shape[1]))
            mfcc = np.pad(mfcc, pad_width, mode='constant')
        
        # ת��ΪPyTorch���������ά��
        mfcc = mfcc.astype(np.float32)
        mfcc_tensor = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0)
        return mfcc_tensor
    except Exception as e:
        raise RuntimeError(f"Audio preprocessing failed:{str(e)}")

def predict_digit_cnn(model, mfcc_tensor):

    #ʹ�� CNN ģ�ͶԵ���������������Ԥ�⡣
    #���룺torch.Tensor����״ӦΪ (1, 1, 13, T)
    #���أ�Ԥ��������

    with torch.no_grad():
        output = model(mfcc_tensor)
        predicted = torch.argmax(output, dim=1).item()
    return predicted
