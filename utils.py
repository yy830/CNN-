#utils.py
import matplotlib.pyplot as plt
import librosa.display
import streamlit as st
import numpy as np
import plotly.graph_objs as go
import seaborn as sns
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix


def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='b')
    ax.set_title("Waveform")
    st.pyplot(fig)


def plot_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mfcc, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set_title("MFCC")
    st.pyplot(fig)


def plot_fft(y, sr):
    Y = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(Y), 1/sr)
    magnitude = np.abs(Y)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs[:len(freqs) // 2], magnitude[:len(Y) // 2])
    ax.set_title("FFT - Frequency Domain")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    st.pyplot(fig)

def plot_fft_3d_interactive(y, sr):
    """绘制交互式3D FFT频谱图"""
    import plotly.graph_objects as go
    from scipy import signal
    
    # 使用STFT计算时频谱
    f, t, Zxx = signal.stft(y, sr, nperseg=1024, noverlap=512)
    
    # 计算magnitude谱
    Zxx = np.abs(Zxx)
    Zxx = 20 * np.log10(Zxx + 1e-10)
    
    # 创建3D图表
    fig = go.Figure(data=[go.Surface(
        x=t,
        y=f,
        z=Zxx,
        colorscale='Viridis'
    )])
    
    # 设置图表布局
    fig.update_layout(
        title={
            'text': '3D FFT频谱图',
            'font': {'size': 24}
        },
        scene={
            'xaxis_title': '时间 (秒)',
            'yaxis_title': '频率 (Hz)',
            'zaxis_title': '幅度 (dB)',
            'camera': {
                'eye': {'x': 1.8, 'y': 1.8, 'z': 1.5}
            }
        },
        width=800,
        height=600,
        margin={'l': 65, 'r': 50, 'b': 65, 't': 90}
    )
    
    return fig

def evaluate_model_traditional(model, X_test, y_test):
    """评估传统模型性能，返回混淆矩阵图像和分类报告"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('传统模型混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    return fig, report

def evaluate_model_cnn(model, X_test, y_test):
    """评估CNN模型性能，返回混淆矩阵图像"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report

    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    if len(X_test_tensor.shape) == 3:
        X_test_tensor = X_test_tensor.unsqueeze(1)  # (N, 1, 13, T)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    batch_size = 32
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test_tensor), batch_size):
            batch = X_test_tensor[i:i+batch_size]
            outputs = model(batch)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('CNN model confusion matrix')
    plt.xlabel('Tag Estimation')
    plt.ylabel('True label')
    return fig,report
