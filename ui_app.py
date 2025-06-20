import streamlit as st
import librosa
import numpy as np
from preprocessing import load_audio, extract_mfcc
from utils import plot_waveform, plot_mfcc, plot_fft, plot_fft_3d_interactive,evaluate_model_cnn,evaluate_model_traditional
from model import train_model_from_file, predict_digit
from predict_cnn import load_cnn_model, preprocess_audio_for_cnn, predict_digit_cnn
from sklearn.model_selection import train_test_split
import matplotlib


matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False    # 正确显示负号

st.title("🎙️ 语音信号识别系统")

# 选择模型类型
st.sidebar.title("模型选择")
model_type = st.sidebar.radio("请选择识别模型类型", ["传统模型", "CNN 模型"])

# 加载传统模型
model = train_model_from_file()
st.sidebar.success("传统模型已加载")

# 加载 CNN 模型
try:
    cnn_model = load_cnn_model("C:/Users/daiyan/Desktop/Stu/AI_design/cnn_model.pth")
    if cnn_model is not None:
        st.sidebar.success("CNN 模型已加载")
    else:
        st.sidebar.error("CNN 模型加载失败")
except Exception as e:
    st.sidebar.error(f"CNN 模型加载错误: {str(e)}")
    cnn_model = None

uploaded_file = st.file_uploader("📤 上传一个 .wav 文件", type="wav")

if uploaded_file is not None:
    # 1. 加载音频
    y, sr = load_audio(uploaded_file)
    st.write("采样率:", sr)

    # 2. 可视化
    st.subheader("⏱️ 原始波形（时域）")
    plot_waveform(y, sr)

    mfcc = extract_mfcc(y, sr)
    st.markdown("### 🧠 MFCC 特征图（感知域）")
    plot_mfcc(y, sr)

    st.markdown("### 📊 FFT 频谱图（频域）")
    plot_fft(y, sr)

    # 显示3D FFT频谱图
    st.markdown("### 🌈 3D FFT频谱图")
    with st.spinner('正在生成3D频谱图...'):
        try:
            fig_3d = plot_fft_3d_interactive(y, sr)
            st.plotly_chart(fig_3d, use_container_width=True)
        except Exception as e:
            st.error(f"生成3D频谱图时出错：{str(e)}")
            st.exception(e)


    # 3. 识别
    try:
        if model_type == "传统模型":
            prediction = predict_digit(model, mfcc)
            st.subheader("🌟 识别结果（传统模型）")
            st.write(f"📢 模型预测的数字是：**{prediction}**")
        else:
            if cnn_model is None:
                st.error("CNN 模型未正确加载，请检查模型文件")
            else:
                mfcc_tensor = preprocess_audio_for_cnn(y, sr)
                cnn_prediction = predict_digit_cnn(cnn_model, mfcc_tensor)
                st.subheader("🌟 识别结果（CNN 模型）")
                st.write(f"📢 模型预测的数字是：**{cnn_prediction}**")
    except Exception as e:
        st.error(f"预测过程出错：{str(e)}")
        st.error("详细错误信息：")
        st.exception(e)

   

if st.sidebar.button("📊 评估CNN模型"):
    st.subheader("CNN模型评估")
    try:
        data = np.load("digit_mfcc_cnn.npz")
        X, y = data["X"], data["y"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        if len(X_test.shape) == 3:
            X_test = X_test[:, np.newaxis, :, :]
        fig, report = evaluate_model_cnn(cnn_model, X_test, y_test)
        st.pyplot(fig)
        # 转为DataFrame并展示
        import pandas as pd
        report_df = pd.DataFrame(report).T
        st.subheader("分类指标表")
        st.dataframe(report_df.style.format("{:.2f}"))
    except Exception as e:
        st.error(f"模型评估出错: {str(e)}")
        st.exception(e)

if st.sidebar.button("📊 评估传统模型"):
    st.subheader("传统模型评估")
    try:
        data = np.load("digit_mfcc_cnn.npz")
        X, y = data["X"], data["y"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        # 只取每个样本的均值特征（如训练时那样），假设传统模型用的是每帧MFCC的均值
        if len(X_test.shape) == 3:
            # (N, 13, T) → (N, 13)
            X_test_feat = X_test.mean(axis=2)
        else:
            X_test_feat = X_test
        fig, report = evaluate_model_traditional(model, X_test_feat, y_test)
        st.pyplot(fig)
        import pandas as pd
        report_df = pd.DataFrame(report).T
        st.subheader("分类指标表")
        st.dataframe(report_df.style.format("{:.2f}"))
    except Exception as e:
        st.error(f"模型评估出错: {str(e)}")
        st.exception(e)

#cd C:\Users\daiyan\Desktop\Stu\AI_design
#streamlit run ui_app.py
