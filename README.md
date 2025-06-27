# 基于CNN的语音识别系统

 🎤 基于CNN的语音识别系统

这是一个基于 PyTorch 实现的语音识别项目，采用 CNN 卷积神经网络对语音数字进行分类识别，支持 MFCC 特征提取、CNN 与 SVM 对比实验，并集成了 Streamlit 可视化前端界面。项目使用 Free Spoken Digit Dataset（FSDD）作为数据源，适合语音识别初学者或课程项目使用。

🧠 项目亮点

CNN深度学习模型：自动提取语音时频图特征，准确率高达 98.83%  
对比实验：支持传统 SVM 与 CNN 的性能对比  
交互式可视化界面：使用 Streamlit 展示波形图、频谱图、MFCC 图和预测结果  
结构灵活，易于扩展：支持模型自定义、前后端分离部署等  

 📁 项目结构

CNN-
├── data/                     # 存放语音数据（如FSDD）  
├── features/                 # MFCC特征缓存文件  
├── models/                   # 存放训练好的模型（如 best_model.pth）  
├── utils/                    # 数据处理与特征提取工具函数  
│   └── feature_extraction.py  
├── cnn_model.py              # CNN模型结构定义  
├── svm_model.py              # SVM模型训练与预测  
├── train.py                  # CNN模型训练脚本  
├── evaluate.py               # 模型评估与混淆矩阵分析  
├── ui_app.py                 # Streamlit前端入口  
├── requirements.txt          # 项目依赖包  
└── README.md                 # 项目说明文档  

📦 安装方法

1. 克隆项目仓库：  
git clone https://github.com/yy830/CNN-.git  
cd CNN-  
  
2. 创建并激活虚拟环境（推荐使用 Conda 或 venv）：  
  
python -m venv venv  
source venv/bin/activate  # Windows: venv\Scripts\activate  
  
3. 安装依赖项：  
pip install -r requirements.txt  
  
  
📌数据准备  
项目使用 [Free Spoken Digit Dataset (FSDD)](https://github.com/Jakobovski/free-spoken-digit-dataset) 数据集。  
git clone https://github.com/Jakobovski/free-spoken-digit-dataset.git data/  
项目会自动处理音频并提取 MFCC 特征。  
  
📌模型训练与评估  
训练 CNN 模型：  
python train.py  
评估并比较 CNN 和 SVM 模型：  
python evaluate.py  
  
📌启动前端可视化界面  
运行 Streamlit 前端界面：  
streamlit run ui_app.py  
  
📌前端功能：  
上传语音文件（WAV格式）  
查看时域波形图、频谱图、MFCC图  
查看 CNN 与 SVM 模型分类结果  
混淆矩阵与分类指标展示  
  
📌实验结果  
  
| 模型 | 准确率 | Macro Precision | Macro Recall | F1 Score |
|------|--------|-----------------|---------------|----------|
| CNN  | 98.83% | 98.84%          | 98.83%        | 98.83%   |
| SVM  | 68.00% | 71.00%          | 68.00%        | 68.00%   |
  
📌 TODO（可选扩展）  
加入实时麦克风输入功能  
支持更多模型结构（如 RNN, LSTM）  
引入多说话人数据增强  
部署成 Web API（FastAPI 或 Flask）  
致谢  
[FSDD 开源语音数据集](https://github.com/Jakobovski/free-spoken-digit-dataset)  
[PyTorch](https://pytorch.org/)  
[Streamlit](https://streamlit.io/)  
