#model_SVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np

# 训练一个分类模型（从文件加载数据）
def train_model_from_file(file="digit_mfcc_dataset_success.npz"):
    data = np.load(file)
    X, y = data["X"], data["y"]
    model = SVC(kernel='rbf', probability=True)
    model.fit(X, y)
    return model

# 预测语音对应的数字
def predict_digit(model, mfcc_feature):
    x_input = np.mean(mfcc_feature.T, axis=0).reshape(1, -1)
    pred = model.predict(x_input)
    return int(pred[0])