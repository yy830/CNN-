# -*- coding: utf-8 -*-
# ...existing code...
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

DATASET_PATH = "C:/Users/daiyan/Desktop/Stu/AI_design/free-spoken-digit-dataset-master/recordings"

X, y = [], []

for filename in os.listdir(DATASET_PATH):
    if filename.endswith(".wav"):
        label = int(filename[0])  # 提取数字标签
        file_path = os.path.join(DATASET_PATH, filename)
        y_audio, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        X.append(mfcc_mean)
        y.append(label)

X = np.array(X)
y = np.array(y)


plt.figure(figsize=(8, 4))
plt.hist(y, bins=10, edgecolor='black', rwidth=0.8)
plt.title("Distribution of voice and digital categories")#语音和数字类别的分布
plt.xlabel("Digimarc")
plt.ylabel("Sample size")
plt.xticks(range(10))
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("类别直方图.png")
plt.show()


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="tab10", legend='full')
plt.title("Two-dimensional dimensionality reduction (PCA) of MFCC features")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.legend(title="Digimarc")
plt.tight_layout()
plt.savefig("PCA散点图.png")
plt.show()
