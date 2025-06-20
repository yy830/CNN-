import os
import librosa
import numpy as np

DATASET_PATH = "C:/Users/daiyan/Desktop/Stu/AI_design/free-spoken-digit-dataset-master/recordings"

def prepare_dataset():
    X, y = [], []
    print("开始提取特征...")

    # 检查数据集路径是否存在
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"找不到数据集路径: {DATASET_PATH}")

    # 获取所有wav文件
    wav_files = [f for f in os.listdir(DATASET_PATH) if f.endswith('.wav')]
    
    if not wav_files:
        raise ValueError(f"在 {DATASET_PATH} 中没有找到.wav文件")

    for filename in wav_files:
        try:
            label = int(filename[0])  # 获取文件名中的第一个数字作为标签
            file_path = os.path.join(DATASET_PATH, filename)
            
            # 加载音频文件
            y_audio, sr = librosa.load(file_path, sr=16000)
            
            # 提取MFCC特征
            mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
            
            # 确保特征维度一致（可以选择填充或裁剪）
            target_length = 50  # 设置一个固定长度
            if mfcc.shape[1] > target_length:
                mfcc = mfcc[:, :target_length]
            elif mfcc.shape[1] < target_length:
                pad_width = ((0, 0), (0, target_length - mfcc.shape[1]))
                mfcc = np.pad(mfcc, pad_width, mode='constant')
            
            X.append(mfcc)
            y.append(label)
            
            if len(X) % 100 == 0:
                print(f"已处理 {len(X)} 个文件...")
                
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
            continue

    if not X:
        raise ValueError("没有成功处理任何音频文件！")

    X = np.array(X)
    y = np.array(y)
    
    print(f"数据集形状: X={X.shape}, y={y.shape}")
    
    # 保存数据集
    np.savez("digit_mfcc_cnn.npz", X=X, y=y)
    print(f"✅ 数据集已保存！共处理 {len(X)} 个样本")
    
    return X, y

if __name__ == "__main__":
    try:
        prepare_dataset()
    except Exception as e:
        print(f"错误: {str(e)}")
        print("\n请确保:")
        print("1. 数据集文件夹路径正确")
        print("2. 文件夹中包含.wav文件")
        print("3. 文件名格式正确（第一个字符为数字标签）")