# -*- coding: utf-8 -*-
# model_cnn.py
import torch
import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNClassifier, self).__init__()
        
        # CNN特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 分类器层 - 使用与保存模型相同的结构
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 13, 8)
            out = self.features(dummy_input)
            flattened_size = out.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
            # 计算展平后的特征维度
        self.features_dim = 2304  # 根据实际的特征图大小计算
        
        self.classifier = nn.Sequential(
            nn.Linear(self.features_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x