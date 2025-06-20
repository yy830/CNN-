# -*- coding: utf-8 -*-
# train_cnn.py
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model_cnn import CNNClassifier

class MFCCDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, 13, T)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 1. 加载数据
data = np.load("digit_mfcc_cnn.npz", allow_pickle=True)
X, y = data["X"], data["y"]

# 2. 分训练/测试
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 3. 创建数据集和加载器
train_dataset = MFCCDataset(X_train, y_train)
test_dataset = MFCCDataset(X_test, y_test)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0  # Windows下建议设为0
)
test_loader = DataLoader(test_dataset, batch_size=32)

# 4. 初始化模型
model = CNNClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 5. 训练配置
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 添加调试信息
        print(f"Input batch shape: {data.shape}")
        
        optimizer.zero_grad()
        output = model(data)
        
        # 添加调试信息
        print(f"Output shape: {output.shape}")
        print(f"Target shape: {target.shape}")
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
# 6. 训练模型
num_epochs = 10
best_acc = 0.0

print("Start training...")
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    # 验证阶段
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    acc = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}, Accuracy: {acc:.2f}%')
    
    # 保存最佳模型
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pth")

print(f"Training completed! Best accuracy rate:{best_acc:.2f}%")

train(model, train_loader, criterion, optimizer, device)

# 7. 保存模型
torch.save(model.state_dict(), "cnn_model.pth")
print("CNN has been saved!")
