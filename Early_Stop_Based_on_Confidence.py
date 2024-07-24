import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np

# 读取数据
df = pd.read_excel('cgls_results_with_labels.xlsx')

# 分离特征和标签
X = df[['Norms1/Norms0', 'Norms2/Norms0', 'Norms3/Norms0', 'Norms4/Norms0', 'Norms5/Norms0']].values
y = df['Label'].values

# 转换为PyTorch张量
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# 创建数据集和数据加载器
dataset = TensorDataset(X, y)
train_size = int(0.7 * len(dataset))
val_size = (len(dataset) - train_size) // 2
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch[:, :1])  # 只用第一个特征训练
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    # 在验证集上评估模型
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch[:, :1])
            val_loss += criterion(outputs, y_batch).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
    
    val_loss /= len(val_loader)
    val_accuracy = correct / len(val_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}')

# 保存模型
torch.save(model.state_dict(), 'mlp_model.pth')

# 定义置信度计算和提前停止函数
def compute_confidence_and_predict(model, X_row):
    confidences = []
    for i in range(X_row.shape[0]):
        x = torch.tensor(X_row[i:i+1].reshape(-1, 1), dtype=torch.float32)
        with torch.no_grad():
            output = model(x)
            probs = torch.softmax(output, dim=1).numpy()[0]
        label = np.argmax(probs)
        confidence = probs[label]
        confidences.append((label, confidence))
        print(f'Norms{i+1}/Norms0: {X_row[i]:.6f}, Label: {label}, Confidence: {confidence:.2%}')
        if confidence >= 1.0:  # 如果置信度达到 100%，提前停止
            break
    return confidences

# 在测试集上评估模型并输出每个Norms/norms0的预测结果和置信度
model.eval()
for i in range(test_dataset.__len__()):
    X_row = X[test_dataset.indices[i]]
    print(f'Testing row {i+1}:')
    compute_confidence_and_predict(model, X_row)
    print('-' * 50)
