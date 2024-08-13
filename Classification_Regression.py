import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd
import numpy as np

# 数据加载和预处理
data = pd.read_csv('cgls_result4.csv')

# 特征和标签
features = data[['M', 'N'] + [f'Norm{i}/Norm0' for i in range(1, 21)]]
labels = data['Iter1']
labels_class = (labels != -1).astype(int)  # 分类标签：0表示不能收敛，1表示能收敛

# 数据标准化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 数据划分
X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
    features_scaled, labels_class, labels, test_size=0.2, random_state=42
)

# 分类模型
class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(22, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))

# 回归模型
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(22, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# 初始化模型、损失函数和优化器
classification_model = ClassificationModel()
regression_model = RegressionModel()

criterion_class = nn.BCELoss()
criterion_reg = nn.MSELoss()
optimizer_class = optim.Adam(classification_model.parameters(), lr=0.001)
optimizer_reg = optim.Adam(regression_model.parameters(), lr=0.001)

# 训练分类模型
for epoch in range(100):  # 可调整训练轮数
    classification_model.train()
    optimizer_class.zero_grad()
    outputs = classification_model(torch.tensor(X_train, dtype=torch.float32))
    loss = criterion_class(outputs, torch.tensor(y_train_class.values, dtype=torch.float32).view(-1, 1))
    loss.backward()
    optimizer_class.step()

    if epoch % 10 == 0:
        classification_model.eval()
        with torch.no_grad():
            preds = classification_model(torch.tensor(X_test, dtype=torch.float32)).round()
            accuracy = accuracy_score(y_test_class, preds.numpy())
            print(f'Epoch {epoch}, Classification Accuracy: {accuracy:.4f}')

# 保存分类模型
torch.save(classification_model.state_dict(), 'classification_model.pth')

# 对能收敛的数据进行回归训练
X_train_reg = X_train[y_train_class == 1]
y_train_reg = y_train_reg[y_train_class == 1]

for epoch in range(100):  # 可调整训练轮数
    regression_model.train()
    optimizer_reg.zero_grad()
    outputs = regression_model(torch.tensor(X_train_reg, dtype=torch.float32))
    loss = criterion_reg(outputs, torch.tensor(y_train_reg.values, dtype=torch.float32).view(-1, 1))
    loss.backward()
    optimizer_reg.step()

    if epoch % 10 == 0:
        regression_model.eval()
        with torch.no_grad():
            preds = regression_model(torch.tensor(X_test[y_test_class == 1], dtype=torch.float32))
            mse = mean_squared_error(y_test_reg[y_test_class == 1], preds.numpy())
            print(f'Epoch {epoch}, Regression MSE: {mse:.4f}')

# 保存回归模型
torch.save(regression_model.state_dict(), 'regression_model.pth')

# 置信度计算函数（贝叶斯推断）
def calculate_confidence(predicted_iter, reg_output, uncertainty=1.0):
    # 这里的置信度是根据预测值和回归模型输出的差异以及不确定性来估计的
    variance = np.abs(predicted_iter - reg_output) / uncertainty
    confidence = 1 / (1 + variance)
    return confidence

# 定义预测函数
def predict_iter(input_data):
    classification_model.eval()
    regression_model.eval()

    # 分类模型预测
    with torch.no_grad():
        is_converged = classification_model(torch.tensor([input_data], dtype=torch.float32)).item()

    if is_converged >= 0.5:
        # 回归模型预测
        with torch.no_grad():
            predicted_iter = regression_model(torch.tensor([input_data], dtype=torch.float32)).item()
            confidence = calculate_confidence(predicted_iter, regression_model(torch.tensor([input_data], dtype=torch.float32)).item())
        return predicted_iter, confidence
    else:
        return -1, 1.0  # 不收敛的情况，置信度为1

# 测试模型
example_input = [16384	,1024,	0.296528,	0.0970483	,0.0320176,	0.010795	,0.00357498	,0.00115623,	0.000367411,	0.000120232	,0.0000377, 	0.000012,	0.0000039,	0.00000125,	0.000000415,	0.000000135,	0.0000000438,	0.0000000137,	0.00000000441,	0.00000000139,	0.000000000436,	0.000000000135]
# 标准化输入数据
example_input_scaled = scaler.transform([example_input])

# 预测Iter1和置信度
predicted_iter, confidence = predict_iter(example_input_scaled[0])
print(f'Predicted Iter1: {predicted_iter}, Confidence: {confidence:.4f}')



