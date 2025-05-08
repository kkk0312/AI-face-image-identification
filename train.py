import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import timm
import requests
# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
# 设置全局超时时间为 30 秒
requests.Session().get = lambda *args, **kwargs: requests.Session().get(*args, timeout=30, **kwargs)
# 加载预训练的 ConvNeXt_xlarge 模型
model_name = 'convnext_large'# 使用最大规模的变种
pretrained_model = timm.create_model(model_name, pretrained=True).to(device)
pretrained_model.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 图像路径
real_path = r"C:\py\wj\Pycharmobject\PyCharm Community Edition 2024.1.3\real_images0"
fake_path = r"C:\py\wj\Pycharmobject\PyCharm Community Edition 2024.1.3\fake_images0"
# 读取图像
def load_images_from_folder(folder):
    images = []
    for filename in tqdm(os.listdir(folder), desc="Loading images"):
        img_path = os.path.join(folder, filename)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            with Image.open(img_path) as img:
                # 确保图像是三通道的
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(preprocess(img).unsqueeze(0).to(device))
    return images

# 提取特征
def extract_features(images):
    features = []
    for image in tqdm(images, desc="Extracting features"):
        with torch.no_grad():
            feature = pretrained_model.forward_features(image).mean(dim=(-2, -1)).squeeze()
            features.append(feature)
    return torch.stack(features).to(torch.float32)

# 检查是否已经保存了训练集特征
features_file = r'C:\py\wj\test\train_features.pth'
labels_file = r'C:\py\wj\test\train_labels.pth'

# 如果文件存在，则加载已保存的训练集特征和标签
if os.path.exists(features_file) and os.path.exists(labels_file):
    print("Loading saved features and labels...")
    X_train = torch.load(features_file, map_location=device)
    y_train = torch.load(labels_file, map_location=device)

    # 输出保存的图片数量和特征数量
    num_real_images = (y_train == 0).sum().item()
    num_fake_images = (y_train == 1).sum().item()
    num_features = X_train.shape[0]
    print(f"Number of real images: {num_real_images}")
    print(f"Number of fake images: {num_fake_images}")
    print(f"Total number of features: {num_features}")
else:
    # 如果文件不存在，则从图像文件夹中提取特征
    print("Loading real images...")
    real_images = load_images_from_folder(real_path)
    print("Loading fake images...")
    fake_images = load_images_from_folder(fake_path)

    print("Extracting real features...")
    real_features = extract_features(real_images)
    print("Extracting fake features...")
    fake_features = extract_features(fake_images)

    # 创建数据集
    X_train = torch.cat((real_features, fake_features), dim=0)
    y_train = torch.cat((torch.zeros(len(real_features)), torch.ones(len(fake_features))), dim=0)

    # 保存合并后的训练集特征和标签
    torch.save(X_train, features_file)
    torch.save(y_train, labels_file)

    # 输出保存的图片数量和特征数量
    num_real_images = len(real_images)
    num_fake_images = len(fake_images)
    num_features = X_train.shape[0]
    print(f"Number of real images: {num_real_images}")
    print(f"Number of fake images: {num_fake_images}")
    print(f"Total number of features: {num_features}")

# 定义多层感知机（MLP）
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.5):
        super(MLPClassifier, self).__init__()
        layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze()

# 数据标准化（在GPU上）
def standardize_tensor(tensor):
    mean = tensor.mean(dim=0)
    std = tensor.std(dim=0)
    return (tensor - mean) / std

# 训练函数
def train_model(X_train, y_train, params):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化模型
    model = MLPClassifier(input_dim=X_train.shape[1], hidden_dims=params['hidden_dims'],
                          dropout_rate=params['dropout_rate']).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['l2_lambda'])
    scheduler = CosineAnnealingLR(optimizer, T_max=params['num_epochs'])

    # 数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    batch_size = int(params['batch_size'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    num_epochs = params['num_epochs']
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")
        scheduler.step()

    return model

# 参数设置
input_dim = X_train.shape[1]

# 固定超参数
params = {
    'hidden_dims': [2048,1024,512,256],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'l2_lambda': 0.001,
    'batch_size': 128,
    'num_epochs': 20
}

# 使用最佳超参数重新训练完整模型
X_train_scaled = standardize_tensor(X_train)
y_train_tensor = y_train.to(device)

# 训练最终模型
final_model = train_model(X_train_scaled, y_train_tensor, params)

# 保存最终模型
torch.save(final_model.state_dict(), r'C:\py\wj\test\model2')