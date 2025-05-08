import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import csv
import timm
from safetensors.torch import load_file as safe_load

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 预训练模型的权重文件路径
pretrained_weights_path = os.path.join(current_dir, 'model.safetensors')

# 检查文件是否存在
if not os.path.exists(pretrained_weights_path):
    raise FileNotFoundError(f"Pretrained weights file {pretrained_weights_path} not found.")

# 加载预训练的 ConvNeXt_large 模型并加载权重
model_name = 'convnext_large'  # 使用最大规模的变种
pretrained_model = timm.create_model(model_name, pretrained=False).to(device)
state_dict = safe_load(pretrained_weights_path)
pretrained_model.load_state_dict(state_dict)
pretrained_model.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 图像路径
test_data_path = os.path.join('/', 'testdata')

# 读取图像
def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in tqdm(sorted(os.listdir(folder)), desc="Loading images"):  # 按字典序排序
        img_path = os.path.join(folder, filename)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            with Image.open(img_path) as img:
                # 确保图像是三通道的
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(preprocess(img).unsqueeze(0).to(device))
                filenames.append(os.path.splitext(filename)[0])  # 去掉扩展名
    return images, filenames

# 提取特征
def extract_features(images):
    features = []
    for image in tqdm(images, desc="Extracting features"):
        with torch.no_grad():
            feature = pretrained_model.forward_features(image).mean(dim=(-2, -1)).squeeze()
            features.append(feature)
    return torch.stack(features).to(torch.float32)

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

# 加载测试数据
print("Loading test images...")
test_images, filenames = load_images_from_folder(test_data_path)

# 提取测试集特征
test_features = extract_features(test_images)
X_test_scaled = standardize_tensor(test_features)

# 转换为Tensor
X_test_tensor = X_test_scaled.to(device)

# 加载最终模型
input_dim = X_test_scaled.shape[1]  # 使用测试集的特征维度
hidden_dims = [2048,1024,512,256]
dropout_rate = 0.3

final_model = MLPClassifier(input_dim=input_dim, hidden_dims=hidden_dims,
                            dropout_rate=dropout_rate).to(device)

# 从当前工作目录加载模型文件
model_file = os.path.join(current_dir, 'model2')
if os.path.exists(model_file):
    final_model.load_state_dict(torch.load(model_file, map_location=device))
else:
    raise FileNotFoundError(f"Model file {model_file} not found in the current working directory.")

final_model.eval()

# 预测
with torch.no_grad():
    test_outputs = final_model(X_test_tensor).squeeze().cpu()
    predictions = (test_outputs > 0.5).float()

# 将结果保存到CSV文件
output_file = os.path.join(current_dir, 'cla_pre.csv')  # 保存到当前工作目录

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    for filename, prediction in zip(filenames, predictions):
        writer.writerow([filename, int(prediction)])

print(f"Predictions saved to {output_file}")