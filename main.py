import os
import torch
import shutil  # 用于移动文件
import numpy as np
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models
from torchvision.models import ResNet18_Weights
from PIL import Image
from tqdm import tqdm  # 进度条


# 获取图片路径
def get_image_paths(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, file))
    return image_paths


# 运行时输入文件夹地址
folder_path = input("请输入图片文件夹地址：")
if not os.path.exists(folder_path):
    print("文件夹不存在，请检查路径！")
    exit()

image_paths = get_image_paths(folder_path)
print(f"Found {len(image_paths)} images.")

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载预训练模型（去掉最后一层全连接）
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])  # 提取全局平均池化前的特征
model = model.to(device)  # 将模型移动到GPU（如果可用）
model.eval()  # 设置为评估模式

# 图片预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整大小
    transforms.ToTensor(),  # 转为Tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )  # ImageNet标准化
])


# 提取特征向量
def extract_features(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)  # 增加batch维度并移动到GPU
        with torch.no_grad():
            features = model(image_tensor)
        return features.cpu().squeeze().numpy()  # 转为numpy数组并移回CPU
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


# 提取所有图片的特征向量
features_dict = {}
for path in tqdm(image_paths, desc="Extracting features"):
    feature = extract_features(path)
    if feature is not None:
        features_dict[path] = feature

# 将特征向量堆叠成矩阵
paths = list(features_dict.keys())
features = np.array([features_dict[path] for path in paths])

# 计算余弦相似度矩阵
print("Calculating similarity matrix...")
similarity_matrix = cosine_similarity(features)

# 筛选重复图片
duplicates = set()
threshold = 0.98  # 可调整阈值（0.9~0.99）

for i in tqdm(range(len(similarity_matrix)), desc="Finding duplicates"):
    for j in range(i + 1, len(similarity_matrix)):
        if similarity_matrix[i][j] > threshold:
            duplicates.add((paths[i], paths[j]))

# 创建“重复图片”文件夹
duplicates_folder = os.path.join(folder_path, "重复图片")
if not os.path.exists(duplicates_folder):
    os.makedirs(duplicates_folder)

# 移动重复图片到子文件夹
for idx, pair in enumerate(duplicates):
    # 创建子文件夹
    subfolder = os.path.join(duplicates_folder, f"重复_{idx + 1}")
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    # 移动图片
    for image_path in pair:
        if os.path.exists(image_path):
            shutil.move(image_path, subfolder)
            print(f"Moved {image_path} to {subfolder}")

print("重复图片已移动到“重复图片”文件夹。")
