import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
#from sklearn.metrics import top_k_accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model(**inputs).last_hidden_state.mean(dim=1)
    return features.cpu().numpy()

def extract_features_per_second(folder_path, frames_per_second=60):
    features = []
    img_files = sorted(os.listdir(folder_path))

    # Group frames by seconds (1 sec = 60 frames in this case)
    for i in range(0, len(img_files), frames_per_second):
        batch_features = []
        for img_file in img_files[i:i + frames_per_second]:
            if img_file.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(folder_path, img_file)
                feature = extract_features(img_path)
                batch_features.append(feature)
        if batch_features:
            features.append(np.mean(batch_features, axis=0))  # Average features for this second
    return np.array(features), len(img_files), len(img_files) // frames_per_second

def extract_horse_features_multiple_dirs(horse_dir_list, frames_per_second=60):
    all_features = []
    total_frames = 0
    total_seconds = 0
    
    for horse_dir in horse_dir_list:
        feats, frames, secs = extract_features_per_second(horse_dir, frames_per_second)
        # feats 现在可能是 (N,1,768) 或 (N,768)，我们做一次 squeeze
        feats = np.squeeze(feats)  # 默认把所有 size=1 的维度挤掉
        # 如果 feats 原本是 (N, 768)， squeeze 后还是 (N, 768)；如果是 (N,1,768)，则变成 (N,768)
        
        # 如果只想要 2D 结构，可做一下判断:
        if feats.ndim == 1:
            # 说明只有一个秒级特征
            feats = feats[np.newaxis, :]  # (1, 768)
        
        all_features.append(feats)
        total_frames += frames
        total_seconds += secs
    
    if len(all_features) == 0:
        return None, 0, 0
    
    # all_features 里每个元素现在是 (N_i, 768)，拼起来
    all_features = np.concatenate(all_features, axis=0)  # (sum_N, 768)
    # 对所有特征再做一次平均，得到 (768,)
    final_feature = np.mean(all_features, axis=0)  # shape (768,)
    # 再扩展成 (1,768) 以便后续 dot
    final_feature = np.expand_dims(final_feature, axis=0)  # shape (1,768)
    
    return final_feature, total_frames, total_seconds


def compute_similarity_matrix_multiple(train_folders_dict, test_folders_dict, frames_per_second=60):
    """
    train_folders_dict: { horse_name: [list_of_dirs_for_that_horse_in_train] }
    test_folders_dict:  { horse_name: [list_of_dirs_for_that_horse_in_test] }
    
    返回:
      similarity_matrix: (num_train_horses, num_test_horses)
      train_horses: 训练马匹名字列表
      test_horses: 测试马匹名字列表
      train_stats: { horse_name: (frames, seconds) } (合并后的统计)
      test_stats:  { horse_name: (frames, seconds) }
    """
    train_horses = sorted(train_folders_dict.keys())
    test_horses = sorted(test_folders_dict.keys())
    
    similarity_matrix = []
    train_stats = {}
    test_stats = {}
    
    # 先把 test_horses 每个都提取好，后面在循环train时可直接使用
    test_features_dict = {}
    for test_horse in tqdm(test_horses, desc="Extracting Test Horses"):
        feature_vec, frames, secs = extract_horse_features_multiple_dirs(
            test_folders_dict[test_horse], frames_per_second
        )
        test_features_dict[test_horse] = feature_vec  # (1, feature_dim)
        test_stats[test_horse] = (frames, secs)
    
    # 逐个训练马匹
    for train_horse in tqdm(train_horses, desc="Processing Training Horses"):
        train_feature_vec, train_frames, train_secs = extract_horse_features_multiple_dirs(
            train_folders_dict[train_horse], frames_per_second
        )
        train_stats[train_horse] = (train_frames, train_secs)
        
        # 与所有测试马匹做相似度
        row_sim = []
        for test_horse in test_horses:
            test_feature_vec = test_features_dict[test_horse]
            if train_feature_vec is None or test_feature_vec is None:
                row_sim.append(0.0)
                continue
            
            # 取 (1, feature_dim)，计算余弦相似度
            train_norm = train_feature_vec / np.linalg.norm(train_feature_vec, axis=1, keepdims=True)
            test_norm  = test_feature_vec  / np.linalg.norm(test_feature_vec, axis=1, keepdims=True)
            similarity = np.dot(train_norm, test_norm.T).squeeze()  # scalar
            row_sim.append(similarity)
        
        similarity_matrix.append(row_sim)
    
    similarity_matrix = np.array(similarity_matrix)
    return similarity_matrix, train_horses, test_horses, train_stats, test_stats

def calculate_top_k_accuracy(logits, targets, k=2):
    values, indices = torch.topk(logits, k=k, sorted=True)
    y = torch.reshape(targets, [-1, 1])
    correct = (y == indices) * 1.  # Compare predictions with ground truth
    top_k_accuracy = torch.mean(correct) * k  # Calculate final accuracy
    return top_k_accuracy

def compute_top_k_accuracy(similarity_matrix, k=1):
    logits = torch.tensor(similarity_matrix)
    targets = torch.arange(len(similarity_matrix))
    return calculate_top_k_accuracy(logits, targets, k=k).item()

def plot_confusion_matrix(similarity_matrix, train_horses, test_horses, output_path="confusion_matrix_2.png"):
    predicted_labels = np.argmax(similarity_matrix, axis=1)
    true_labels = range(len(train_horses))

    confusion_matrix = pd.DataFrame(
        similarity_matrix,
        index=[f"True: {label}" for label in train_horses],
        columns=[f"Predicted: {label}" for label in test_horses]
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Confusion Matrix Based on Similarity")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def gather_horse_folders(video_dirs):
    """
    输入：多个视频文件夹的列表 video_dirs。
    假设每个视频文件夹下都包含若干马匹子文件夹（horse1, horse2, ...）。
    
    输出：一个字典 horse_folders_dict
        key：马匹名称 (str)
        val：该马匹在所有video_dirs中对应的子文件夹路径列表 (List[str])
    """
    horse_folders_dict = defaultdict(list)
    
    for video_dir in video_dirs:
        # 这个video_dir应该是一个大的视频文件夹，比如 "/path/to/train_video1"
        if not os.path.isdir(video_dir):
            continue

        # 获取该video_dir里所有的子文件夹(各个马匹)
        horses_in_dir = sorted(os.listdir(video_dir))
        for horse_name in horses_in_dir:
            sub_path = os.path.join(video_dir, horse_name)
            if os.path.isdir(sub_path):
                horse_folders_dict[horse_name].append(sub_path)

    return horse_folders_dict


train_paths = [
    "/home/liu.9756/Drone_video/DJI_0265_0_30_s/",
    "/home/liu.9756/Drone_video/DJI_0265_30_60_s/",
    "/home/liu.9756/Drone_video/labeled_Dataset_DJI_0266_60_120/crop/"
]
test_paths = [
    "/home/liu.9756/Drone_video/labeled_DJI_0267_0_60/crop/",
    "/home/liu.9756/Drone_video/labeled_DJI_0267_120_180/crop/",
    "/home/liu.9756/Drone_video/labeled_Dataset_DJI_0268_0_30/crop/"
] 

# 2) 收集各匹马的子文件夹
train_folders_dict = gather_horse_folders(train_paths)
test_folders_dict  = gather_horse_folders(test_paths)

# 3) 计算相似度矩阵
similarity_matrix, train_horses, test_horses, train_stats, test_stats = compute_similarity_matrix_multiple(
    train_folders_dict, 
    test_folders_dict, 
    frames_per_second=60
)

# 4) 计算 Top-k Accuracy
for k in [1, 3, 5]:
    top_k_accuracy_val = compute_top_k_accuracy(similarity_matrix, k=k)
    print(f"Top-{k} Accuracy: {top_k_accuracy_val:.2%}")

# 5) 打印统计信息
print("\nTraining Set Statistics:")
for horse, (frames, seconds) in train_stats.items():
    print(f"{horse}: {frames} frames, ~{seconds} seconds")

print("\nTesting Set Statistics:")
for horse, (frames, seconds) in test_stats.items():
    print(f"{horse}: {frames} frames, ~{seconds} seconds")

# 6) 绘制混淆矩阵
plot_confusion_matrix(similarity_matrix, train_horses, test_horses)
