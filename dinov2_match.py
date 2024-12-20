import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image
import requests
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# init：device|processor|model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

# image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_features(image_path, model, processor):
    """from path"""
    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model(**inputs).last_hidden_state.mean(dim=1)
    return features.cpu().numpy()

def load_images_and_extract_features_video1(folder_path, max_frames=600):
    features = []
    img_files = sorted(os.listdir(folder_path))[:max_frames] 

    for img_file in tqdm(img_files, desc=f"Processing {os.path.basename(folder_path)}"):
        img_path = os.path.join(folder_path, img_file)
        if img_file.lower().endswith(('png', 'jpg', 'jpeg')):
            feature = extract_features(img_path, model, processor)
            features.append(feature)
    return np.array(features).squeeze()

def load_images_and_extract_features(folder_path):
    features = []
    img_files = sorted(os.listdir(folder_path))
    for img_file in tqdm(img_files, desc=f"Processing {os.path.basename(folder_path)}"):
        img_path = os.path.join(folder_path, img_file)
        if img_file.lower().endswith(('png', 'jpg', 'jpeg')):
            feature = extract_features(img_path, model, processor)
            features.append(feature)
    return np.array(features).squeeze()

horse1_path = "/home/liu.9756/Drone_video/Horse_7_new"
#horse1_features = load_images_and_extract_features_video1(horse1_path, max_frames=600)
horse1_features = load_images_and_extract_features(horse1_path)

horse_3060_base_path = "/home/liu.9756/Drone_video/30_60_sec"
horse_3060_features = []
horse_names = []

pattern = re.compile(r"^Horse_\d+_3060$")
for horse_folder in sorted(os.listdir(horse_3060_base_path)):
    if pattern.match(horse_folder):  
        horse_path = os.path.join(horse_3060_base_path, horse_folder)
        features = load_images_and_extract_features(horse_path)
        horse_3060_features.append(features.mean(axis=0))  
        horse_names.append(horse_folder)

def cosine_similarity(vec1, vec2):
    vec1 = vec1 / np.linalg.norm(vec1, axis=-1, keepdims=True)
    vec2 = vec2 / np.linalg.norm(vec2, axis=-1, keepdims=True)
    return np.dot(vec1, vec2.T)

horse1_avg_feature = horse1_features.mean(axis=0)


similarities = [
    cosine_similarity(horse1_avg_feature.reshape(1, -1), horse_feature.reshape(1, -1))[0, 0]
    for horse_feature in horse_3060_features
]

best_match_index = np.argmax(similarities)
best_match_name = horse_names[best_match_index]
print(f"The first horse in the first video best matches the: {best_match_name}，Similarity: {similarities[best_match_index]:.4f}")

# Hist-plotting..
horse_names = ["Horse_1_3060", "Horse_2_3060", "Horse_3_3060", "Horse_4_3060", "Horse_5_3060", "Horse_7_3060"]

plt.figure(figsize=(10, 6))
colors = ['gray'] * len(similarities)
colors[best_match_index] = 'red'  
plt.bar(horse_names, similarities, alpha=0.8, color=colors)

plt.title("Similarity histogram", fontsize=14)
plt.xlabel("The horses in the second video", fontsize=12)
plt.ylabel("Similarity", fontsize=12)


for i, v in enumerate(similarities):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)

plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()
plt.show()
plt.savefig("similarity_histogram_horse_7.png", dpi=300)





similarity_matrix = pd.DataFrame(
    [similarities],
    index=["Horse_7_new"],
    columns=horse_names
)


plt.figure(figsize=(10, 6))
sns.heatmap(similarity_matrix, annot=True, fmt=".4f", cmap="Blues")
plt.title("Horse similarity matrix")
plt.xlabel("The horses in the second video")
plt.ylabel("Horse from the first video")
plt.show()
plt.savefig("similarity_matrix_horse_7.png", dpi=300)


