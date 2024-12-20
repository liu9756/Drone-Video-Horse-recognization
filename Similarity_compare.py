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
    """Extract features for each second by averaging frame features."""
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
    return np.array(features)

def compute_similarity_matrix(train_set_path, test_set_path):
    """Compute similarity matrix between train set and test set."""
    train_horses = sorted(os.listdir(train_set_path))
    test_horses = sorted(os.listdir(test_set_path))

    similarity_matrix = []

    for train_horse in tqdm(train_horses, desc="Processing Training Horses"):
        train_path = os.path.join(train_set_path, train_horse)
        train_features = extract_features_per_second(train_path)

        similarities = []
        for test_horse in test_horses:
            test_path = os.path.join(test_set_path, test_horse)
            test_features = extract_features_per_second(test_path)

            second_similarities = []
            for train_sec in train_features:
                sec_similarities = [
                    np.dot(train_sec.squeeze() / np.linalg.norm(train_sec.squeeze()),test_sec.squeeze() / np.linalg.norm(test_sec.squeeze()))
                    for test_sec in test_features
                ]
                second_similarities.append(np.mean(sec_similarities))

            similarities.append(np.mean(second_similarities))  # Average similarity for train_horse vs test_horse

        similarity_matrix.append(similarities)

    similarity_matrix = np.array(similarity_matrix)
    return similarity_matrix, train_horses, test_horses

def plot_confusion_matrix(similarity_matrix, train_horses, test_horses, output_path="confusion_matrix.png"):
    """Plot and save the confusion matrix."""
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

train_set_path = "/home/liu.9756/Drone_video/DJI_0265_0_30_s"  
test_set_path = "/home/liu.9756/Drone_video/DJI_0265_30_60_s"  

similarity_matrix, train_horses, test_horses = compute_similarity_matrix(train_set_path, test_set_path)
plot_confusion_matrix(similarity_matrix, train_horses, test_horses)
