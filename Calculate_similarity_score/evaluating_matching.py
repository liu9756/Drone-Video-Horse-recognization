import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from scipy.optimize import linear_sum_assignment
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
import argparse


def load_images_from_video(video_path):
    horse_images = {}
    for segment in sorted(os.listdir(video_path)):
        segment_path = os.path.join(video_path, segment)
        crop_path = os.path.join(segment_path, "crop")
        if not os.path.exists(crop_path):
            continue
        for horse in sorted(os.listdir(crop_path)):
            horse_path = os.path.join(crop_path, horse)
            if horse not in horse_images:
                horse_images[horse] = []
            for img_file in sorted(os.listdir(horse_path)):
                if img_file.lower().endswith(('.jpg', '.png')):
                    horse_images[horse].append(os.path.join(horse_path, img_file))
    return horse_images


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
dino_model.eval()
for param in dino_model.parameters():
    param.requires_grad = False

num_classes = 7 
classifier_head = nn.Sequential(
    nn.Linear(768, 512),
    nn.ReLU(),
    nn.Linear(512, num_classes)
).to(device)


class DinoClassifier(nn.Module):
    def __init__(self, dino_model, classifier):
        super().__init__()
        self.dino_model = dino_model
        self.classifier = classifier  
    def forward(self, x):
        with torch.no_grad():
            outputs = self.dino_model(x)
            feats = outputs.last_hidden_state.mean(dim=1)  
        intermediate = self.classifier[0](feats)       
        activated = self.classifier[1](intermediate)   
        logits = self.classifier[2](activated)         
        return logits

    def extract_features(self, x):
        with torch.no_grad():
            outputs = self.dino_model(x)
            feats = outputs.last_hidden_state.mean(dim=1)
        intermediate = self.classifier[0](feats)
        activated = self.classifier[1](intermediate)
        return activated

model = DinoClassifier(dino_model, classifier_head).to(device)
print("Loading best model from best_test_2.pt ...")
model.load_state_dict(torch.load("best_test_2.pt", map_location=device))
model.eval()

def extract_intermediate_feature(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    feat = model.extract_features(image) 
    feat = F.normalize(feat, p=2, dim=1)
    return feat.squeeze(0).detach().cpu().numpy()

def combine_horse_features(horse_img_list):
    feats = []
    for img_path in tqdm(horse_img_list, desc="Extracting features", leave=False):
        feat = extract_intermediate_feature(img_path)
        feats.append(feat)
    feats = np.stack(feats, axis=0)
    avg_feat = np.mean(feats, axis=0)
    norm = np.linalg.norm(avg_feat)
    if norm > 0:
        avg_feat = avg_feat / norm
    return avg_feat

def combine_video_features(video_path):
    horse_images = load_images_from_video(video_path)
    combined = {}
    for horse, img_list in horse_images.items():
        print(f"Processing {horse} with {len(img_list)} images")
        combined[horse] = combine_horse_features(img_list)
    return combined

def compute_similarity_matrix(agg_ref, agg_comp, sim_metric='euclidean'):
    ref_names = sorted(agg_ref.keys())
    comp_names = sorted(agg_comp.keys())
    ref_vecs = np.stack([agg_ref[name] for name in ref_names], axis=0)
    comp_vecs = np.stack([agg_comp[name] for name in comp_names], axis=0)
    if sim_metric == 'euclidean':
        diff = ref_vecs[:, None, :] - comp_vecs[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        sim_matrix = 1.0 / (1.0 + dist)
    else:
        sim_matrix = np.dot(ref_vecs, comp_vecs.T)
        sim_matrix = (sim_matrix + 1) / 2
    return sim_matrix, ref_names, comp_names

def bipartite_match(sim_matrix):
    cost_matrix = -sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind

if __name__ == '__main__':
    ref_video_paths = '/home/liu.9756/Drone_video/labeled_Dataset_DJI_0266/'
    
    comp_video_paths = '/home/liu.9756/Drone_video/labeled_Dataset_DJI_0267/'


    print("Processing reference video:")
    ref_features = combine_video_features(ref_video_paths)
    print("Processing comparison video:")
    comp_features = combine_video_features(comp_video_paths)

    sim_matrix, ref_names, comp_names = compute_similarity_matrix(ref_features, comp_features, sim_metric='euclidean')
    row_ind, col_ind = bipartite_match(sim_matrix)
    print("Bipartite Matching Results:")
    for i, j in zip(row_ind, col_ind):
        print(f"  {ref_names[i]}  <-->  {comp_names[j]}, Score: {sim_matrix[i, j]:.4f}")
    import matplotlib.pyplot as plt
    def visualize_similarity(sim_matrix, ref_names, comp_names, title, save_path):
        plt.figure(figsize=(8, 6))
        im = plt.imshow(sim_matrix, interpolation='nearest', cmap='viridis')
        plt.colorbar(im)
        plt.xticks(np.arange(len(comp_names)), comp_names, rotation=45)
        plt.yticks(np.arange(len(ref_names)), ref_names)
        plt.title(title)
        for i in range(sim_matrix.shape[0]):
            for j in range(sim_matrix.shape[1]):
                plt.text(j, i, f"{sim_matrix[i, j]:.2f}",
                         ha="center", va="center", color="white", fontsize=9)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    visualize_similarity(sim_matrix, ref_names, comp_names, "Similarity Heatmap", "similarity_heatmap.png")
    print("Similarity heatmap saved as similarity_heatmap.png")