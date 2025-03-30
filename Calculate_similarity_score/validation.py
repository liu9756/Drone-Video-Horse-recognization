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



model = DinoClassifier(dino_model, classifier_head).to(device)
model.load_state_dict(torch.load("best_test_2.pt", map_location=device))
model.eval()

if __name__ == '__main__':  
    ref_video_paths = [
        '/home/liu.9756/Drone_video/labeled_Dataset_DJI_0265/'
    ]
    comp_video_paths = [
        '/home/liu.9756/Drone_video/labeled_Dataset_DJI_0267/'
    ]
    print("Preprocessing reference videos...")
    combined_ref = combine_video_features(ref_video_paths, fps=60)
    print("Preprocessing comparison videos...")
    combined_comp = combine_video_features(comp_video_paths, fps=60)

    ref_names = sorted(combined_ref.keys())
    comp_names = sorted(combined_comp.keys())


