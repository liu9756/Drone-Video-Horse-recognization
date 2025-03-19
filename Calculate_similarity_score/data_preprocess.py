import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from data_loading import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = None
model = None

def init_dinov2():
    global processor, model
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base')
    model.config.return_dict = False
    model.eval()
    model.to(device)

init_dinov2()

def extract_feature(image_path):
    # Extract features for images, and do L2 normalize for output
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs[0]  
    feature = last_hidden_state.mean(dim=1).squeeze(0)
    feature = F.normalize(feature, p=2, dim=0)
    return feature.cpu().numpy()

def group_frames_by_second(frame_paths, fps=60):
    groups = []
    for i in range(0, len(frame_paths), fps):
        groups.append(frame_paths[i:i+fps])
    return groups

def process_horse_frames(frame_paths, fps=60):
    grouped_frames = group_frames_by_second(frame_paths, fps)
    pooled_features = []
    #average pooling
    for group in grouped_frames:
        features = []
        for frame in group:
            feat = extract_feature(frame)
            features.append(feat)
        features = np.stack(features, axis=0)  
        avg_feature = np.mean(features, axis=0)
        norm = np.linalg.norm(avg_feature)
        if norm > 0:
            avg_feature = avg_feature / norm
        pooled_features.append(avg_feature)
    return pooled_features

def data_preprocess(video_path, fps=60):
    loader = DataLoader(video_path)
    raw_data = loader.load_data()  
    processed_data = {}
    for segment, horses in raw_data.items():
        print(f"Processing segment: {segment}")
        processed_data[segment] = {}
        for horse, frame_paths in horses.items():
            print(f"Processing {horse} with {len(frame_paths)} frames")
            pooled_features = process_horse_frames(frame_paths, fps)
            print(f"Obtained {len(pooled_features)} seconds of pooled features")
            processed_data[segment][horse] = pooled_features

    return processed_data

if __name__ == '__main__':
    video_path = '/home/liu.9756/Drone_video/labeled_Dataset_DJI_0265/'  
    processed_data = data_preprocess(video_path, fps=60)
    for segment, horses in processed_data.items():
        for horse, features in horses.items():
            print(f"Segment: {segment}, {horse} => {len(features)} seconds processed.")