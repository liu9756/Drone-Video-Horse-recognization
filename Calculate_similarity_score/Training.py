import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from utils import gather_data_from_videos,create_label_dict,get_confusion_matrix, get_classification_report
from visualization import plot_confusion_matrix
from transformers import AutoImageProcessor, AutoModel, Dinov2Config,Dinov2PreTrainedModel
from tqdm import tqdm
import argparse

class VideoDataset(Dataset):
    def __init__(self, data_list, label_dict, transform=None):
        """
        data_list: List of (img_path, horse_name) or (img_path, label)
        label_dict: dict {horse_name: label_id}
        transform: preprocess transforms
        """
        self.data_list = data_list
        self.label_dict = label_dict
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, horse_name = self.data_list[idx]
        label = self.label_dict[horse_name]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label



def main(args):
    train_dir_list = [
        "/home/liu.9756/Drone_video/labeled_Dataset_DJI_0265/",
        "/home/liu.9756/Drone_video/labeled_Dataset_DJI_0266/"
    ]

    test_dir_list = [
        "/home/liu.9756/Drone_video/labeled_Dataset_DJI_0267/",
        "/home/liu.9756/Drone_video/labeled_Dataset_DJI_0268/",
        "/home/liu.9756/Drone_video/labeled_Dataset_DJI_0269/"
    ]

    train_data_list, train_horse_set = gather_data_from_videos(train_dir_list)
    test_data_list, test_horse_set  = gather_data_from_videos(test_dir_list)

    all_horses = train_horse_set.union(test_horse_set)
    label_dict = create_label_dict(all_horses)

    print("Total classes:", len(label_dict))
    print("Class mapping:")
    for horse_name, label in sorted(label_dict.items(), key=lambda x: x[1]):
        print(f"  {label}: {horse_name}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])


    train_dataset = VideoDataset(train_data_list, label_dict, transform=train_transform)
    test_dataset  = VideoDataset(test_data_list,  label_dict, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    dino_model.eval()
    for param in dino_model.parameters():
        param.requires_grad = False

    feature_dim = 768 
    num_classes = len(label_dict)
    classifier_head = nn.Sequential(
        nn.Linear(feature_dim, 512),
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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier_head.parameters(), lr=1e-4)

    num_epochs = args.num_epochs
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    model.eval()
    all_preds = []
    all_labels = []
    pbar = tqdm(test_loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), "best.pt")
            print(f"New best model saved with accuracy: {accuracy*100:.2f}%")

    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load("best.pt"))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Final Evaluation", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    final_acc = (all_preds == all_labels).mean()
    print(f"Final Test Accuracy: {final_acc*100:.2f}%")

    cm = get_confusion_matrix(all_labels, all_preds)
    index_to_name = sorted(label_dict, key=lambda k: label_dict[k])
    plot_confusion_matrix(cm, index_to_name)
    report = get_classification_report(all_labels, all_preds, index_to_name, 3)
    print(report)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DINOv2 Horse Classification Training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()

    main(args)
