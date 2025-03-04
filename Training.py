import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, AutoModel

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


def gather_data_from_videos(video_dirs):
    """
    video_dirs: [path_to_video1, path_to_video2, ...]
    horse1, horse2, ...。
    return: data_list = [(img_path, horse_name), ...]
           horse_set = set([horse_name1, horse_name2, ...])
    """
    data_list = []
    horse_set = set()

    for vdir in video_dirs:
        if not os.path.isdir(vdir):
            continue
        horses = sorted(os.listdir(vdir))
        for hname in horses:
            hpath = os.path.join(vdir, hname)
            if not os.path.isdir(hpath):
                continue
            # horse hname
            horse_set.add(hname)

            # collect images
            for fname in os.listdir(hpath):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fpath = os.path.join(hpath, fname)
                    data_list.append((fpath, hname))
    return data_list, horse_set

def create_label_dict(horses):
    """
    horses: iterable of horse_names (strings)
    return: {horse_name: label_id}
    """
    horse_list = sorted(list(horses))
    return { horse_name: idx for idx, horse_name in enumerate(horse_list) }



train_dir_list = [
    "/home/liu.9756/Drone_video/DJI_0265_0_30_s/",
    "/home/liu.9756/Drone_video/DJI_0265_30_60_s/",
    "/home/liu.9756/Drone_video/labeled_Dataset_DJI_0266_60_120/crop/"
]

test_dir_list = [
    "/home/liu.9756/Drone_video/labeled_DJI_0267_0_60/crop/",
    "/home/liu.9756/Drone_video/labeled_DJI_0267_120_180/crop/",
    "/home/liu.9756/Drone_video/labeled_Dataset_DJI_0268_0_30/crop/"
] 

train_data_list, train_horse_set = gather_data_from_videos(train_dir_list)
test_data_list, test_horse_set  = gather_data_from_videos(test_dir_list)

# Merging objects ...:
all_horses = train_horse_set.union(test_horse_set)
label_dict = create_label_dict(all_horses)  

print("Total classes:", len(label_dict))

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalize 
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = VideoDataset(train_data_list, label_dict, transform=transform)
test_dataset  = VideoDataset(test_data_list,  label_dict, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
dino_model.eval()   

# froze the para in dino
for param in dino_model.parameters():
    param.requires_grad = False

feature_dim = 768

num_classes = len(label_dict)
classifier_head = nn.Linear(feature_dim, num_classes).to(device)

class DinoClassifier(nn.Module):
    def __init__(self, dino_model, classifier):
        super().__init__()
        self.dino_model = dino_model
        self.classifier = classifier

    def forward(self, x):
        with torch.no_grad():
            outputs = self.dino_model(x)
            feats = outputs.last_hidden_state.mean(dim=1)  
            
        logits = self.classifier(feats) 
        return logits

model = DinoClassifier(dino_model, classifier_head).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier_head.parameters(), lr=1e-3)

num_epochs = 10  

for epoch in range(num_epochs):
    model.train()  
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images) 
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")


model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
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


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


cm = confusion_matrix(all_labels, all_preds) 
print("Confusion Matrix:\n", cm)


index_to_name = sorted(label_dict, key=lambda k: label_dict[k])



plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d") 
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(ticks=np.arange(len(index_to_name))+0.5, labels=index_to_name, rotation=45)
plt.yticks(ticks=np.arange(len(index_to_name))+0.5, labels=index_to_name, rotation=45)
plt.tight_layout()
plt.show()


from sklearn.metrics import classification_report

report = classification_report(
    all_labels,
    all_preds,
    target_names=index_to_name,  
    digits=3  
)
print(report)
