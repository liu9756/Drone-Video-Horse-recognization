import torch
import torch.nn as nn
from transformers import AutoModel

def build_model(device, num_classes):
    from transformers import AutoModel
    dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    dino_model.eval()
    for param in dino_model.parameters():
        param.requires_grad = False

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
            logits = self.classifier(feats)
            return logits


        def extract_features(self, x):
            with torch.no_grad():
                outputs = self.dino_model(x)
                feats = outputs.last_hidden_state.mean(dim=1) 
                feat512 = self.classifier[0](feats)
                feat512 = self.classifier[1](feat512)
            return feat512

    model = DinoClassifier(dino_model, classifier_head).to(device)
    return model


def transform_feature(features_768, model, device):
    with torch.no_grad():
        x = torch.from_numpy(features_768).float().to(device)
        feat512 = model.classifier[0](x)
        feat512 = model.classifier[1](feat512)
    return feat512.cpu().numpy()


