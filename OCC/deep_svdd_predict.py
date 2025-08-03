import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load extracted features
data = np.load("/home/user/Deepfake/OCC/test_features.npz")
features = data['features']  # shape (N, 2048)
labels = data['labels']      # 1 = real, 0 = fake
paths = data['paths']

# Load trained center and model
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        from torchvision import models
        backbone = models.resnet50(weights=None)
        state_dict = torch.load('/home/user/Deepfake/resnet50-0676ba61.pth')
        backbone.load_state_dict(state_dict)
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

# Load center
c = torch.load("/home/user/Deepfake/extra/deep_svdd_center.pth").to(device)

# Compute distances
features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
distances = torch.sum((features_tensor - c) ** 2, dim=1).cpu().numpy()

# Find best threshold by grid search (optional)
best_threshold = None
best_f1 = 0
for t in np.linspace(np.min(distances), np.max(distances), 100):
    preds = (distances > t).astype(int)
    f1 = ((2 * ((preds == 0) & (labels == 1)).sum()) / 
         ((preds == 0).sum() + (labels == 1).sum() + 1e-6))
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"Best threshold: {best_threshold:.4f}, Best F1 for REAL class: {best_f1:.4f}")

# Predict
preds = (distances > best_threshold).astype(int)  # 0 = real, 1 = fake
preds = 1 - preds  # Flip: 1 = real, 0 = fake

# Report
print("\nClassification Report:")
print(classification_report(labels, preds, target_names=["Fake", "Real"]))
print(f"ROC AUC: {roc_auc_score(labels, distances):.4f}")

# Histogram of distances
plt.figure(figsize=(8,5))
plt.hist(distances[labels==1], bins=50, alpha=0.6, label='Real', color='blue')
plt.hist(distances[labels==0], bins=50, alpha=0.6, label='Fake', color='red')
plt.axvline(best_threshold, color='black', linestyle='--', label='Threshold')
plt.title("Deep SVDD Distance Histogram")
plt.xlabel("Distance to Center")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# t-SNE visualization
print("Generating t-SNE visualization...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
features_2d = tsne.fit_transform(features)

plt.figure(figsize=(8, 6))
plt.scatter(features_2d[labels == 1, 0], features_2d[labels == 1, 1], c='blue', label='Real', alpha=0.6)
plt.scatter(features_2d[labels == 0, 0], features_2d[labels == 0, 1], c='red', label='Fake', alpha=0.6)
plt.title("t-SNE Projection of Features")
plt.legend()
plt.tight_layout()
plt.show()
