import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import os

# ----------------------------
# Device Setup
# ----------------------------
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ----------------------------
# Autoencoder Definition
# ----------------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# ----------------------------
# Load Model
# ----------------------------
model = Autoencoder().to(device)
model.load_state_dict(torch.load("/home/user/Deepfake/extra/autoencoder_best.pth"))
model.eval()

# ----------------------------
# Image Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----------------------------
# Load Test Data
# ----------------------------
from torch.utils.data import Dataset, DataLoader

class FaceDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.dataframe.iloc[idx]['image_path'])
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label

def create_dataframe(root_dir, datatype='test'):
    real_dir = os.path.join(root_dir, datatype, 'real')
    fake_dir = os.path.join(root_dir, datatype, 'fake')

    real_images = [{'image_path': os.path.join(datatype, 'real', f), 'label': 1} for f in os.listdir(real_dir)]
    fake_images = [{'image_path': os.path.join(datatype, 'fake', f), 'label': 0} for f in os.listdir(fake_dir)]
    return pd.DataFrame(real_images + fake_images)

import pandas as pd

root_dir = '/home/user/Deepfake/Datasets/real_vs_fake/real-vs-fake'
test_df = create_dataframe(root_dir)
test_dataset = FaceDataset(test_df, root_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ----------------------------
# Evaluate Model on Test Set
# ----------------------------
recon_errors = []
labels = []
criterion = nn.MSELoss(reduction='none')

with torch.no_grad():
    for images, lbls in test_loader:
        images = images.to(device)
        outputs = model(images)
        loss = criterion(outputs, images)
        loss = loss.view(loss.size(0), -1).mean(dim=1).cpu().numpy()
        recon_errors.extend(loss)
        labels.extend(lbls.numpy())

recon_errors = np.array(recon_errors)
labels = np.array(labels)

# ----------------------------
# Threshold Selection
# ----------------------------
best_threshold = None
best_f1 = 0
for t in np.linspace(np.min(recon_errors), np.max(recon_errors), 100):
    preds = (recon_errors > t).astype(int)
    f1 = ((2 * ((preds == 0) & (labels == 1)).sum()) /
         ((preds == 0).sum() + (labels == 1).sum() + 1e-6))
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"Best threshold: {best_threshold:.4f}, Best F1 for REAL class: {best_f1:.4f}")

# Predict
preds = (recon_errors > best_threshold).astype(int)
preds = 1 - preds  # Flip to: 1 = real, 0 = fake

# Report
print("\nClassification Report:")
print(classification_report(labels, preds, target_names=["Fake", "Real"]))
print(f"ROC AUC: {roc_auc_score(labels, recon_errors):.4f}")

# Visualization
plt.figure(figsize=(8,5))
plt.hist(recon_errors[labels==1], bins=50, alpha=0.6, label='Real', color='blue')
plt.hist(recon_errors[labels==0], bins=50, alpha=0.6, label='Fake', color='red')
plt.axvline(best_threshold, color='black', linestyle='--', label='Threshold')
plt.title("Autoencoder Reconstruction Error Histogram")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()
