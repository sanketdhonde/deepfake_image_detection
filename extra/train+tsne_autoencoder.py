import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # Prevent GUI issues if running remotely
import matplotlib.pyplot as plt

# Device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset
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

        if self.transform:
            image = self.transform(image)

        label = self.dataframe.iloc[idx]['label']

        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def create_dataframe(root_dir, datatype='test'):
    real_dir = os.path.join(root_dir, datatype, 'real')
    fake_dir = os.path.join(root_dir, datatype, 'fake')

    real_images = [{'image_path': os.path.join(datatype, 'real', f), 'label': 1} for f in os.listdir(real_dir)]
    fake_images = [{'image_path': os.path.join(datatype, 'fake', f), 'label': 0} for f in os.listdir(fake_dir)]
    return pd.DataFrame(real_images + fake_images)

# Autoencoder definition
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
        return out, z

# Load best model
print("Loading model...")
model = Autoencoder().to(device)
model.load_state_dict(torch.load("autoencoder_best.pth"))
model.eval()
print("Model loaded successfully.")

# Prepare data
root_dir = '/home/user/Deepfake/Datasets/real_vs_fake/real-vs-fake'
print("Creating dataframe...")
test_df = create_dataframe(root_dir)
print(f"Dataframe created with {len(test_df)} samples.")

print("Preparing dataset...")
test_dataset = FaceDataset(test_df, root_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print("Dataset ready.")

# Extract features and reconstruction errors
print("Extracting features and reconstruction errors...")
features = []
labels = []
recon_errors = []
criterion = nn.MSELoss(reduction='none')

with torch.no_grad():
    for inputs, lbls in test_loader:
        inputs = inputs.to(device)
        outputs, z = model(inputs)
        features.append(z.view(z.size(0), -1).cpu().numpy())
        labels.extend(lbls.numpy())
        loss = criterion(outputs, inputs)
        per_sample_loss = loss.view(loss.size(0), -1).mean(dim=1).cpu().numpy()
        recon_errors.extend(per_sample_loss)

features = np.concatenate(features, axis=0)
labels = np.array(labels)
recon_errors = np.array(recon_errors)
print(f"Features shape: {features.shape}")
print("Feature extraction complete.")

# Optional PCA before t-SNE if feature dimension is large
if features.shape[1] > 50:
    print("Applying PCA to reduce feature dimensions before t-SNE...")
    pca = PCA(n_components=50)
    features = pca.fit_transform(features)
    print(f"Feature shape after PCA: {features.shape}")

# t-SNE visualization
print("Running t-SNE, this might take a minute...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
features_2d = tsne.fit_transform(features)
print("t-SNE complete. Generating plot...")

plt.figure(figsize=(8,6))
plt.scatter(features_2d[labels==1, 0], features_2d[labels==1, 1], c='blue', label='Real', alpha=0.6)
plt.scatter(features_2d[labels==0, 0], features_2d[labels==0, 1], c='red', label='Fake', alpha=0.6)
plt.legend()
plt.title('t-SNE Visualization of Latent Space (Real vs Fake)')
plt.savefig("tsne_visualization.png")
plt.close()
print("t-SNE plot saved as 'tsne_visualization.png'.")

# Reconstruction Error Histogram
print("Generating reconstruction error histogram...")
plt.figure(figsize=(8,6))
plt.hist(recon_errors[labels==1], bins=50, alpha=0.6, label='Real', color='blue')
plt.hist(recon_errors[labels==0], bins=50, alpha=0.6, label='Fake', color='red')
plt.title('Reconstruction Error Distribution')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()
plt.savefig("reconstruction_error_histogram.png")
plt.close()
print("Reconstruction error histogram saved as 'reconstruction_error_histogram.png'.")

print("All tasks completed successfully.")
