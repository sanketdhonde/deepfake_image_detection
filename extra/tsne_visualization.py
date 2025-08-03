import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from tqdm import tqdm

# Configuration
data_dir = "/home/user/Deepfake/Datasets/real_vs_fake/real-vs-fake/train/"
image_size = 64  # Resize images to 64x64
batch_size = 64

# 1. Load images with basic transforms
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),  # [C, H, W]
])

dataset = datasets.ImageFolder(data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 2. Extract raw image features (flattened pixels)
features = []
labels = []

print("Extracting raw pixel features...")
for images, lbls in tqdm(loader):
    # Flatten each image to a 1D vector: [batch, C*H*W]
    batch_features = images.view(images.size(0), -1).numpy()
    features.append(batch_features)
    labels.extend(lbls.numpy())

features = np.concatenate(features, axis=0)
labels = np.array(labels)

print(f"Shape of raw image features: {features.shape}")  # e.g., (N, 12288)

# 3. Apply t-SNE
print("Applying t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_result = tsne.fit_transform(features)

# 4. Plotting
plt.figure(figsize=(10, 7))
colors = ['red' if label == 0 else 'blue' for label in labels]
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=colors, alpha=0.6)
plt.title("t-SNE on Raw Images (Real vs Fake)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', label='Real', markerfacecolor='red', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Fake', markerfacecolor='blue', markersize=10)
])
plt.grid(True)
plt.tight_layout()
plt.show()
