import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt

# ----------------------------
# Setup
# ----------------------------
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Image preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----------------------------
# Model Definition
# ----------------------------
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        backbone = models.resnet50(weights=None)
        state_dict = torch.load('/home/user/Deepfake/resnet50-0676ba61.pth')
        backbone.load_state_dict(state_dict)
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

# ----------------------------
# Load model and center
# ----------------------------
model = FeatureExtractor().to(device)
model.load_state_dict(torch.load("/home/user/Deepfake/extra/deep_svdd_best.pth"))
model.eval()

center = torch.load("deep_svdd_center.pth").to(device)

# ----------------------------
# Prediction Function
# ----------------------------
def predict_image(image_path, threshold):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(image_tensor)
        distance = torch.sum((features - center) ** 2, dim=1).item()

    print(f"\nImage: {image_path}")
    print(f"Distance from center: {distance:.4f}")

    plt.imshow(image)
    plt.axis('off')
    if distance <= threshold:
        plt.title(f"Prediction: REAL ✅\nDistance: {distance:.2f}", color='blue')
    else:
        plt.title(f"Prediction: FAKE ❌\nDistance: {distance:.2f}", color='red')
    plt.show()

# ----------------------------
# Batch Distance Analysis
# ----------------------------
def get_distances(label_dir):
    distances = []
    for fname in os.listdir(label_dir):
        path = os.path.join(label_dir, fname)
        try:
            image = Image.open(path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                features = model(image_tensor)
                dist = torch.sum((features - center) ** 2, dim=1).item()
                distances.append(dist)
        except:
            continue
    return distances

# ----------------------------
# Plot Distance Distributions
# ----------------------------
def plot_distances(real_dists, fake_dists, threshold):
    plt.figure(figsize=(10,6))
    plt.hist(real_dists, bins=50, alpha=0.6, label='Real', color='blue')
    plt.hist(fake_dists, bins=50, alpha=0.6, label='Fake', color='red')
    plt.axvline(threshold, color='green', linestyle='--', label=f'Threshold = {threshold}')
    plt.xlabel('Distance from center')
    plt.ylabel('Frequency')
    plt.title('Distance Distributions: Real vs Fake')
    plt.legend()
    plt.grid(True)
    plt.show()

# ----------------------------
# Example Usage
# ----------------------------
# Adjust this threshold based on your plotted histogram
chosen_threshold = 85.0
predict_image("/home/user/Deepfake/Datasets/real_vs_fake/real-vs-fake/test/real/00028.jpg", threshold=chosen_threshold)

# Optional: Plot histogram
real_dir = "/home/user/Deepfake/Datasets/real_vs_fake/real-vs-fake/test/real"
fake_dir = "/home/user/Deepfake/Datasets/real_vs_fake/real-vs-fake/test/fake"
real_dists = get_distances(real_dir)
fake_dists = get_distances(fake_dir)
plot_distances(real_dists, fake_dists, threshold=chosen_threshold)
