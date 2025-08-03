import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# ----------------------------
# Device Setup
# ----------------------------
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Dataset Definition
# ----------------------------
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

        return image, label, self.dataframe.iloc[idx]['image_path']

# ----------------------------
# Data Preparation
# ----------------------------
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

root_dir = '/home/user/Deepfake/Datasets/real_vs_fake/real-vs-fake'
test_df = create_dataframe(root_dir)
test_dataset = FaceDataset(test_df, root_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# ----------------------------
# Feature Extractor Definition
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
# Feature Extraction
# ----------------------------
model = FeatureExtractor().to(device)
model.eval()

features_list = []
labels_list = []
paths_list = []

with torch.no_grad():
    for inputs, labels, paths in tqdm(test_loader, desc="Extracting features"):
        inputs = inputs.to(device)
        features = model(inputs)
        features_list.append(features.cpu().numpy())
        labels_list.extend(labels.numpy())
        paths_list.extend(paths)

features_array = np.concatenate(features_list, axis=0)
labels_array = np.array(labels_list)
paths_array = np.array(paths_list)

# Save to npz
np.savez("/home/user/Deepfake/OCC/test_features.npz", features=features_array, labels=labels_array, paths=paths_array)
print("Saved extracted features to test_features.npz")
