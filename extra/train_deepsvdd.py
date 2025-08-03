import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

# ----------------------------
# Device Setup
# ----------------------------
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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

        if self.transform:
            image = self.transform(image)

        return image

# ----------------------------
# Model Definition (Feature Extractor with Offline Pretrained Weights)
# ----------------------------
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        backbone = models.resnet50(weights=None)
        state_dict = torch.load('/home/user/Deepfake/resnet50-0676ba61.pth')
        backbone.load_state_dict(state_dict)
        backbone.fc = nn.Identity()  # Remove final classifier
        self.backbone = backbone

    def forward(self, x):
        features = self.backbone(x)
        return features

# ----------------------------
# Data Preparation
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def create_dataframe(root_dir, datatype='train'):
    real_dir = os.path.join(root_dir, datatype, 'real')
    real_images = [{'image_path': os.path.join(datatype, 'real', f)} for f in os.listdir(real_dir)]
    return pd.DataFrame(real_images)

root_dir = '/home/user/Deepfake/Datasets/real_vs_fake/real-vs-fake'
train_df = create_dataframe(root_dir)
train_dataset = FaceDataset(train_df, root_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

# ----------------------------
# Deep SVDD Training with Resume
# ----------------------------
model = FeatureExtractor().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

checkpoint_path = "deep_svdd_checkpoint.pth"
best_model_path = "deep_svdd_best.pth"
center_path = "deep_svdd_center.pth"

# Initialize hypersphere center
c = torch.zeros(2048).to(device)

start_epoch = 0
best_loss = float('inf')

# Load checkpoint if exists
if os.path.exists(checkpoint_path) and os.path.exists(center_path):
    print("Resuming from checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['best_loss']
    c = torch.load(center_path)
    print(f"Resumed from epoch {start_epoch} with best loss {best_loss:.6f}")
else:
    print("Initializing center...")
    n_samples = 0
    c_sum = torch.zeros(2048).to(device)
    model.eval()
    with torch.no_grad():
        for inputs in tqdm(train_loader, desc="Initializing Center"):
            inputs = inputs.to(device)
            features = model(inputs)
            c_sum += features.sum(dim=0)
            n_samples += features.size(0)
    c = c_sum / n_samples
    torch.save(c, center_path)

# Training Loop
num_epochs = 100
model.train()
for epoch in range(start_epoch, num_epochs):
    running_loss = 0.0
    for inputs in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        inputs = inputs.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        dist = torch.sum((outputs - c) ** 2, dim=1)
        loss = torch.mean(dist)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")

    # Save best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), best_model_path)
        print("Best model updated.")

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
    }, checkpoint_path)
    print("Checkpoint saved.")
