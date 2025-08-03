import os
import random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import scipy.fftpack
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ----------------------------
# Model Blocks
# ----------------------------
class BT1(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(BT1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))

class BT2(nn.Module):
    def __init__(self, channels):
        super(BT2, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + x)

class BT3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BT3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = self.shortcut(x)
        return torch.relu(out + shortcut)

class BT4(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(BT4, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)
    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class VANet(nn.Module):
    def __init__(self):
        super(VANet, self).__init__()
        self.spatial = nn.Sequential(
            BT1(3, 64), BT1(64, 64),
            BT2(64), BT2(64), BT2(64), BT2(64),
            BT3(64, 128), BT3(128, 256), BT3(256, 256), BT3(256, 256)
        )
    def forward(self, x):
        return self.spatial(x)

class FrequencyLearner(nn.Module):
    def __init__(self):
        super(FrequencyLearner, self).__init__()
        self.freq = nn.Sequential(
            BT1(48, 64, groups=4), BT1(64, 64), BT1(64, 64),
            BT2(64), BT2(64), BT2(64), BT2(64), BT2(64), BT2(64),
            BT3(64, 128), BT3(128, 128), BT3(128, 128)
        )
    def forward(self, x):
        return self.freq(x)

class CANet(nn.Module):
    def __init__(self):
        super(CANet, self).__init__()
        self.dilated_conv = nn.Conv2d(11, 32, kernel_size=3, padding=2, dilation=2)
        self.regular_conv = nn.Conv2d(11, 32, kernel_size=3, padding=1)
        self.bt1 = BT1(64, 64)
        self.bt2_blocks = nn.Sequential(BT2(64), BT2(64), BT2(64), BT2(64))
        self.bt3_blocks = nn.Sequential(BT3(64, 128), BT3(128, 256), BT3(256, 256), BT3(256, 256))
    def forward(self, x):
        x1 = self.dilated_conv(x)
        x2 = self.regular_conv(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.bt1(x)
        x = self.bt2_blocks(x)
        x = self.bt3_blocks(x)
        return x

class MCNet(nn.Module):
    def __init__(self):
        super(MCNet, self).__init__()
        self.vanet = VANet()
        self.freq_learner = FrequencyLearner()
        self.canet = CANet()
        self.bt4 = BT4(256 + 128 + 256, 1)
    def forward(self, x_spatial, x_freq, x_compression):
        f_spatial = self.vanet(x_spatial)
        f_freq = self.freq_learner(x_freq)
        f_compress = self.canet(x_compression)
        concat = torch.cat([f_spatial, f_freq, f_compress], dim=1)
        out = self.bt4(concat)
        return torch.sigmoid(out)

# ----------------------------
# DCT Functions
# ----------------------------
def block_dct(image_np, block_size=4, stride=2):
    h, w, c = image_np.shape
    dct_features = []
    for ch in range(c):
        channel = image_np[:, :, ch]
        features = []
        for y in range(0, h - block_size + 1, stride):
            for x in range(0, w - block_size + 1, stride):
                patch = channel[y:y+block_size, x:x+block_size]
                dct_patch = scipy.fftpack.dct(scipy.fftpack.dct(patch.T, norm='ortho').T, norm='ortho')
                features.append(dct_patch.flatten())
        features = np.array(features).T
        dct_features.append(features)
    dct_features = np.concatenate(dct_features, axis=0)
    side = int(np.sqrt(dct_features.shape[1]))
    return dct_features.reshape(48, side, side)

def binarized_dct(gray_image, num_bins=11, threshold=10):
    dct_coeff = scipy.fftpack.dct(scipy.fftpack.dct(gray_image.T, norm='ortho').T, norm='ortho')
    dct_coeff = np.abs(dct_coeff)
    dct_coeff = np.clip(dct_coeff, 0, threshold)
    binarized = np.zeros((num_bins, gray_image.shape[0], gray_image.shape[1]))
    for i in range(num_bins):
        binarized[i] = (dct_coeff == i).astype(np.float32)
    return binarized

# ----------------------------
# Dataset Class
# ----------------------------
class FaceDataset(Dataset):
    def __init__(self, dataframe, spatial_transform=None):
        self.dataframe = dataframe
        self.spatial_transform = spatial_transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB').resize((128, 128))
        label = self.dataframe.iloc[idx]['label']

        image_np = np.array(image).astype(np.float32) / 255.0
        spatial_image = self.spatial_transform(image) if self.spatial_transform else transforms.ToTensor()(image)
        freq_feature = torch.from_numpy(block_dct(image_np)).float()
        gray_image = np.mean(image_np, axis=2)
        comp_feature = torch.from_numpy(binarized_dct(gray_image)).float()

        return spatial_image, freq_feature, comp_feature, torch.tensor(label, dtype=torch.float32)

# ----------------------------
# Training Function
# ----------------------------
def train_model(model, dataloader, device, num_epochs, lr, checkpoint_path, final_model_path):
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    best_acc = 0

    for epoch in range(num_epochs):
        model.train()
        total, correct, running_loss = 0, 0, 0.0

        for spatial, freq, comp, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            spatial, freq, comp, labels = spatial.to(device), freq.to(device), comp.to(device), labels.unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(spatial, freq, comp)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * spatial.size(0)
            correct += torch.sum((outputs > 0.5).float() == labels).item()
            total += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}: Loss={running_loss / total:.4f}, Accuracy={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), final_model_path)
            print(f"✅ Best model saved with acc={acc:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

# ----------------------------
# t-SNE Function
# ----------------------------
def run_tsne(model, dataloader, device):
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for spatial, freq, comp, lbl in tqdm(dataloader, desc="TSNE Extract"):
            spatial, freq, comp = spatial.to(device), freq.to(device), comp.to(device)
            out = model.vanet(spatial)
            features.append(out.view(out.size(0), -1).cpu())
            labels.append(lbl)

    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()
    tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(features)
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap='coolwarm', s=5)
    plt.colorbar()
    plt.title("t-SNE Visualization")
    plt.show()

# ----------------------------
# Main Driver Code
# ----------------------------
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    handcrafted_dir = "/home/user/Deepfake/Datasets/handcrafted_real_fake"
    real_dir = "/home/user/Deepfake/Datasets/real_vs_fake/real-vs-fake/train/real"
    fake_dir = "/home/user/Deepfake/Datasets/real_vs_fake/real-vs-fake/train/fake"
    checkpoint_path = "/home/user/Deepfake/test/mcnet_finetune_checkpoint.pth"
    final_model_path = "/home/user/Deepfake/test/mcnet_finetuned_best.pth"

    real_images = random.sample([os.path.join(real_dir, f) for f in os.listdir(real_dir)], 5000)
    fake_images = random.sample([os.path.join(fake_dir, f) for f in os.listdir(fake_dir)], 5000)

    handcrafted_real = [os.path.join(handcrafted_dir, 'real', f) for f in os.listdir(os.path.join(handcrafted_dir, 'real'))]
    handcrafted_fake = [os.path.join(handcrafted_dir, 'fake', f) for f in os.listdir(os.path.join(handcrafted_dir, 'fake'))]

    data = (
        [{'image_path': path, 'label': 1} for path in real_images + handcrafted_real] +
        [{'image_path': path, 'label': 0} for path in fake_images + handcrafted_fake]
    )
    df = pd.DataFrame(data)
    print(f"Total samples: {len(df)}")

    spatial_transform = transforms.Compose([
        transforms.RandomCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor()
    ])

    dataset = FaceDataset(df, spatial_transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    model = MCNet().to(device)
    for name, param in model.named_parameters():
        if 'BT1' in name or 'bt1' in name:
            param.requires_grad = False

    print("Trainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"✅ {name}")

    train_model(model, dataloader, device, num_epochs=20, lr=1e-4,
                checkpoint_path=checkpoint_path, final_model_path=final_model_path)

    run_tsne(model, dataloader, device)
