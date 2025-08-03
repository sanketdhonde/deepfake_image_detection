import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
from tqdm import tqdm

# ---------------- AMTENNet with CBAM ----------------
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x).view(x.size(0), -1)
        max_out = self.max_pool(x).view(x.size(0), -1)
        scale = self.sigmoid_channel(self.fc(avg_out) + self.fc(max_out)).view(x.size(0), -1, 1, 1)
        x = x * scale

        avg = torch.mean(x, dim=1, keepdim=True)
        max_ = torch.max(x, dim=1, keepdim=True)[0]
        spatial_att = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg, max_], dim=1)))
        x = x * spatial_att
        return x

class AMTENNet(nn.Module):
    def __init__(self):
        super(AMTENNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

        self.trace_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.cbam1 = CBAM(64)
        self.trace_relu1 = nn.ReLU(inplace=False)
        self.trace_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.trace_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.cbam2 = CBAM(128)
        self.trace_relu2 = nn.ReLU(inplace=False)
        self.trace_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.trace_conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.cbam3 = CBAM(256)
        self.trace_relu3 = nn.ReLU(inplace=False)
        self.trace_pool3 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)

        x = self.trace_conv1(x)
        x = self.cbam1(x)
        x = self.trace_relu1(x)
        x = self.trace_pool1(x)

        x = self.trace_conv2(x)
        x = self.cbam2(x)
        x = self.trace_relu2(x)
        x = self.trace_pool2(x)

        x = self.trace_conv3(x)
        x = self.cbam3(x)
        x = self.trace_relu3(x)
        x = self.trace_pool3(x)

        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x

# ---------------- Dataset ----------------
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

# ---------------- Train Setup ----------------
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def create_dataframe(root_dir, datatype='train'):
    real_dir = os.path.join(root_dir, datatype, 'real')
    fake_dir = os.path.join(root_dir, datatype, 'fake')
    real_images = [{'image_path': os.path.join(datatype, 'real', f), 'label': 1} for f in os.listdir(real_dir)]
    fake_images = [{'image_path': os.path.join(datatype, 'fake', f), 'label': 0} for f in os.listdir(fake_dir)]
    return pd.DataFrame(real_images + fake_images)

root_dir = '/home/user/Deepfake/Datasets/real_vs_fake/real-vs-fake'
train_df = create_dataframe(root_dir, 'train')
val_df = create_dataframe(root_dir, 'valid')

train_loader = DataLoader(FaceDataset(train_df, root_dir, transform), batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(FaceDataset(val_df, root_dir, transform), batch_size=32, shuffle=False, num_workers=4)

# ---------------- Training ----------------
def build_model(resume=False, checkpoint_path='checkpoint_amtennetcbam.pth'):
    model = AMTENNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    start_epoch = 0
    best_acc = 0.0

    if resume and os.path.exists(checkpoint_path):
        print("Resuming training from checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']

    return model, optimizer, start_epoch, best_acc

model, optimizer, start_epoch, best_acc = build_model(resume=True)
criterion = nn.BCELoss()


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15, start_epoch=0, best_acc=0.0):
    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}\n' + '-' * 15)

        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        model.eval()
        val_loss, val_corrects = 0.0, 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validating'):
                inputs = inputs.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                preds = (outputs > 0.5).float()
                val_corrects += torch.sum(preds == labels.data)

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n')

        # Save checkpoint
        checkpoint_path = 'checkpoint_amtennetcbam.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc
        }, checkpoint_path)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model_amtennetcbam.pth')
            print("Best model updated.")

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15, start_epoch=start_epoch, best_acc=best_acc)
