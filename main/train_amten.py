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
from amtennet import AMTENNet

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

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
        label = self.dataframe.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# DataFrame
def create_dataframe(root_dir, datatype='train'):
    real_dir = os.path.join(root_dir, datatype, 'real')
    fake_dir = os.path.join(root_dir, datatype, 'fake')

    real_images = [{'image_path': os.path.join(datatype, 'real', f), 'label': 1} for f in os.listdir(real_dir)]
    fake_images = [{'image_path': os.path.join(datatype, 'fake', f), 'label': 0} for f in os.listdir(fake_dir)]

    return pd.DataFrame(real_images + fake_images)

# Paths
root_dir = '/home/user/Deepfake/Datasets/real_vs_fake/real-vs-fake'
train_df = create_dataframe(root_dir)
val_df = create_dataframe(root_dir, datatype='valid')

# Dataloaders
batch_size = 32
train_dataset = FaceDataset(train_df, root_dir, transform=transform)
val_dataset = FaceDataset(val_df, root_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Training utilities
start_epoch = 0
best_acc = 0.0
checkpoint_path = 'checkpoint_amtennet.pth'
criterion = nn.BCELoss()

# Build Model
def build_model(resume=False):
    global checkpoint_path, start_epoch, best_acc
    model = AMTENNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if resume and os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"Resumed from epoch {start_epoch} with best acc {best_acc:.4f}")

    return model, optimizer

model, optimizer = build_model(resume=True)

# Training
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15, start_epoch=0, best_acc=0.0):
    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 15)

        model.train()
        running_loss = 0.0
        running_corrects = 0

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
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validating'):
                inputs = inputs.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                preds = (outputs > 0.5).float()
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n')

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc
        }, checkpoint_path)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model_amtennet.pth')
            print("Best model updated.")

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15, start_epoch=start_epoch, best_acc=best_acc)
