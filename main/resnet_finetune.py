import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
batch_size = 32
criterion = nn.BCELoss()
checkpoint_path = '/home/user/Deepfake/test/combined_finetune_checkpoint.pth'
final_model_path = '/home/user/Deepfake/test/fine_tuned_combined.pth'

# Dataset Class
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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Build Model Function
def build_model():
    model = resnet50(weights=None)
    state_dict = torch.load('/home/user/Deepfake/resnet50-0676ba61.pth')
    model.load_state_dict(state_dict)
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    
    model.to(device)
    return model

# Prepare 140k Dataset
def create_dataframe(root_dir, datatype='train'):
    real_dir = os.path.join(root_dir, datatype, 'real')
    fake_dir = os.path.join(root_dir, datatype, 'fake')
    
    real_images = [{'image_path': os.path.join(datatype, 'real', f), 'label': 1} for f in os.listdir(real_dir)]
    fake_images = [{'image_path': os.path.join(datatype, 'fake', f), 'label': 0} for f in os.listdir(fake_dir)]
    
    return pd.DataFrame(real_images + fake_images)

df_140k = create_dataframe('/home/user/Deepfake/Datasets/real_vs_fake/real-vs-fake')

# Sample 10k from 140k Dataset
df_140k_sample = df_140k.sample(n=10000, random_state=42).reset_index(drop=True)

# Prepare Handcrafted Dataset
real_dir = '/home/user/Deepfake/Datasets/handcrafted_real_fake/real'
fake_dir = '/home/user/Deepfake/Datasets/handcrafted_real_fake/fake'

real_images = [{'image_path': os.path.join('real', f), 'label': 1} for f in os.listdir(real_dir)]
fake_images = [{'image_path': os.path.join('fake', f), 'label': 0} for f in os.listdir(fake_dir)]
df_handcrafted = pd.DataFrame(real_images + fake_images)

# Combine Datasets
combined_df = pd.concat([df_140k_sample, df_handcrafted], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Dataset
root_dir_combined = '/home/user/Deepfake/Datasets/handcrafted_real_fake/'  # For handcrafted images
root_dir_140k = '/home/user/Deepfake/Datasets/real_vs_fake/real-vs-fake/'  # For 140k images

# Add full path prefix for each row
def add_full_path(row):
    if 'train/real' in row['image_path'] or 'train/fake' in row['image_path']:
        row['root'] = root_dir_140k
    else:
        row['root'] = root_dir_combined
    return row

combined_df = combined_df.apply(add_full_path, axis=1)

# Custom Dataset for mixed root dirs
class MixedDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(row['root'], row['image_path'])
        image = Image.open(img_path).convert('RGB')
        label = row['label']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

full_dataset = MixedDataset(combined_df, transform=transform)

# Split Dataset
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load Model
model = build_model()
model.load_state_dict(torch.load('/home/user/Deepfake/test/best_model.pth'))

# Freeze all layers
for name, param in model.named_parameters():
    param.requires_grad = False

# Unfreeze layer4 and fc
for name, param in model.named_parameters():
    if name.startswith('layer4') or name.startswith('fc'):
        param.requires_grad = True

# Optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# Resume Logic
start_epoch = 0
if os.path.exists(checkpoint_path):
    print(f"Resuming from checkpoint {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed from epoch {start_epoch}")

# Training Loop
num_epochs = 50
for epoch in range(start_epoch, num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    print('-' * 10)
    
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
    
    print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
    # Validation
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
    
    print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
    
    # Save Checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

# Save Final Model
torch.save(model.state_dict(), final_model_path)
print(f"Fine-tuned model saved at {final_model_path}")
