import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from AmtenNet_train import AMTENNet

# Import AMTENNet from your training file


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Load Pretrained AMTENNet ---------------- #

model = AMTENNet()
model.load_state_dict(torch.load("/home/user/Deepfake/test/amtennet_best.pt"))
model.to(device)
count=0
# Freeze all layers
for name,param in model.named_parameters():
    param.requires_grad = False
    count+=1
    print(name,param.shape)
print(count)

# Unfreeze classifier block
for param in model.fc1.parameters():
    param.requires_grad = True
for param in model.fc2.parameters():
    param.requires_grad = True
for param in model.fc3.parameters():
    param.requires_grad = True

# ----------------- Data Preparation with Augmentation ---------------- #

#data_transforms = transforms.Compose([
#    transforms.Resize((128, 128)),
#    transforms.RandomHorizontalFlip(p=0.5),
#    transforms.RandomRotation(degrees=15),
#    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#    transforms.ToTensor(),
#])
data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5,0.5])  # Normalize to [-1, 1]
])

data_dir = "/home/user/Deepfake/Datasets/handcrafted_real_fake/"

train_dataset = datasets.ImageFolder(
    root=data_dir,
    transform=data_transforms
)

# Optional: Inspect class mapping to confirm labels
print("Class to index mapping:", train_dataset.class_to_idx)
# Example output: {'training_fake': 0, 'training_real': 1}

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

# ----------------- Fine-Tuning Setup ---------------- #

trainable_param = filter(lambda p: p.requires_grad, model.parameters())

count=0
# Freeze all layers
for name,param in model.named_parameters():

    count+=1
    print(name,param.shape,param.requires_grad)
print(count)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-7, momentum=0.95, weight_decay=0.005)


# ----------------- Fine-Tuning Loop ---------------- #

num_epochs = 20

model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=running_loss/(total//16), acc=100*correct/total)
    
    
print("Fine-tuning complete.")

torch.save(model.state_dict(), "amtennet_finetuned_handcrafted.pth")

print("Fine-tuned model saved to amtennet_finetuned_handcrafted.pth")