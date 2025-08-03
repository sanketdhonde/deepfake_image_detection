import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from train import build_model

# Configuration
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

test_dir = "/home/user/Deepfake/Datasets/real_vs_fake/real-vs-fake/test/"
batch_size = 16

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load Model
model, _ = build_model()
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Collect predictions and true labels
all_preds = []
all_labels = []

# Track correctly classified fake images
correct_fake_images = []
global_idx = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        for i in range(labels.size(0)):
            true_label = labels[i].item()
            pred_label = preds[i].item()
            img_path, _ = test_dataset.samples[global_idx]

            if true_label == 1 and pred_label == 1:
                correct_fake_images.append(img_path)

            global_idx += 1

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

# Optional: Visualize the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for AMTENNet")
plt.show()

# Correctly classified fake images
print(f"Total correctly classified fake images: {len(correct_fake_images)}")
if correct_fake_images:
    print("Example correctly classified fake image:")
    print(correct_fake_images[0])
