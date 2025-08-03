import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import scipy.fftpack
from mcnet_finetune import MCNet, block_dct, binarized_dct   # Or define MCNet here directly

# ----------------------------
# Feature extraction utils
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
# Dataset for test evaluation
# ----------------------------
class FaceDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        path = self.dataframe.iloc[idx]['image_path']
        label = self.dataframe.iloc[idx]['label']
        image = Image.open(path).convert('RGB').resize((128, 128))
        image_np = np.array(image).astype(np.float32) / 255.0
        spatial = self.transform(image) if self.transform else transforms.ToTensor()(image)
        freq = torch.from_numpy(block_dct(image_np)).float()
        gray = np.mean(image_np, axis=2)
        comp = torch.from_numpy(binarized_dct(gray)).float()
        return spatial, freq, comp, torch.tensor(label, dtype=torch.float32)

# ----------------------------
# Inference & evaluation
# ----------------------------
def evaluate(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for spatial, freq, comp, labels in dataloader:
            spatial, freq, comp = spatial.to(device), freq.to(device), comp.to(device)
            outputs = model(spatial, freq, comp).cpu()
            predictions = (outputs > 0.5).float().squeeze()
            y_pred.extend(predictions.tolist())
            y_true.extend(labels.tolist())
    return np.array(y_true), np.array(y_pred)

# ----------------------------
# Confusion matrix plot
# ----------------------------
def plot_confusion_matrix(y_true, y_pred, labels=["Real", "Fake"]):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    model_path = "/home/user/Deepfake/test/mcnet_finetuned_best.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load test data (update paths accordingly)
    test_real_dir = "/home/user/Deepfake/Datasets/real_vs_fake/real-vs-fake/test/real"
    test_fake_dir = "/home/user/Deepfake/Datasets/real_vs_fake/real-vs-fake/test/fake"

    test_data = (
        [{'image_path': os.path.join(test_real_dir, f), 'label': 1} for f in os.listdir(test_real_dir)] +
        [{'image_path': os.path.join(test_fake_dir, f), 'label': 0} for f in os.listdir(test_fake_dir)]
    )
    test_df = pd.DataFrame(test_data)
    print(f"🧪 Test samples: {len(test_df)}")

    spatial_transform = transforms.Compose([
        transforms.CenterCrop(128),
        transforms.ToTensor()
    ])

    test_dataset = FaceDataset(test_df, spatial_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    model = MCNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    y_true, y_pred = evaluate(model, test_loader, device)

    # Print metrics
    print("📊 Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=["Fake", "Real"]))

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred)

