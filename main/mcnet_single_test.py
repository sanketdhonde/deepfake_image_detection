#import torch
#from torchvision import transforms
#from PIL import Image
#import numpy as np
#import scipy.fftpack
#from mcnet_finetune import MCNet, block_dct, binarized_dct  # import from your existing training script if modularized
#
## ----------------------------
## In case not modularized, redefine the required utils here
## ----------------------------
#def block_dct(image_np, block_size=4, stride=2):
#    h, w, c = image_np.shape
#    dct_features = []
#    for ch in range(c):
#        channel = image_np[:, :, ch]
#        features = []
#        for y in range(0, h - block_size + 1, stride):
#            for x in range(0, w - block_size + 1, stride):
#                patch = channel[y:y+block_size, x:x+block_size]
#                dct_patch = scipy.fftpack.dct(scipy.fftpack.dct(patch.T, norm='ortho').T, norm='ortho')
#                features.append(dct_patch.flatten())
#        features = np.array(features).T
#        dct_features.append(features)
#    dct_features = np.concatenate(dct_features, axis=0)
#    side = int(np.sqrt(dct_features.shape[1]))
#    return dct_features.reshape(48, side, side)
#
#def binarized_dct(gray_image, num_bins=11, threshold=10):
#    dct_coeff = scipy.fftpack.dct(scipy.fftpack.dct(gray_image.T, norm='ortho').T, norm='ortho')
#    dct_coeff = np.abs(dct_coeff)
#    dct_coeff = np.clip(dct_coeff, 0, threshold)
#    binarized = np.zeros((num_bins, gray_image.shape[0], gray_image.shape[1]))
#    for i in range(num_bins):
#        binarized[i] = (dct_coeff == i).astype(np.float32)
#    return binarized
#
## ----------------------------
## Inference on a single image
## ----------------------------
#def test_single_image(model_path, image_path, device):
#    # --- Load and preprocess image ---
#    image = Image.open(image_path).convert('RGB').resize((128, 128))
#    image_np = np.array(image).astype(np.float32) / 255.0
#    gray_image = np.mean(image_np, axis=2)
#
#    # Spatial transform
#    spatial_transform = transforms.Compose([
#        transforms.CenterCrop(128),
#        transforms.ToTensor()
#    ])
#    spatial_tensor = spatial_transform(image).unsqueeze(0).to(device)
#    freq_tensor = torch.from_numpy(block_dct(image_np)).unsqueeze(0).to(device)
#    comp_tensor = torch.from_numpy(binarized_dct(gray_image)).unsqueeze(0).to(device)
#
#    # --- Load model ---
#    model = MCNet().to(device)
#    model.load_state_dict(torch.load(model_path, map_location=device))
#    model.eval()
#
#    # --- Inference ---
#    with torch.no_grad():
#        output = model(spatial_tensor, freq_tensor.float(), comp_tensor.float())
#        prediction = (output.item() > 0.5)
#        print(f"Prediction: {'REAL ✅' if prediction else 'FAKE ❌'} (Score: {output.item():.4f})")
#
## ----------------------------
## Driver Code
## ----------------------------
#if __name__ == '__main__':
#    model_path = "/home/user/Deepfake/test/mcnet_finetuned_best.pth"
#    test_image_path = "/home/user/Deepfake/Datasets/real_vs_fake/real-vs-fake/train/fake/ZZD8A4LPXT.jpg"  # <-- change this
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#    print(f"🔍 Evaluating on: {test_image_path}")
#    test_single_image(model_path, test_image_path, device)

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
from mcnet_finetune import MCNet, block_dct, binarized_dct  # If model is modularized

# ----------------------------
# Define utils here if not importing
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
# Inference and Visualization
# ----------------------------
def test_and_show_image(model_path, image_path, device):
    # Load image
    image = Image.open(image_path).convert('RGB').resize((128, 128))
    image_np = np.array(image).astype(np.float32) / 255.0
    gray_image = np.mean(image_np, axis=2)

    # Preprocess
    spatial_transform = transforms.Compose([
        transforms.CenterCrop(128),
        transforms.ToTensor()
    ])
    spatial_tensor = spatial_transform(image).unsqueeze(0).to(device)
    freq_tensor = torch.from_numpy(block_dct(image_np)).unsqueeze(0).to(device)
    comp_tensor = torch.from_numpy(binarized_dct(gray_image)).unsqueeze(0).to(device)

    # Load model
    model = MCNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Predict
    with torch.no_grad():
        output = model(spatial_tensor, freq_tensor.float(), comp_tensor.float())
        score = output.item()
        prediction = 'REAL ✅' if score > 0.5 else 'FAKE ❌'

    # Show image with prediction
    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Prediction: {prediction}\nConfidence: {score:.4f}", fontsize=12, color='green' if score > 0.5 else 'red')
    plt.tight_layout()
    plt.show()

# ----------------------------
# Main Driver
# ----------------------------
if __name__ == '__main__':
    model_path = "/home/user/Deepfake/test/mcnet_finetuned_best.pth"
    test_image_path = "/home/user/Deepfake/Datasets/handcrafted_real_fake/fake/mid_72_1111.jpg"  # Change to your test image
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"🔍 Running inference on {test_image_path}")
    test_and_show_image(model_path, test_image_path, device)
