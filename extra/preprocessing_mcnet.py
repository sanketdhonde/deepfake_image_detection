import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.fftpack
import random

# ---------------------------
# Preprocessing Functions
# ---------------------------

def get_spatial_patch(image_path, patch_size=512):
    image = Image.open(image_path).convert('RGB')
    w, h = image.size

    if w < patch_size or h < patch_size:
        image = image.resize((patch_size, patch_size))

    left = random.randint(0, w - patch_size)
    top = random.randint(0, h - patch_size)

    patch = image.crop((left, top, left + patch_size, top + patch_size))
    patch_np = np.array(patch).astype(np.float32) / 255.0
    patch_np = np.transpose(patch_np, (2, 0, 1))  # (C, H, W)

    return patch_np, patch  # Return both NumPy and PIL versions

def block_dct(image, block_size=4, stride=2):
    h, w, c = image.shape
    num_blocks_y = (h - block_size) // stride + 1
    num_blocks_x = (w - block_size) // stride + 1

    dct_features = []

    for ch in range(c):
        channel = image[:, :, ch]
        features = []

        for y in range(0, h - block_size + 1, stride):
            for x in range(0, w - block_size + 1, stride):
                patch = channel[y:y+block_size, x:x+block_size]
                dct_patch =
scipy.fftpack.dct(scipy.fftpack.dct(patch.T, norm='ortho').T,
norm='ortho')
                features.append(dct_patch.flatten())

        features = np.array(features).T  # (16, num_blocks)
        dct_features.append(features)

    dct_features = np.concatenate(dct_features, axis=0)  # Shape: (48,
num_blocks)

    side = int(np.sqrt(dct_features.shape[1]))
    dct_features = dct_features.reshape(48, side, side)

    return dct_features

def simulate_binarized_dct(patch, num_bins=11, threshold=10):
    h, w, c = patch.shape
    gray = np.mean(patch, axis=2)

    dct_coeff = scipy.fftpack.dct(scipy.fftpack.dct(gray.T,
norm='ortho').T, norm='ortho')
    dct_coeff = np.abs(dct_coeff)
    dct_coeff = np.clip(dct_coeff, 0, threshold)

    binarized = np.zeros((num_bins, h, w))

    for i in range(num_bins):
        binarized[i] = (dct_coeff == i).astype(np.float32)

    return binarized

# ---------------------------
# True Frequency Spectrum
# ---------------------------

def visualize_frequency_spectrum(image_patch):
    """
    image_patch: (H, W, C) NumPy array
    """
    plt.figure(figsize=(15, 4))

    for i in range(3):  # For each RGB channel
        plt.subplot(1, 3, i + 1)

        channel = image_patch[:, :, i]
        dct_channel = scipy.fftpack.dct(scipy.fftpack.dct(channel.T,
norm='ortho').T, norm='ortho')

        magnitude = np.log(np.abs(dct_channel) + 1e-3)  # Log scale
for better visibility

        plt.imshow(magnitude, cmap='gray')
        plt.title(f"True Frequency Spectrum - Channel {i+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# ---------------------------
# Combined Comparison Visualization
# ---------------------------

def visualize_all(spatial_patch, freq_features, comp_features):

    plt.figure(figsize=(20, 12))

    # Spatial Image
    plt.subplot(3, 4, 1)
    img = np.transpose(spatial_patch, (1, 2, 0))
    plt.imshow(img)
    plt.title("Spatial Patch (RGB)")
    plt.axis('off')

    # DCT Block Features (3 random channels)
    for i in range(3):
        plt.subplot(3, 4, i + 2)
        plt.imshow(freq_features[i], cmap='gray')
        plt.title(f"DCT Block Feature Channel {i+1}")
        plt.axis('off')

    # Compression Features (3 random channels)
    for i in range(3):
        plt.subplot(3, 4, i + 5)
        plt.imshow(comp_features[i], cmap='gray')
        plt.title(f"Binarized DCT Channel {i+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# ---------------------------
# Main Code
# ---------------------------

image_path = "/home/user/Deepfake/ExternalTest/1f.jpg"

# Spatial Patch with larger size
spatial_patch, pil_patch = get_spatial_patch(image_path, patch_size=512)

# Frequency Features (DCT Block Features)
patch_img = np.transpose(spatial_patch, (1, 2, 0))  # H, W, C
freq_features = block_dct(patch_img)

# Compression Features
comp_features = simulate_binarized_dct(patch_img)

# Visual Comparison
visualize_all(spatial_patch, freq_features, comp_features)

# True Frequency Domain Spectrum
visualize_frequency_spectrum(patch_img)