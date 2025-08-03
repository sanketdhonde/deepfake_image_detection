import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

# Load your local image (update the filename if needed)
image = Image.open("/home/user/Deepfake/ExternalTest/100r.jpeg").convert('RGB')

# Preprocess the image
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
img_tensor = transform(image).unsqueeze(0)  # Shape: [1, 3, 128, 128]

# Define a multi-layer ConvNet (2 blocks)
model = nn.Sequential(
    nn.Conv2d(3, 8, kernel_size=3, padding=1),     # Layer 1: Conv
    nn.BatchNorm2d(8),                             # Layer 2: BatchNorm
    nn.ReLU(),                                     # Layer 3: ReLU
    nn.MaxPool2d(kernel_size=2, stride=2),         # Layer 4: MaxPooling (Output: 64x64)

    nn.Conv2d(8, 16, kernel_size=3, padding=1),    # Layer 5: Conv
    nn.BatchNorm2d(16),                            # Layer 6: BatchNorm
    nn.ReLU(),                                     # Layer 7: ReLU
    nn.MaxPool2d(kernel_size=2, stride=2)          # Layer 8: MaxPooling (Output: 32x32)
)

# Pass image through the network and save intermediate outputs
outputs = []
x = img_tensor
for layer in model:
    x = layer(x)
    outputs.append(x.clone())

# Visualize feature maps from the first and second conv layers
fig, axes = plt.subplots(2, 9, figsize=(22, 6))

# First layer feature maps
for i in range(8):
    fmap = outputs[3][0, i].detach().numpy()
    axes[0, i].imshow(fmap, cmap='viridis')
    axes[0, i].set_title(f'Layer 1 - Map {i+1}')
    axes[0, i].axis('off')

axes[0, 8].imshow(image.resize((128, 128)))
axes[0, 8].set_title("Original")
axes[0, 8].axis('off')

# Second layer feature maps (16 channels, show first 8)
for i in range(8):
    fmap = outputs[7][0, i].detach().numpy()
    axes[1, i].imshow(fmap, cmap='plasma')
    axes[1, i].set_title(f'Layer 2 - Map {i+1}')
    axes[1, i].axis('off')

axes[1, 8].axis('off')  # Empty

plt.tight_layout()
plt.show()

