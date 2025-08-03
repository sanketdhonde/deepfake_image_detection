import numpy as np
import torch
import os
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Autoencoder definition
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

# Load best model
model = Autoencoder().to(device)
model.load_state_dict(torch.load("autoencoder_best.pth"))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Example threshold (should be calculated from real image errors)
threshold = 0.015  # Adjust based on your real image reconstruction error distribution

def test_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Compute reconstruction error
    criterion = nn.MSELoss(reduction='none')
    with torch.no_grad():
        output, _ = model(image_tensor)
        loss = criterion(output, image_tensor)
        recon_error = loss.view(loss.size(0), -1).mean(dim=1).item()

    print(f"Reconstruction error for {os.path.basename(image_path)}: {recon_error:.6f}")

    # Prediction based on threshold
    if recon_error <= threshold:
        print("Prediction: Real Image ✅")
    else:
        print("Prediction: Fake Image ❌")

    # Visualize original and reconstructed image
    output_image = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
    input_image = image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(input_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(output_image)
    plt.title('Reconstructed Image')
    plt.axis('off')

    plt.show()

# Example usage
image_path = "/home/user/Deepfake/Datasets/real_vs_fake/real-vs-fake/test/fake/009ZTJ3621.jpg" # Replace with your image path
test_image(image_path)
