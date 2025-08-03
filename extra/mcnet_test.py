import torch
import os
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from TRAIN_MCNET2 import MCNet, block_dct, binarized_dct  # Assuming this is your correct model file

# Select device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Image preprocessing for spatial stream
spatial_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Build MCNet model for testing
def build_mcnet_for_test():
    model = MCNet()
    model.load_state_dict(torch.load("/home/user/Deepfake/test/mcnet_finetuned_handcrafted.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

# Predict and visualize
def predict_image_mcnet(image_path, model, device):
    try:
        image = Image.open(image_path).convert("RGB").resize((128, 128))
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    image_np = np.array(image).astype(np.float32) / 255.0
    spatial_tensor = spatial_transform(image).unsqueeze(0).to(device)

    freq_feature = block_dct(image_np)
    freq_feature = torch.from_numpy(freq_feature).unsqueeze(0).float().to(device)

    gray_image = np.mean(image_np, axis=2)
    comp_feature = binarized_dct(gray_image)
    comp_feature = torch.from_numpy(comp_feature).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = model(spatial_tensor, freq_feature, comp_feature)
        prob = output.item()
        prediction = "Real" if prob > 0.8 else "Fake"

        if prediction == "Fake":
            prob = 1 - prob

    # Show image with prediction
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Prediction: {prediction} (Confidence: {prob:.4f})",
              fontsize=14, fontweight="bold", color="blue")
    plt.show()

    print(f"🔹 Image: {image_path}\n🔹 Prediction: {prediction} (Confidence: {prob:.4f})")
    return prediction, prob

# ----------------------------
# Example Call
# ----------------------------

model = build_mcnet_for_test()
image_path =  "/home/user/Deepfake/ExternalTest/103r.png" # Your test image path
predict_image_mcnet(image_path, model, device)
