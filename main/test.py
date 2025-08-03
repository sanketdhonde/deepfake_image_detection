import torch
import os
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Select device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Build ResNet-50-based model
def build_model_for_test():
    model = resnet50(weights=None)

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )

    model.load_state_dict(torch.load('fine_tuned_combined.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

# Predict and visualize
def predict_image(image_path, model, transform, device):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
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

# Load model and predict
model = build_model_for_test()
image_path = "/home/user/Deepfake/ExternalTest/100r.jpeg"
predict_image(image_path, model, transform, device)
