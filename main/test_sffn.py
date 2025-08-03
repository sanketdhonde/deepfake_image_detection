import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define the same model architecture used during training
class SFFNPlus(nn.Module):
    def __init__(self):
        super(SFFNPlus, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load model
def build_model_for_test(model_path='best_model_sffn.pth'):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = SFFNPlus().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

# Prediction function
def predict_image(image_path, model, transform, device):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = output.item()
        prediction = "Real" if prob > 0.8 else "Fake"
        confidence = prob if prediction == "Real" else 1 - prob

    # Show result
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"{prediction} (Confidence: {confidence:.4f})", fontsize=14, fontweight='bold')
    plt.show()

    print(f"🔍 Image: {image_path}")
    print(f"📌 Prediction: {prediction} (Confidence: {confidence:.4f})")

# Set up transform (should match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Test example
if __name__ == "__main__":
    image_path = "/home/user/Deepfake/ExternalTest/101f.jpg"
    model, device = build_model_for_test()
    predict_image(image_path, model, transform, device)
