import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import numpy as np
from skimage.transform import resize
import os

# 1. SFFNPlus Model Definition
class SFFNPlus(nn.Module):
    def __init__(self):
        super(SFFNPlus, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),       # 0
            nn.BatchNorm2d(16),                               # 1
            nn.ReLU(),                                        # 2
            nn.MaxPool2d(2),                                  # 3

            nn.Conv2d(16, 32, kernel_size=3, padding=1),      # 4
            nn.BatchNorm2d(32),                               # 5
            nn.ReLU(),                                        # 6
            nn.MaxPool2d(2),                                  # 7

            nn.Conv2d(32, 64, kernel_size=3, padding=1),      # 8
            nn.BatchNorm2d(64),                               # 9
            nn.ReLU(),                                        # 10
            nn.MaxPool2d(2),                                  # 11

            nn.Conv2d(64, 128, kernel_size=3, padding=1),     # 12
            nn.BatchNorm2d(128),                              # 13
            nn.ReLU(),                                        # 14
            nn.MaxPool2d(2),                                  # 15

            nn.Conv2d(128, 256, kernel_size=3, padding=1),    # 16
            nn.BatchNorm2d(256),                              # 17
            nn.ReLU(),                                        # 18
            nn.MaxPool2d(2),                                  # 19

            nn.Conv2d(256, 512, kernel_size=3, padding=1),    # 20
            nn.BatchNorm2d(512),                              # 21
            nn.ReLU(),                                        # 22
            nn.AdaptiveAvgPool2d((1, 1))                      # 23
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

# 2. Grad-CAM Class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook()

    def hook(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor):
        output = self.model(input_tensor)
        self.model.zero_grad()
        output.backward(torch.tensor([[1.0]], device=input_tensor.device))
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam = resize(cam, (224, 224), mode='reflect', anti_aliasing=True)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, output.item()

# 3. Preprocessing and Overlay
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    original = np.array(image)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0), original

def show_cam_on_image(original, cam, prediction, confidence, layer_name):
    image_resized = np.array(Image.fromarray(original).resize((224, 224))).astype(np.float32) / 255
    cmap = cm.Greens if prediction == "Real" else cm.Reds
    heatmap = cmap(cam)[..., :3]
    overlay = 0.5 * image_resized + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(6, 5))
    plt.imshow(overlay)
    plt.title(f"{layer_name} → {prediction} ({confidence:.4f})", fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# 4. Main
if __name__ == "__main__":
    model_path = "best_model_sffn.pth"
    image_path = "/home/user/Deepfake/ExternalTest/4f.jpg"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model = SFFNPlus().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    layer_dict = {
        "conv1_relu": model.features[2],
        "conv2_relu": model.features[6],
        "conv3_relu": model.features[10],
        "conv4_relu": model.features[14],
        "conv5_relu": model.features[18],
        "conv6_relu": model.features[22]
    }

    if not os.path.exists(image_path):
        print("Image not found.")
        exit()

    input_tensor, original_image = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)

    for name, layer in layer_dict.items():
        print(f"\n🔍 Grad-CAM @ {name}")
        cam_extractor = GradCAM(model, layer)
        cam, raw_score = cam_extractor.generate(input_tensor)
        prediction = "Real" if raw_score > 0.8 else "Fake"
        confidence = raw_score if prediction == "Real" else 1 - raw_score
        show_cam_on_image(original_image, cam, prediction, confidence, name)
