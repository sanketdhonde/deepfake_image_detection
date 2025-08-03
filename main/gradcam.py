import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import numpy as np
import os
from skimage.transform import resize

# 1. Device Setup
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Load your trained ResNet50 model and weights
model_path = "best_model.pth"
model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 1),
    nn.Sigmoid()
)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# 3. Grad-CAM Class
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
        input_tensor = input_tensor.to(device)
        output = self.model(input_tensor)
        self.model.zero_grad()
        output.backward(torch.tensor([[1.0]], device=device))

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = resize(cam, (224, 224), mode='reflect', anti_aliasing=True)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# 4. Preprocess Image
def preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")
    orig = np.array(image)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0), orig

# 5. Display CAM Overlay
def show_cam_on_image(original, cam, layer_name):
    pil_image = Image.fromarray(original)
    original_resized = np.array(pil_image.resize((224, 224))).astype(np.float32) / 255

    heatmap = cm.jet(cam)[..., :3]  # Drop alpha
    overlay = 0.5 * original_resized + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(6, 5))
    plt.imshow(overlay)
    plt.title(f"Grad-CAM @ {layer_name}", fontsize=14, fontweight="bold")
    plt.axis('off')
    plt.show()

# 6. Layers to visualize
layer_dict = {
    'layer2[3].conv3': model.layer2[3].conv3,
    'layer2[2].conv3': model.layer2[2].conv3,
    'layer2[1].conv3': model.layer2[1].conv3,
    'layer3[5].conv3': model.layer3[5].conv3,
    'layer3[4].conv3': model.layer3[4].conv3,
    'layer3[3].conv3': model.layer3[3].conv3,
    'layer3[2].conv3': model.layer3[2].conv3,
    'layer4[1].conv3': model.layer4[1].conv3,
    'layer4[2].conv3': model.layer4[2].conv3
}

# 7. Main
if __name__ == "__main__":
    img_path = "/home/user/Deepfake/Datasets/handcrafted_real_fake/real/real_00006.jpg"
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        exit()

    input_tensor, original_image = preprocess_image(img_path)

    for name, layer in layer_dict.items():
        print(f"\nGenerating Grad-CAM for layer: {name}")
        cam_generator = GradCAM(model, layer)
        cam = cam_generator.generate(input_tensor)
        show_cam_on_image(original_image, cam, name)
