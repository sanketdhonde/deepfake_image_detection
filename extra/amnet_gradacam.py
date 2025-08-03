import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import numpy as np
import os
from skimage.transform import resize
from amnet_cbam import AMTENNet  # Rename your combined model file to this

# Device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model_path = "best_model_amtennetcbam.pth"
model = AMTENNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# GradCAM class
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
        output.backward(torch.ones_like(output))

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam = resize(cam, (224, 224), mode='reflect', anti_aliasing=True)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# Image preprocessing
def preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")
    orig = np.array(image)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0), orig

# CAM display
def show_cam_on_image(original, cam, layer_name):
    pil_image = Image.fromarray(original)
    original_resized = np.array(pil_image.resize((224, 224))).astype(np.float32) / 255
    heatmap = cm.jet(cam)[..., :3]
    overlay = 0.5 * original_resized + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)
    plt.figure(figsize=(6, 5))
    plt.imshow(overlay)
    plt.title(f"Grad-CAM @ {layer_name}", fontsize=14)
    plt.axis('off')
    plt.show()

# Extract all Conv2d layers
def get_target_conv_layers(model):
    # Return only selected key convolutional layers for Grad-CAM visualization
    return {
        "trace_conv1": model.trace_conv1,
        "trace_conv2": model.trace_conv2,
        "trace_conv3": model.trace_conv3
    }
    
def get_target_cbam_layers(model):
    return {
        "cbam1": model.cbam1,
        "cbam2": model.cbam2,
        "cbam3": model.cbam3
    }

# Main
if __name__ == "__main__":
    img_path = "/home/user/Deepfake/ExternalTest/6f.jpg"
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        exit()

    input_tensor, original_image = preprocess_image(img_path)

    #target_layers = get_target_conv_layers(model)
    # OR if you want CBAM targets
    target_layers = get_target_cbam_layers(model)


    for name, layer in target_layers.items():
        print(f"\n🎯 Grad-CAM for layer: {name}")
        cam_generator = GradCAM(model, layer)
        cam = cam_generator.generate(input_tensor)
        show_cam_on_image(original_image, cam, name)

