import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import numpy as np
import os
from skimage.transform import resize
from effnet_cbam import EfficientNet_CBAM

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = EfficientNet_CBAM()
model.load_state_dict(torch.load("efficient_cbam_best_model.pth", map_location=device))
model = model.to(device)
model.eval()

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
        return cam, output.item()

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
    return transform(image).unsqueeze(0).to(device), orig

def show_cam_on_image(original, cam, prediction, confidence, layer_name):
    pil_image = Image.fromarray(original)
    original_resized = np.array(pil_image.resize((224, 224))).astype(np.float32) / 255
    heatmap = cm.jet(cam)[..., :3]
    overlay = 0.5 * original_resized + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)
    plt.figure(figsize=(6, 5))
    plt.imshow(overlay)
    plt.title(f"{prediction} ({confidence:.4f}) @ {layer_name}", fontsize=14)
    plt.axis('off')
    plt.show()

layer_dict = {
    "block2_CBAM": model.block2[1],
    "block4_CBAM": model.block4[1],
    "block6_CBAM": model.block6[1],
}

if __name__ == "__main__":
    img_path = "/home/user/Deepfake/Datasets/real_vs_fake/real-vs-fake/test/fake/02XAKN4F4U.jpg"
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        exit()
    input_tensor, original_image = preprocess_image(img_path)
    for name, layer in layer_dict.items():
        print(f"\nGenerating Grad-CAM for layer: {name}")
        cam_generator = GradCAM(model, layer)
        cam, output = cam_generator.generate(input_tensor)
        prediction = "Real" if output > 0.8 else "Fake"
        confidence = output if prediction == "Real" else 1 - output
        show_cam_on_image(original_image, cam, prediction, confidence, name)

#def predict_image(image_path, model, device):
#    try:
#        image = Image.open(image_path).convert("RGB")
#    except Exception as e:
#        print(f"Error loading image: {e}")
#        return None
#
#    input_tensor,og = preprocess_image(image_path)
#
#    with torch.no_grad():
#        output = model(input_tensor)
#        prob = output.item()
#        prediction = "Real" if prob > 0.8 else "Fake"
#        prob = prob if prediction == "Real" else 1 - prob
#
##    cam = generate_gradcam(model, input_tensor, target_class=1 if prediction == "Real" else 0)
##    overlay = overlay_heatmap(image, cam, prediction)
#
#    plt.figure(figsize=(6, 6))
#    plt.imshow(og)
#    plt.axis("off")
#    plt.title(f"Prediction: {prediction} (Confidence: {prob:.4f})", 
#              fontsize=14, fontweight="bold", color="blue")
#    plt.show()
# 
#
#    print(f"🔹 Image: {image_path}\n🔹 Prediction: {prediction} (Confidence: {prob:.4f})")
#    return prediction, prob
#
#
#
#
##image_path = "/home/user/Deepfake/
#image_path = "/home/user/Deepfake/Datasets/real_vs_fake/real-vs-fake/test/fake/02XAKN4F4U.jpg"
#
#
#predict_image(image_path, model, device)
