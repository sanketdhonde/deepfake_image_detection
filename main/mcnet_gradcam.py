import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import scipy.fftpack
from mcnet_finetune import MCNet, block_dct, binarized_dct

def generate_gradcam(model, input_tensor, target_layer, device):
    gradients = []
    activations = []

    def save_gradients_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activations_hook(module, input, output):
        activations.append(output)

    handle1 = target_layer.register_forward_hook(save_activations_hook)
    handle2 = target_layer.register_full_backward_hook(save_gradients_hook)

    model.eval()
    input_tensor = [t.unsqueeze(0).to(device) for t in input_tensor]
    output = model(*input_tensor)
    model.zero_grad()
    output.backward()

    grads = gradients[0].squeeze().detach().cpu().numpy()
    acts = activations[0].squeeze().detach().cpu().numpy()
    weights = np.mean(grads, axis=(1, 2))
    cam = np.sum(weights[:, None, None] * acts, axis=0)
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)

    handle1.remove()
    handle2.remove()
    return cam

def preprocess_image(image_path, device):
    image = Image.open(image_path).convert('RGB').resize((128, 128))
    image_np = np.array(image).astype(np.float32) / 255.0
    gray_image = np.mean(image_np, axis=2)

    spatial_transform = transforms.Compose([
        transforms.CenterCrop(128),
        transforms.ToTensor()
    ])
    spatial_tensor = spatial_transform(image).unsqueeze(0).to(device)
    freq_tensor = torch.from_numpy(block_dct(image_np)).float().unsqueeze(0).to(device)
    comp_tensor = torch.from_numpy(binarized_dct(gray_image)).float().unsqueeze(0).to(device)

    return spatial_tensor, freq_tensor, comp_tensor, image

def test_single_image_with_gradcam(model_path, image_path, device):
    print(f"\n🔍 Evaluating and visualizing Grad-CAM for: {image_path}")
    spatial_tensor, freq_tensor, comp_tensor, pil_img = preprocess_image(image_path, device)

    model = MCNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        output = model(spatial_tensor, freq_tensor, comp_tensor)
        pred_score = output.item()
        pred_label = "REAL ✅" if pred_score > 0.5 else "FAKE ❌"
        print(f"Prediction: {pred_label} (Score: {pred_score:.4f})")

    # Grad-CAM for all layers in VANet, FrequencyLearner, and CANet
    layers = {
        f"VANet_{i}": layer for i, layer in enumerate(model.vanet.spatial)
    }
    layers.update({
        f"Freq_{i}": layer for i, layer in enumerate(model.freq_learner.freq)
    })
    layers.update({
        f"CANet_{i}": layer for i, layer in enumerate(model.canet.bt3_blocks)
    })

    for name, layer in layers.items():
        cam = generate_gradcam(model, [spatial_tensor.squeeze(0), freq_tensor.squeeze(0), comp_tensor.squeeze(0)], layer, device)
        cam_img = Image.fromarray(np.uint8(255 * cam)).resize((128, 128))

        plt.figure(figsize=(6, 3))
        plt.imshow(pil_img)
        plt.imshow(cam_img, cmap='jet', alpha=0.5)
        plt.title(f"{name}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    model_path = "/home/user/Deepfake/test/mcnet_finetuned_best.pth"
    test_image_path = "/home/user/Deepfake/Datasets/real_vs_fake/real-vs-fake/train/fake/ZZD8A4LPXT.jpg"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_single_image_with_gradcam(model_path, test_image_path, device)
