import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchcam.methods import GradCAM
from AmtenNet_train import AMTENNet

# Configuration
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load Model
model = AMTENNet()
model.load_state_dict(torch.load("amtennet_best.pt", map_location=device))
model.to(device)
model.eval()

# Grad-CAM Extractor for Conv6
cam_extractor = GradCAM(model, target_layer="conv6")

# Test Single Image with trace visualization and Grad-CAM
def test_single_image(img_path):
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, fmt, f1, f2, freu = model(tensor, return_features=True)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    print(f"Image: {img_path}")
    print(f"Probability Real: {probs[0,1]*100:.2f}%")
    print(f"Probability Fake: {probs[0,0]*100:.2f}%")
    print(f"Predicted Label: {'Real' if pred == 1 else 'Fake'}")

    # Visualize traces
    def show_tensor_image(tensor, title, num_channels=3):
        img = tensor.detach().cpu().numpy()
        plt.figure(figsize=(12, 4))
        for i in range(min(num_channels, img.shape[0])):
            plt.subplot(1, num_channels, i+1)
            channel_img = img[i]
            channel_img = (channel_img - channel_img.min()) / (channel_img.max() - channel_img.min() + 1e-8)
            plt.imshow(channel_img, cmap='gray')
            plt.title(f"{title} - Channel {i+1}")
            plt.axis('off')
        plt.show()

    show_tensor_image(fmt[0], "FMT - Manipulation Trace")
    show_tensor_image(f1[0], "F1 Feature Map (First 3 channels)")
    show_tensor_image(f2[0], "F2 Feature Map (First 3 channels)")
    show_tensor_image(freu[0, :3], "Freu Combined Feature Map (First 3 channels)")

    # Grad-CAM
    activation_map = cam_extractor(pred, logits)
    plt.imshow(img)
    plt.imshow(activation_map[0].cpu().numpy(), cmap='jet', alpha=0.5)
    plt.title("Grad-CAM on Conv6")
    plt.axis('off')
    plt.show()

# Example usage:
test_single_image("/home/user/Deepfake/ExternalTest/104r.jpg")
