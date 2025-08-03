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
model.load_state_dict(torch.load("amtennet_finetuned_handcrafted.pth", map_location=device))
model.to(device)
model.eval()

# List of target layers
target_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'conv9']

# Function to test single image with Grad-CAM for all layers
# Function to test single image with Grad-CAM for all layers
def test_single_image(img_path):
    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    img_np = np.array(img)

#    for layer in target_layers:
#        print(f"Generating Grad-CAM for {layer}...")
#
#        cam_extractor = GradCAM(model, target_layer=layer)  # Initialize BEFORE forward pass
#
#        output = model(input_tensor)  # Forward pass AFTER initializing GradCAM
#        pred = output.argmax(dim=1).item()
#
#        activation_map = cam_extractor(pred, output)
#        grayscale_cam = activation_map[0].cpu().detach().numpy().squeeze()
#
#        plt.figure(figsize=(5, 5))
#        plt.imshow(img.resize((128,128)))
#        plt.imshow(grayscale_cam, cmap='jet', alpha=0.5)
#        plt.title(f'Grad-CAM: {layer} | Predicted Class: {pred}')
#        plt.axis('off')
#        plt.show()
    for layer in target_layers:
        print(f"Generating Grad-CAM for {layer}...")
    
        cam_extractor = GradCAM(model, target_layer=layer)
        output = model(input_tensor)
        pred = output.argmax(dim=1).item()
    
        activation_map = cam_extractor(pred, output)  # List with one tensor
    
        # Check actual shape
        print(f"{layer} raw activation map shape: {activation_map[0].shape}")
    
        # Ensure proper 4D tensor shape: [N, C, H, W]
        if activation_map[0].dim() == 2:
            activation_map_tensor = activation_map[0].unsqueeze(0).unsqueeze(0)
        elif activation_map[0].dim() == 3:
            activation_map_tensor = activation_map[0].unsqueeze(0)
        else:
            activation_map_tensor = activation_map[0]
    
        activation_map_resized = F.interpolate(activation_map_tensor, size=(128, 128), mode='bilinear', align_corners=False)
    
        grayscale_cam = activation_map_resized.squeeze().cpu().numpy()
    
        plt.figure(figsize=(5, 5))
        plt.imshow(img.resize((128,128)))
        plt.imshow(grayscale_cam, cmap='jet', alpha=0.5)
        plt.title(f'Grad-CAM: {layer} | Predicted Class: {pred}')
        plt.axis('off')
        plt.show()



# Test the image
img_path = "/home/user/Deepfake/Datasets/real_vs_fake/real-vs-fake/test/fake/01MI46B2OH.jpg"
test_single_image(img_path)
