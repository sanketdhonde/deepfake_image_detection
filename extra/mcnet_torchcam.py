import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchcam.methods import GradCAM
from TRAIN_MCNET2 import MCNet, block_dct, binarized_dct

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def visualize_gradcam(image_path, model_path):
    model = MCNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert('RGB').resize((128, 128))
    image_tensor = transform(image).unsqueeze(0).to(device)

    image_np = np.array(image).astype(np.float32) / 255.0
    freq_feature = block_dct(image_np)
    freq_feature = torch.from_numpy(freq_feature).unsqueeze(0).float().to(device)
    gray_image = np.mean(image_np, axis=2)
    comp_feature = binarized_dct(gray_image)
    comp_feature = torch.from_numpy(comp_feature).unsqueeze(0).float().to(device)

    layer_names = {
        "VANet_BT3_final": model.vanet.spatial[9],
        "Freq_BT3_final": model.freq_learner.freq[11],
        "CANet_BT3_final": model.canet.bt3_blocks[3]
    }

    for name, layer in layer_names.items():
        cam_extractor = GradCAM(model, target_layer=layer)

        output = model(image_tensor, freq_feature, comp_feature)  # raw logits
 

        activation_map = cam_extractor(output, class_idx=0)[0].cpu().numpy()

        heatmap = np.maximum(activation_map, 0)
        heatmap /= np.max(heatmap) + 1e-8
        heatmap = np.uint8(255 * heatmap)
        heatmap = np.array(Image.fromarray(heatmap).resize((128, 128)))

        image_np_uint8 = (image_np * 255).astype(np.uint8)
        overlay = plt.get_cmap('jet')(heatmap / 255.0)[..., :3]
        overlay = (overlay * 255).astype(np.uint8)
        blended = (0.6 * image_np_uint8 + 0.4 * overlay).astype(np.uint8)

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(image_np_uint8)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(blended)
        plt.title(f"GradCAM: {name}")
        plt.axis('off')

        plt.show()

# ----------------------------
# Example Call
# ----------------------------
image_path = '/home/user/Deepfake/ExternalTest/10f.jpg'
model_path = 'best_model_mcnet.pth'
visualize_gradcam(image_path, model_path)
