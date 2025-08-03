import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.fftpack
from torchvision import transforms
from TRAIN_MCNET2 import MCNet, block_dct, binarized_dct  # Ensure correct path to your model file

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def apply_colormap_on_image(org_im, activation, colormap_name='jet'):
    color_map = plt.get_cmap(colormap_name)
    heatmap = color_map(activation)
    heatmap = np.delete(heatmap, 3, 2)  # Remove alpha channel
    heatmap = np.uint8(255 * heatmap)
    overlayed = heatmap * 0.4 + org_im * 0.6
    overlayed = overlayed.astype(np.uint8)
    return overlayed


def generate_gradcam(image_path, model_path):
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

    # Collect all layers
    layer_dict = {}

    # VANet Layers
    for idx, layer in enumerate(model.vanet.spatial):
        layer_dict[f'VANet_Layer_{idx}'] = layer

    # Frequency Learner Layers
    for idx, layer in enumerate(model.freq_learner.freq):
        layer_dict[f'FreqLearner_Layer_{idx}'] = layer

    # CANet Initial Convs
    layer_dict['CANet_DilatedConv'] = model.canet.dilated_conv
    layer_dict['CANet_RegularConv'] = model.canet.regular_conv
    layer_dict['CANet_BT1'] = model.canet.bt1

    # CANet BT2 Blocks
    for idx, layer in enumerate(model.canet.bt2_blocks):
        layer_dict[f'CANet_BT2_Layer_{idx}'] = layer

    # CANet BT3 Blocks
    for idx, layer in enumerate(model.canet.bt3_blocks):
        layer_dict[f'CANet_BT3_Layer_{idx}'] = layer

    for name, target_layer in layer_dict.items():
        feature_map = None
        gradients = None

        def hook_fn_forward(module, input, output):
            nonlocal feature_map
            feature_map = output

        def hook_fn_backward(module, grad_in, grad_out):
            nonlocal gradients
            gradients = grad_out[0]

        forward_handle = target_layer.register_forward_hook(hook_fn_forward)
        backward_handle = target_layer.register_full_backward_hook(hook_fn_backward)

        output = model(image_tensor, freq_feature, comp_feature)
        loss = output

        model.zero_grad()
        loss.backward()

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        for i in range(feature_map.shape[1]):
            feature_map[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(feature_map, dim=1).squeeze().cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)

        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
        else:
            heatmap = np.zeros_like(heatmap)

        heatmap = np.uint8(255 * heatmap)
        heatmap = np.array(Image.fromarray(heatmap).resize((128, 128)))

        image_np_uint8 = (image_np * 255).astype(np.uint8)
        cam_image = apply_colormap_on_image(image_np_uint8, heatmap / 255.0)

        plt.figure(figsize=(8, 4))
        plt.suptitle(f"Grad-CAM: {name}")
        plt.subplot(1, 2, 1)
        plt.imshow(image_np_uint8)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cam_image)
        plt.title(f"Activation Map")
        plt.axis('off')

        plt.show()

        forward_handle.remove()
        backward_handle.remove()


# ----------------------------
# Example Call
# ----------------------------

image_path =  "/home/user/Deepfake/Datasets/real_vs_fake/real-vs-fake/test/real/00063.jpg"# Your test image path
model_path = "/home/user/Deepfake/test/mcnet_finetuned_handcrafted.pth"

generate_gradcam(image_path, model_path)
