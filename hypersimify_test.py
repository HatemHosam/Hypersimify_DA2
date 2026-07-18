import torch
import torch.nn.functional as F
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from depth_anything_v2.dpt import DepthAnythingV2


model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

def main():
    # 1. Initialize the parser
    parser = argparse.ArgumentParser(description="Hypersimify DepthAnythingV2")

    # 2. Add positional arguments (Required by default)
    parser.add_argument("--img_path", type=str, help="Display a square of a given number", required= True)

    # 3. Add optional arguments (Flags that take a value)
    parser.add_argument("--checkpoint_path", type=str, help="Path to the DA2 indoor metric checkpoint file", required= True)
    parser.add_argument("--device", type=str, help="device to run the model cuda or cpu", default = 'cpu')
    
    args = parser.parse_args()
    
    DEVICE = args.device
    encoder = 'vitl' # or 'vits', 'vitb'
    dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 20 # 20 for indoor model, 80 for outdoor model

    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'))
    model.to('cuda')
    model.eval()
    
    image_path = args.img_path
    # Load and preprocess an image.
    raw_img = cv2.imread(image_path)
    # First, resize the image to DA2 native resolution 
    raw_img1 = cv2.resize(raw_img, (714,518))

    #Second, remove highlights (very bright pixels)
    #Convert to grayscale to identify bright spots
    gray = cv2.cvtColor(raw_img1, cv2.COLOR_BGR2GRAY)
    # High values are mostly >235 detect very bright pixels
    ret, mask_hl = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    # Inpaint the highlights
    im0 = cv2.inpaint(raw_img1, mask_hl, 3, cv2.INPAINT_TELEA)


    # Third. CLAHE on luminance (LAB space) ---
    lab = cv2.cvtColor(im0, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    im0 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    depth = model.infer_image(im0) # HxW raw depth map in numpy
    depth = F.interpolate(depth, (3008, 4112), mode="bilinear", align_corners=True)
    depth= depth.squeeze(0).squeeze(0).cpu().numpy()

    depth_upsampled = cv2.ximgproc.guidedFilter(
        guide=raw_img.copy(), 
        src=depth, 
        radius=6, 
        eps=0.01
    )
            
    depth1 = depth_upsampled * 100  # Depth in [cm].
    plt.figure('depth')
    plt.imshow(depth, cmap= 'magma')
    plt.show()


if __name__ == "__main__":
    main()