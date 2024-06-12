import os
import configs.infer_config_autoBG as cfg
import cv2
import torch
from utils.data_loader import videoLoader
import numpy as np

def get_binary_mask(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of image file paths in the input folder
    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]

    # Load the images
    frames = [cv2.imread(path) for path in image_paths]

    # Create a videoLoader instance with the loaded frames
    vid_loader = videoLoader(frames, empty_bg=cfg.BSUVNet.emtpy_bg, 
                             empty_win_len=cfg.BSUVNet.empty_win_len,
                             recent_bg=cfg.BSUVNet.recent_bg,
                             seg_network=cfg.BSUVNet.seg_network,
                             transforms_pre=cfg.BSUVNet.transforms_pre,
                             transforms_post=cfg.BSUVNet.transforms_post)
    tensor_loader = torch.utils.data.DataLoader(dataset=vid_loader, batch_size=1)

    # Load BSUV-Net
    bsuvnet = torch.load(cfg.BSUVNet.model_path)
    bsuvnet.cuda().eval()

    # Start Inference
    with torch.no_grad():
        for idx, inp in enumerate(tensor_loader):
            bgs_pred = bsuvnet(inp.cuda().float()).cpu().numpy()[0, 0, :, :]
            binary_mask = (bgs_pred > 0.5).astype(np.uint8)

            # Save the binary mask
            output_path = os.path.join(output_folder, f"mask_{idx}.png")
            cv2.imwrite(output_path, binary_mask * 255)

    print(f"Binary masks saved in {output_folder}")

# Example usage
input_folder = r'C:\Users\Aurja\OneDrive\Desktop\Thesis_Testing_Dataset\Public\UCF-fishes'
output_folder = r'C:\Users\Aurja\Downloads\BSUV-Net-inference\output'
get_binary_mask(input_folder, output_folder)
