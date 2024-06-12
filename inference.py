import sys
import configs.infer_config_autoBG as cfg
import cv2
import torch
from utils.data_loader import videoLoader
import time
import numpy as np
import os

assert len(sys.argv) > 2, "You must provide the input video path and output directory"
inp_path = sys.argv[1]
out_dir = sys.argv[2]

# Create output directory if it does not exist
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Start Video Loader
vid_loader = videoLoader(
    inp_path, 
    empty_bg=cfg.BSUVNet.emtpy_bg, 
    empty_win_len=cfg.BSUVNet.empty_win_len,
    empty_bg_path=cfg.BSUVNet.empty_bg_path,
    recent_bg=cfg.BSUVNet.recent_bg,
    seg_network=cfg.BSUVNet.seg_network,
    transforms_pre=cfg.BSUVNet.transforms_pre,
    transforms_post=cfg.BSUVNet.transforms_post
)
tensor_loader = torch.utils.data.DataLoader(dataset=vid_loader, batch_size=1)

# Load BSUV-Net
bsuvnet = torch.load(cfg.BSUVNet.model_path, map_location='cpu').eval()

# Start Inference
num_frames = 0
start = time.time()  # Inference start time
with torch.no_grad():
    for inp in tensor_loader:
        num_frames += 1
        bgs_pred = bsuvnet(inp.float()).numpy()[0, 0, :, :]

        # Normalize the prediction to be in the range [0, 255]
        bgs_pred = (bgs_pred * 255).astype(np.uint8)

        # Save the frame as an image file
        frame_filename = os.path.join(out_dir, f"frame_{num_frames:06d}.png")
        cv2.imwrite(frame_filename, bgs_pred)

        if num_frames % 100 == 0:
            print(f"{num_frames} frames completed")

end = time.time()  # Inference end time
fps = num_frames / (end - start)

# Extract height and width from the last processed frame
height, width = bgs_pred.shape
print(f"{fps:.3f} FPS for ({width}, {height}) resolution")
