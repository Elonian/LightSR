import os
import torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from models.model import SRConvnet  # Ensure this is implemented and available

# ---------- Configuration ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_FOLDER = "predictions"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------- List of (Checkpoint, Image) Pairs ----------
CHECKPOINT_IMAGE_PAIRS = [
    ("C:\\Users\\chand\\Downloads\\LightSR\\LightSR\\dif2k_checkpoints\\checkpoints_2x\\model_x2_135.pt",
     "C:\\Users\\chand\\Intro_To_Visual_Learning\\Final_Project\\DML\\sr\\datasets\\FigShare\\Set5\\image_SRF_2\\img_003_SRF_2_LR.png"),
    ("C:\\Users\\chand\\Downloads\\LightSR\\LightSR\\dif2k_checkpoints\\checkpoints_3x\\model_x3_200.pt",
     "C:\\Users\\chand\\Intro_To_Visual_Learning\\Final_Project\\DML\\sr\\datasets\\FigShare\\Set5\\image_SRF_3\\img_003_SRF_3_LR.png"),
    ("C:\\Users\\chand\\Downloads\\LightSR\\LightSR\\dif2k_checkpoints\\checkpoints_4x\\model_x4_190.pt",
     "C:\\Users\\chand\\Intro_To_Visual_Learning\\Final_Project\\DML\\sr\\datasets\\FigShare\\Set5\\image_SRF_4\\img_003_SRF_4_LR.png"),
    ("C:\\Users\\chand\\Downloads\\LightSR\\LightSR\\realsr_checkpoints\\checkpoints_2x\\model_x2_115.pt",
     "C:\\Users\\chand\\Intro_To_Visual_Learning\\Final_Project\\DML\\sr\\datasets\\FigShare\\Set5\\image_SRF_2\\img_003_SRF_2_LR.png"),
    ("C:\\Users\\chand\\Downloads\\LightSR\\LightSR\\realsr_checkpoints\\checkpoints_3x\\model_x3_180.pt",
     "C:\\Users\\chand\\Intro_To_Visual_Learning\\Final_Project\\DML\\sr\\datasets\\FigShare\\Set5\\image_SRF_3\\img_003_SRF_3_LR.png"),
    ("C:\\Users\\chand\\Downloads\\LightSR\\LightSR\\realsr_checkpoints\\checkpoints_4x\\model_x4_200.pt",
     "C:\\Users\\chand\\Intro_To_Visual_Learning\\Final_Project\\DML\\sr\\datasets\\FigShare\\Set5\\image_SRF_4\\img_003_SRF_4_LR.png"),
]

# ---------- Load & Normalize Image ----------
def load_image(image_path):
    img = np.array(Image.open(image_path)).astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    return img_tensor.to(DEVICE)

# ---------- Load Model ----------
def load_model(ckpt_path, scale):
    model = SRConvnet(scale=scale, num_kernels=8, num_acb=4, dimension=64, num_heads=4).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ---------- Inference & Save ----------
def run_predictions():
    for ckpt_path, img_path in CHECKPOINT_IMAGE_PAIRS:
        print(f"Processing {os.path.basename(img_path)} with {os.path.basename(ckpt_path)}")

        # Extract scale factor from checkpoint filename (e.g., model_x3_200.pt)
        try:
            scale = int(ckpt_path.split("_x")[1].split("_")[0])
        except:
            raise ValueError(f"Unable to parse scale from checkpoint: {ckpt_path}")

        # Load model and LR image
        model = load_model(ckpt_path, scale)
        lr = load_image(img_path)  # [1, C, H, W]

        with torch.no_grad():
            # Model super-resolution
            sr = model(lr).clamp(0, 1)

            # Bilinear upsampling
            lr_upsampled = torch.nn.functional.interpolate(
                lr, scale_factor=scale, mode='bilinear', align_corners=False
            ).clamp(0, 1)

        # File naming
        img_base = os.path.splitext(os.path.basename(img_path))[0]
        model_tag = os.path.splitext(os.path.basename(ckpt_path))[0]

        # Save SR output
        sr_path = os.path.join(OUTPUT_FOLDER, f"{img_base}_{model_tag}_SR.png")
        save_image(sr, sr_path)
        print(f"Saved model SR image: {sr_path}")

        # Save bilinear output
        bilinear_path = os.path.join(OUTPUT_FOLDER, f"{img_base}_{model_tag}_BILINEAR.png")
        save_image(lr_upsampled, bilinear_path)
        print(f"Saved bilinear upsampled image: {bilinear_path}")

if __name__ == "__main__":
    run_predictions()
