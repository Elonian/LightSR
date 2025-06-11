import os
import requests
import zipfile
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset, DataLoader
from utils.util import calc_psnr, calc_ssim
from models.model import SRConvnet
import shutil


import stat

def fix_windows_permissions(path):
    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            full_path = os.path.join(root, name)
            try:
                os.chmod(full_path, stat.S_IWRITE)
            except Exception as e:
                print(f"Failed to set write permission for {full_path}: {e}")

DATASET_ZIP = os.path.normpath("C:\\Users\\chand\\Intro_To_Visual_Learning\\Final_Project\\DML\\sr\\datasets\\21586188.zip")
DATASET_ROOT = os.path.normpath("C:\\Users\\chand\\Intro_To_Visual_Learning\\Final_Project\\DML\\sr\\datasets\\FigShare")
SCALE_FACTORS = [2, 3, 4]
CATEGORIES = ["BSD100", "Set5", "Set14", "Urban100"]

CHECKPOINT_DIRS = {
    2: "C:\\Users\\chand\\Downloads\\LightSR\\LightSR\\dif2k_checkpoints\\checkpoints_2x",
    3: "C:\\Users\\chand\\Downloads\\LightSR\\LightSR\\dif2k_checkpoints\\checkpoints_3x",
    4: "C:\\Users\\chand\\Downloads\\LightSR\\LightSR\\dif2k_checkpoints\\checkpoints_4x"
}

CHECKPOINT_LOG_DIRS = {
    2: "C:\\Users\\chand\\Downloads\\LightSR\\LightSR\\dif2k_checkpoints\\logs_2x",
    3: "C:\\Users\\chand\\Downloads\\LightSR\\LightSR\\dif2k_checkpoints\\logs_3x",
    4: "C:\\Users\\chand\\Downloads\\LightSR\\LightSR\\dif2k_checkpoints\\logs_4x"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1

def parse_train_log(filepath):
    epochs, losses = [], []
    with open(filepath, 'r') as f:
        for line in f:
            if "Epoch" in line and "Loss" in line:
                parts = line.strip().split(',')
                epoch = int(parts[0].split()[1])
                loss = float(parts[1].split(':')[1])
                epochs.append(epoch)
                losses.append(loss)
    return epochs, losses

def parse_val_log(filepath):
    epochs, losses, psnrs, ssims = [], [], [], []
    with open(filepath, 'r') as f:
        for line in f:
            if "Epoch" in line and "Loss" in line:
                parts = line.strip().split(',')
                epoch = int(parts[0].split()[1])
                loss = float(parts[1].split(':')[1])
                psnr = float(parts[2].split(':')[1])
                ssim = float(parts[3].split(':')[1])
                epochs.append(epoch)
                losses.append(loss)
                psnrs.append(psnr)
                ssims.append(ssim)
    return epochs, losses, psnrs, ssims

def plot_log_metrics(scale, ckpt_dir):
    train_log = os.path.join(ckpt_dir, "train_log.txt")
    val_log = os.path.join(ckpt_dir, "val_log.txt")

    has_train, has_val = os.path.exists(train_log), os.path.exists(val_log)

    if has_train:
        train_epochs, train_losses = parse_train_log(train_log)
    else:
        train_epochs, train_losses = [], []

    if has_val:
        val_epochs, val_losses, val_psnrs, val_ssims = parse_val_log(val_log)
    else:
        val_epochs, val_losses, val_psnrs, val_ssims = [], [], [], []

    # Plot training and validation loss on the same figure
    if train_losses or val_losses:
        plt.figure()
        if train_losses:
            plt.plot(train_epochs, train_losses, marker='o', label='Train Loss')
        if val_losses:
            plt.plot(val_epochs, val_losses, marker='x', label='Val Loss')
        plt.title(f"Training & Validation Loss vs Epoch (x{scale})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"loss_curve_x{scale}.png")
        plt.close()

    # Plot PSNR and SSIM separately
    if val_epochs:
        plt.figure()
        plt.plot(val_epochs, val_psnrs, marker='o', color='green')
        plt.title(f"Validation PSNR vs Epoch (x{scale})")
        plt.xlabel("Epoch")
        plt.ylabel("PSNR (dB)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"val_psnr_x{scale}.png")
        plt.close()

        plt.figure()
        plt.plot(val_epochs, val_ssims, marker='o', color='blue')
        plt.title(f"Validation SSIM vs Epoch (x{scale})")
        plt.xlabel("Epoch")
        plt.ylabel("SSIM")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"val_ssim_x{scale}.png")
        plt.close()



# ---------- EXTRACT DATASET ----------
def extract_dataset():
    os.makedirs(DATASET_ROOT, exist_ok=True)
    nested_folders_expected = all(
        os.path.isdir(os.path.join(DATASET_ROOT, cat, f"image_SRF_{s}"))
        for s in SCALE_FACTORS for cat in CATEGORIES
    )
    if not nested_folders_expected:
        print("Extracting dataset...")
        zip_path = os.path.abspath(DATASET_ZIP)
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Expected ZIP file not found: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATASET_ROOT)
        fix_windows_permissions(DATASET_ROOT)

        # Extract nested ZIPs inside FigShare
        for cat in CATEGORIES:
            nested_zip = os.path.join(DATASET_ROOT, f"{cat}.zip")
            if os.path.exists(nested_zip):
                cat_dir = os.path.join(DATASET_ROOT, cat)
                with zipfile.ZipFile(nested_zip, 'r') as inner_zip:
                    inner_zip.extractall(cat_dir)
                fix_windows_permissions(cat_dir)
                os.remove(nested_zip)

                # Handle extra nesting (e.g., FigShare/BSD100/BSD100/image_SRF_2)
                nested_root = os.path.join(cat_dir, cat)
                if os.path.isdir(nested_root):
                    for item in os.listdir(nested_root):
                        os.replace(os.path.join(nested_root, item), os.path.join(cat_dir, item))
                    os.rmdir(nested_root)
        print("Extraction completed.")

# ---------- DATASET CLASS ----------
class MixedFlatDirDataset(Dataset):
    def __init__(self, folder, scale):
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Data folder not found: {folder}")

        self.lr_paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(f"SRF_{scale}_LR.png")
        ])
        self.hr_paths = [p.replace("LR.png", "HR.png") for p in self.lr_paths]
        assert all(os.path.exists(p) for p in self.hr_paths), "Missing HR image"

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        lr = np.array(Image.open(self.lr_paths[idx])).astype(np.float32) / 255.
        hr = np.array(Image.open(self.hr_paths[idx])).astype(np.float32) / 255.
        if lr.ndim == 2:
            lr = np.stack([lr]*3, axis=-1)
        if hr.ndim == 2:
            hr = np.stack([hr]*3, axis=-1)
        lr = torch.from_numpy(lr).permute(2, 0, 1)
        hr = torch.from_numpy(hr).permute(2, 0, 1)
        return lr, hr

# ---------- EVALUATION ----------
def evaluate_model(model, loader):
    psnr_sum = ssim_sum = count = 0
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            sr = model(lr).clamp(0, 1)
            for i in range(sr.size(0)):
                psnr_sum += calc_psnr(sr[i], hr[i])
                ssim_sum += calc_ssim(sr[i], hr[i])
                count += 1
    return psnr_sum / count, ssim_sum / count

# ---------- MAIN ----------
def main():
    extract_dataset()
    results = {s: {cat: [] for cat in CATEGORIES} for s in SCALE_FACTORS}


    for scale in SCALE_FACTORS:
        ckpt_dir = CHECKPOINT_DIRS.get(scale)
        if not ckpt_dir or not os.path.exists(ckpt_dir):
            print(f"No checkpoint directory for scale x{scale}")
            continue

        ckpt_paths = sorted(
            glob(os.path.join(ckpt_dir, "*.pt")),
            key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))) or '0')
        )

        for ckpt_idx, ckpt_path in enumerate(ckpt_paths):
            print(f"\nEvaluating Scale x{scale} Model: {os.path.basename(ckpt_path)}")
            model = SRConvnet(scale=scale, num_kernels=8, num_acb=4, dimension=64, num_heads=4).to(DEVICE)
            state_dict = torch.load(ckpt_path, map_location=DEVICE).get("model_state_dict", {})
            if any(k.startswith("module.") for k in state_dict):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            model.eval()

            for category in CATEGORIES:
                data_dir = os.path.normpath(os.path.join(DATASET_ROOT, category, f"image_SRF_{scale}"))
                if not os.path.exists(data_dir):
                    print(f"[Skip] Expected dataset folder not found: {data_dir}")
                    continue
                dataset = MixedFlatDirDataset(data_dir, scale)
                loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
                psnr, ssim = evaluate_model(model, loader)
                results[scale][category].append((psnr, ssim))
                print(f"[{category}] PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")

    # ---------- Summary Table ----------
    headers = ["Model"]
    psnr_headers, ssim_headers = [], []

    for cat in CATEGORIES:
        psnr_headers += [f"{cat} Max PSNR", f"{cat} Final PSNR"]
    for cat in CATEGORIES:
        ssim_headers += [f"{cat} Max SSIM", f"{cat} Final SSIM"]

    headers += psnr_headers + ssim_headers
    data_rows = []

    for scale in SCALE_FACTORS:
        row = [f"x{scale}"]
        psnr_vals, ssim_vals = [], []
        for cat in CATEGORIES:
            scores = results[scale][cat]
            if not scores:
                psnr_vals += ["N/A", "N/A"]
                ssim_vals += ["N/A", "N/A"]
                continue
            psnrs = [psnr for psnr, _ in scores]
            ssims = [ssim for _, ssim in scores]
            max_idx = int(np.argmax(psnrs))
            final_idx = -1
            psnr_vals += [f"{psnrs[max_idx]:.2f}", f"{psnrs[final_idx]:.2f}"]
            ssim_vals += [f"{ssims[max_idx]:.4f}", f"{ssims[final_idx]:.4f}"]
        row += psnr_vals + ssim_vals
        data_rows.append(row)

    # Print aligned table
    col_widths = [max(len(h), 12) for h in headers]
    output_file = "summary_results.txt"
    with open(output_file, "w") as f:
        def write_and_print(line=""):
            print(line)
            f.write(line + "\n")

        write_and_print("\nSummary Table (Grouped PSNRs then SSIMs, Models as rows):")
        header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        write_and_print(header_row)
        write_and_print("-" * len(header_row))
        for row in data_rows:
            write_and_print(" | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))

        write_and_print("\nCSV Format (for Excel import):")
        write_and_print(",".join(headers))
        for row in data_rows:
            write_and_print(",".join(row))

    print(f"\nSummary also saved to {output_file}")

    # Plot
    for scale in SCALE_FACTORS:
        if scale not in results or not results[scale][CATEGORIES[0]]:
            continue

        # Extract epoch numbers from filenames like model_x2_10.pt
        ckpt_dir = CHECKPOINT_DIRS.get(scale)
        ckpt_paths = sorted(
            glob(os.path.join(ckpt_dir, f"model_x{scale}_*.pt")),
            key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0])
        )
        epoch_nums = [
            int(os.path.basename(p).split("_")[-1].split(".")[0])
            for p in ckpt_paths
        ]

        plt.figure(figsize=(10, 6))
        for cat in CATEGORIES:
            psnrs = [x[0] for x in results[scale][cat]]
            if len(psnrs) != len(epoch_nums):
                print(f"Warning: Epoch count mismatch for x{scale} {cat}")
                continue
            plt.plot(epoch_nums, psnrs, marker='o', label=cat)
        plt.title(f"Average PSNR vs Epoch for Scale X{scale}")
        plt.xlabel("Epoch")
        plt.ylabel("PSNR (dB)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"psnr_plot_X{scale}.png")
        plt.close()



if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()