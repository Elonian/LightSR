import os
import shutil

def extract_and_rename_lr_images_custom_suffix(src_dir, dst_dir, scale=None):
    os.makedirs(dst_dir, exist_ok=True)

    lr_images = sorted([f for f in os.listdir(src_dir) if f.endswith(f".png")])
    
    for idx, filename in enumerate(lr_images, start=1):
        src_path = os.path.join(src_dir, filename)
        dst_filename = f"{idx:04d}x{scale}.png"
        # dst_filename = f"{idx:04d}.png"
        dst_path = os.path.join(dst_dir, dst_filename)

        shutil.copy2(src_path, dst_path)
        print(f"Copied: {filename} → {dst_filename}")
        ## remove the original file if needed
        os.remove(src_path)

    print(f"{len(lr_images)} files copied and renamed to {dst_dir}")

if __name__ == "__main__":
    scale = 4
    src_dir = f"/mntdata/main/light_sr/sr/datasets/realsr/Canon/Canon_VALID_LR/X{scale}"
    dst_dir = f"/mntdata/main/light_sr/sr/datasets/realsr/Canon/Canon_VALID_LR/X{scale}"

    extract_and_rename_lr_images_custom_suffix(src_dir, dst_dir, scale)
    print("Renaming completed.")