import os
import random
import numpy as np
import cv2  
import torch
import torch.utils.data as data
import skimage.color as sc
import imageio.v2 as imageio
from utils.util import ndarray2tensor


def read_image_cv2(path, colors=1):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if colors == 1:
        img = img.astype(np.float32)
        img = sc.rgb2ycbcr(img)[:, :, 0:1]
    return img


class CustomDataset(data.Dataset):
    def __init__(self,
                 HR_folder, LR_folder,
                 cache_folder=None,
                 scale=2, colors=1, patch_size=96,
                 train=True, augment=True, repeat=1,
                 max_samples=200):

        super(CustomDataset, self).__init__()
        print("Initializing CustomDataset...")
        self.HR_folder = HR_folder
        self.LR_folder = LR_folder
        self.cache_folder = cache_folder
        self.scale = scale
        self.colors = colors
        self.patch_size = patch_size
        self.train = train
        self.augment = augment
        self.repeat = repeat if train else 1
        self.max_samples = max_samples

        self.img_postfix = '.png'
        self.hr_filenames = []
        self.lr_filenames = []
        self.hr_npy_names = []
        self.lr_npy_names = []

        self._gather_filenames()

        if self.max_samples is not None:
            self.hr_filenames = self.hr_filenames[:self.max_samples]
            self.lr_filenames = self.lr_filenames[:self.max_samples]

        if self.cache_folder is None:
            print(f"Loading {len(self.hr_filenames)} image pairs from PNG files (using OpenCV)...")
            self.hr_images = []
            self.lr_images = []
            self._load_images()
        else:
            print(f"Using cache folder {self.cache_folder} for npy data")
            self._prepare_cache()
            self._load_cache()

        self.nums_samples = len(self.hr_filenames)

    def _gather_filenames(self):
        if self.train:
            start_idx, end_idx = 1, 801
            for i in range(start_idx, end_idx):
                idx = str(i).zfill(4)
                hr = os.path.join(self.HR_folder, idx + self.img_postfix)
                lr = os.path.join(self.LR_folder, f'X{self.scale}', f'{idx}x{self.scale}{self.img_postfix}')
                if not os.path.exists(hr) or not os.path.exists(lr):
                    print(f"Warning: Missing file for index {idx}. HR: {hr}, LR: {lr}")
                    continue
                try:
                    lr_img = read_image_cv2(lr, self.colors)
                    if lr_img.shape[0] < self.patch_size // self.scale or lr_img.shape[1] < self.patch_size // self.scale:
                        print(f"Skipping LR image too small: {lr_img.shape} at {lr}")
                        continue
                except Exception as e:
                    print(f"Failed to read {lr}: {e}")
                    continue

                self.hr_filenames.append(hr)
                self.lr_filenames.append(lr)
            print(f"Found {len(self.hr_filenames)} training image pairs.")
        else:
            tags = sorted(os.listdir(self.HR_folder))
            for tag in tags:
                hr = os.path.join(self.HR_folder, tag)
                #lr = os.path.join(self.LR_folder, f'X{self.scale}', tag.replace('.png', f'x{self.scale}.png'))
                lr = os.path.join(self.LR_folder, tag.replace('.png', f'x{self.scale}.png'))

                self.hr_filenames.append(hr)
                self.lr_filenames.append(lr)
            print(f"Found {len(self.hr_filenames)} validation image pairs.")

    def _prepare_cache(self):
        hr_cache_dir = os.path.join(self.cache_folder, 'hr', 'ycbcr' if self.colors == 1 else 'rgb')
        lr_cache_dir = os.path.join(self.cache_folder, f'lr_x{self.scale}', 'ycbcr' if self.colors == 1 else 'rgb')
        os.makedirs(hr_cache_dir, exist_ok=True)
        os.makedirs(lr_cache_dir, exist_ok=True)

        print("Preparing cache: Converting HR images to npy if needed...")
        self.hr_npy_names = []
        for i, hr_path in enumerate(self.hr_filenames):
            npy_name = os.path.basename(hr_path).replace('.png', '.npy')
            npy_path = os.path.join(hr_cache_dir, npy_name)
            self.hr_npy_names.append(npy_path)
            if not os.path.exists(npy_path):
                hr_image = read_image_cv2(hr_path, self.colors)
                np.save(npy_path, hr_image)
            if i % 50 == 0 or i == len(self.hr_filenames) - 1:
                print(f"  HR images cached: {i+1}/{len(self.hr_filenames)}")

        print("Preparing cache: Converting LR images to npy if needed...")
        self.lr_npy_names = []
        for i, lr_path in enumerate(self.lr_filenames):
            npy_name = os.path.basename(lr_path).replace('.png', '.npy')
            npy_path = os.path.join(lr_cache_dir, npy_name)
            self.lr_npy_names.append(npy_path)
            if not os.path.exists(npy_path):
                lr_image = read_image_cv2(lr_path, self.colors)
                np.save(npy_path, lr_image)
            if i % 50 == 0 or i == len(self.lr_filenames) - 1:
                print(f"  LR images cached: {i+1}/{len(self.lr_filenames)}")

    def _load_cache(self):
        print(f"Loading {len(self.hr_npy_names)} HR npy images into RAM...")
        self.hr_images = [np.load(p) for p in self.hr_npy_names]
        print(f"Loading {len(self.lr_npy_names)} LR npy images into RAM...")
        self.lr_images = [np.load(p) for p in self.lr_npy_names]

    def _load_images(self):
        self.hr_images = []
        self.lr_images = []
        for i, (hr_path, lr_path) in enumerate(zip(self.hr_filenames, self.lr_filenames)):
            hr = read_image_cv2(hr_path, self.colors)
            lr = read_image_cv2(lr_path, self.colors)
            self.hr_images.append(hr)
            self.lr_images.append(lr)
            if i % 20 == 0 or i == len(self.hr_filenames) - 1:
                print(f"Loaded {i+1}/{len(self.hr_filenames)} images")

    def __len__(self):
        return self.nums_samples * self.repeat

    def __getitem__(self, idx):
        if self.cache_folder is not None and hasattr(self, 'hr_npy_names') and hasattr(self, 'lr_npy_names'):
            hr = np.load(self.hr_npy_names[idx])
            lr = np.load(self.lr_npy_names[idx])
        else:
            hr = read_image_cv2(self.hr_filenames[idx], self.colors)
            lr = read_image_cv2(self.lr_filenames[idx], self.colors)

        if lr.ndim == 2:
            lr = lr[:, :, None]
        if hr.ndim == 2:
            hr = hr[:, :, None]

        if self.train:
            lr_patch, hr_patch = self._crop_patch(lr, hr)
            if lr_patch is None or hr_patch is None:
                # Skip this sample by recursively getting another one
                # Here, you can randomly sample a new idx or handle differently
                new_idx = random.randint(0, self.nums_samples - 1)
                return self.__getitem__(new_idx)

            lr = lr_patch
            hr = hr_patch
        else:
            lr_h, lr_w, _ = lr.shape
            hr = hr[0:lr_h * self.scale, 0:lr_w * self.scale, :]

        lr = torch.from_numpy(lr.copy()).permute(2, 0, 1).float() / 255.
        hr = torch.from_numpy(hr.copy()).permute(2, 0, 1).float() / 255.

        return (lr, hr) if self.train else (lr, hr, self.hr_filenames[idx])


    def _crop_patch(self, lr, hr):
        lr_h, lr_w, _ = lr.shape
        hp = self.patch_size
        lp = self.patch_size // self.scale

        if lr_h < lp or lr_w < lp:
            # Skip sample if LR image is too small
            return None, None

        # Random crop top-left coordinates
        lx = random.randint(0, lr_w - lp)
        ly = random.randint(0, lr_h - lp)
        hx, hy = lx * self.scale, ly * self.scale

        # Crop patches
        lr_patch = lr[ly:ly + lp, lx:lx + lp, :]
        hr_patch = hr[hy:hy + hp, hx:hx + hp, :]

        # If patch sizes are not correct (e.g., smaller than expected), skip
        if lr_patch.shape[0] != lp or lr_patch.shape[1] != lp:
            return None, None
        if hr_patch.shape[0] != hp or hr_patch.shape[1] != hp:
            return None, None

        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                lr_patch, hr_patch = lr_patch[:, ::-1, :], hr_patch[:, ::-1, :]
            if random.random() > 0.5:
                lr_patch, hr_patch = lr_patch[::-1, :, :], hr_patch[::-1, :, :]
            if random.random() > 0.5:
                lr_patch, hr_patch = lr_patch.transpose(1, 0, 2), hr_patch.transpose(1, 0, 2)

        return lr_patch, hr_patch


    def _pad_to_size(self, img, size):
        """Pad img with zeros to target (height, width)."""
        h, w, c = img.shape
        target_h, target_w = size
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
        return img



def to_uint8(img):
    # Clip and scale float image assumed to be [0,1]
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img

# if __name__ == "__main__":
#     HR_folder = '/mntdata/main/light_sr/sr/datasets/df2kdata/versions/1/DF2K_train_HR'
#     LR_folder = '/mntdata/main/light_sr/sr/datasets/df2kdata/versions/1/DF2K_train_LR_bicubic'

#     output_dir = '/mntdata/main/light_sr/sr/dataset_utils'
#     os.makedirs(output_dir, exist_ok=True)

#     # Use max_samples=10 for quick test
#     dataset = CustomDataset(
#         HR_folder=HR_folder,
#         LR_folder=LR_folder,
#         cache_folder='/mntdata/main/light_sr/sr/cache',  # new cache folder
#         scale=2,
#         colors=3,
#         patch_size=96,
#         train=True,
#         augment=True,
#         repeat=1
#     )

#     dataloader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=8)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     for i, (lr_tensor, hr_tensor) in enumerate(dataloader):
#         lr_img = lr_tensor.to(device, non_blocking=True)
#         hr_img = hr_tensor.to(device, non_blocking=True)
#         print(lr_img.shape, hr_img.shape)  

#         # Remove batch dimension only
#         lr_img = lr_img[0]  # shape: (C, H, W)
#         hr_img = hr_img[0]

#         # Convert (C, H, W) -> (H, W, C)
#         lr_img = lr_img.permute(1, 2, 0)  # (H, W, C)
#         hr_img = hr_img.permute(1, 2, 0)

#         # Move to CPU and convert to NumPy
#         lr_img_np = lr_img.cpu().numpy()
#         hr_img_np = hr_img.cpu().numpy()

#         # Save images
#         lr_img_uint8 = to_uint8(lr_img_np)
#         hr_img_uint8 = to_uint8(hr_img_np)

#         imageio.imwrite(os.path.join(output_dir, f'lr_sample_{i}.png'), lr_img_uint8)
#         imageio.imwrite(os.path.join(output_dir, f'hr_sample_{i}.png'), hr_img_uint8)

#         print(f"Saved LR and HR images to {output_dir}")

#         if i == 0:
#             break
