import os
import random
import numpy as np
import cv2  
import torch
import torch.utils.data as data
import skimage.color as sc
from utils.util import ndarray2tensor


def read_image_cv2(path, colors=1):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if colors == 1:
        img = sc.rgb2ycbcr(img)[:, :, 0:1]
    return img


class CustomDataset(data.Dataset):
    def __init__(self,
                 HR_folder, LR_folder,
                 cache_folder=None,
                 scale=2, colors=1, patch_size=96,
                 train=True, augment=True, repeat=1,
                 max_samples=None):

        super(CustomDataset, self).__init__()
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
                self.hr_filenames.append(hr)
                self.lr_filenames.append(lr)
        else:
            tags = sorted(os.listdir(self.HR_folder))
            for tag in tags:
                hr = os.path.join(self.HR_folder, tag)
                lr = os.path.join(self.LR_folder, f'X{self.scale}', tag.replace('.png', f'x{self.scale}.png'))
                self.hr_filenames.append(hr)
                self.lr_filenames.append(lr)

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
        idx = idx % self.nums_samples
        hr = self.hr_images[idx]
        lr = self.lr_images[idx]

        if self.train:
            lr, hr = self._crop_patch(lr, hr)
        else:
            lr_h, lr_w, _ = lr.shape
            hr = hr[0:lr_h * self.scale, 0:lr_w * self.scale, :]

        lr, hr = lr.copy(), hr.copy() 
        lr = torch.from_numpy(lr).permute(2, 0, 1).float() / 255.
        hr = torch.from_numpy(hr).permute(2, 0, 1).float() / 255.

        if self.train:
            return lr, hr
        else:
            return lr, hr, self.hr_filenames[idx]

    def _crop_patch(self, lr, hr):
        lr_h, lr_w, _ = lr.shape
        hp = self.patch_size
        lp = self.patch_size // self.scale
        lx, ly = random.randrange(0, lr_w - lp + 1), random.randrange(0, lr_h - lp + 1)
        hx, hy = lx * self.scale, ly * self.scale
        lr_patch, hr_patch = lr[ly:ly + lp, lx:lx + lp, :], hr[hy:hy + hp, hx:hx + hp, :]

        if self.augment:
            if random.random() > 0.5:
                lr_patch, hr_patch = lr_patch[:, ::-1, :], hr_patch[:, ::-1, :]
            if random.random() > 0.5:
                lr_patch, hr_patch = lr_patch[::-1, :, :], hr_patch[::-1, :, :]
            if random.random() > 0.5:
                lr_patch, hr_patch = lr_patch.transpose(1, 0, 2), hr_patch.transpose(1, 0, 2)

        return lr_patch, hr_patch


import os
import numpy as np
import imageio.v2 as imageio
from torch.utils.data import DataLoader

def to_uint8(img):
    # Clip and scale float image assumed to be [0,1]
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img

if __name__ == "__main__":
    HR_folder = '/mntdata/main/light_sr/sr/datasets/df2kdata/versions/1/DF2K_train_HR'
    LR_folder = '/mntdata/main/light_sr/sr/datasets/df2kdata/versions/1/DF2K_train_LR_bicubic'

    output_dir = '/mntdata/main/light_sr/sr/dataset_utils'
    os.makedirs(output_dir, exist_ok=True)

    # Use max_samples=10 for quick test
    dataset = CustomDataset(
        HR_folder=HR_folder,
        LR_folder=LR_folder,
        cache_folder='/mntdata/main/light_sr/sr/cache',  # new cache folder
        scale=2,
        colors=3,
        patch_size=96,
        train=True,
        augment=True,
        repeat=1
    )

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for i, (lr_tensor, hr_tensor) in enumerate(dataloader):
        lr_img = lr_tensor.to(device, non_blocking=True)
        hr_img = hr_tensor.to(device, non_blocking=True)
        print(lr_img.shape, hr_img.shape)  

        # Remove batch dimension only
        lr_img = lr_img[0]  # shape: (C, H, W)
        hr_img = hr_img[0]

        # Convert (C, H, W) -> (H, W, C)
        lr_img = lr_img.permute(1, 2, 0)  # (H, W, C)
        hr_img = hr_img.permute(1, 2, 0)

        # Move to CPU and convert to NumPy
        lr_img_np = lr_img.cpu().numpy()
        hr_img_np = hr_img.cpu().numpy()

        # Save images
        lr_img_uint8 = to_uint8(lr_img_np)
        hr_img_uint8 = to_uint8(hr_img_np)

        imageio.imwrite(os.path.join(output_dir, f'lr_sample_{i}.png'), lr_img_uint8)
        imageio.imwrite(os.path.join(output_dir, f'hr_sample_{i}.png'), hr_img_uint8)

        print(f"Saved LR and HR images to {output_dir}")

        if i == 0:
            break
