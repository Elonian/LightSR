
# Datasets

This repository contains the following datasets organized under the `datasets` folder:

- **BSD100**  
  A widely used benchmark dataset for image super-resolution consisting of 100 natural images.

- **Set4**  
  A small set of 4 images commonly used for evaluating image restoration and super-resolution methods.

- **Set15**  
  A classic dataset containing 15 standard test images for super-resolution and image processing tasks.

- **Urban100**  
  A dataset focused on urban scenes with 100 images, used primarily for evaluating image super-resolution models on structured, repetitive textures.

- **Manga109**  
  A dataset of 109 Japanese comic books used for super-resolution and image restoration, particularly useful for evaluating performance on line-art and text-heavy images.

- **RealSR**  
  A real-world image super-resolution dataset captured with different DSLR cameras (e.g., Nikon, Canon), useful for testing model robustness on authentic low-quality images.

- **DF2Kdata**  
  A large combined dataset that includes images from DIV2K and Flickr2K, used as a common training dataset for super-resolution tasks.

You can find all datasets used in this project in the following UCSD OneDrive folder:

🔗 [LightSR Datasets](https://ucsdcloud-my.sharepoint.com/:f:/r/personal/vmoparthi_ucsd_edu/Documents/LightSR/datasets?csf=1&web=1&e=XbDd5O)

> **Note:** Use your UCSD email (`@ucsd.edu`) to access the shared folder.

## Folder Hierarchy


```plaintext
datasets/
├── BSD100/
├── Set4/
├── Set15/
├── Urban100/
├── Manga109/
├── RealSR/
│     └── Canon/
│          ├── Canon_TRAIN_HR/
│          │   ├── 0001.png
│          │   ├── 0002.png
│          │   └── ...
│          └── Canon_TRAIN_LR/
│              ├── X2/
│              │   ├── 0001x2.png
│              │   ├── 0002x2.png
│              │   └── ...
│              └── X3/
│              │   ├── 0001x3.png
│              │   ├── 0002x3.png
│              │   └── ...
│              └── X4/
│                  ├── 0001x4.png
│                  ├── 0002x4.png
│                  └── ...
└── DF2Kdata/
    └── versions/
        └── 1/
            ├── DF2K_train_HR/
            │   ├── 0001.png
            │   ├── 0002.png
            │   └── ...
            └── DF2K_train_LR_bicubic/
                ├── X2/
                │   ├── 0001x2.png
                │   ├── 0002x2.png
                │   └── ...
                └── X3/
                │   ├── 0001x3.png
                │   ├── 0002x3.png
                │   └── ...
                └── X4/
                    ├── 0001x4.png
                    ├── 0002x4.png
                    └── ...
```

# Checkpoints and Logs

All training checkpoints, logs, and experiment outputs are stored in the following UCSD OneDrive folder:

🔗 [LightSR Checkpoints & Logs](https://ucsdcloud-my.sharepoint.com/:f:/r/personal/vmoparthi_ucsd_edu/Documents/LightSR?csf=1&web=1&e=fR56pw)

> **Note:** Use your UCSD email (`@ucsd.edu`) to access the shared folder.
