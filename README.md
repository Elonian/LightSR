
# Dataset Folder Structure

This repository contains the following datasets organized under the `dataset` folder:

- **BSD100**  
  A widely used benchmark dataset for image super-resolution consisting of 100 natural images.

- **Set4**  
  A small set of 4 images commonly used for evaluating image restoration and super-resolution methods.

- **Set15**  
  A classic dataset containing 15 standard test images for super-resolution and image processing tasks.

- **Urban100**  
  A dataset focused on urban scenes with 100 images, used primarily for evaluating image super-resolution models on structured, repetitive textures.

- **DF2Kdata**  
  A large combined dataset that includes images from DIV2K and Flickr2K, used as a common training dataset for super-resolution tasks.

## Folder Hierarchy


```plaintext
dataset/
├── BSD100/
├── Set4/
├── Set15/
├── Urban100/
└── DF2Kdata/
    └── versions/
        └── 1/
            ├── df2k_train_hr/
            │   ├── image1.png
            │   ├── image2.png
            │   └── ...
            └── df2k_train_lr_bicubic/
                ├── x2/
                │   ├── image1.png
                │   ├── image2.png
                │   └── ...
                └── x3/
                    ├── image1.png
                    ├── image2.png
                    └── ...