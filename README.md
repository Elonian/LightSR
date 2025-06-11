# LightSR: Lightweight and Efficient Image Super-Resolution

LightSR is a compact and efficient Single Image Super-Resolution (SISR) framework designed for deployment on resource-constrained devices such as smartphones, drones, and embedded systems. Built on SRConvNet, a Transformer-style convolutional architecture with Fourier-modulated attention and dynamic convolution, LightSR strikes a balance between reconstruction quality and model compactness.

This project was developed as part of **ECE 285** at UC San Diego.

# Project Presentation (285 Evaluation)

We present **LightSR** as part of our final project for ECE 285 at UC San Diego. The presentation highlights the motivation, architecture, training setup, evaluation results, and key takeaways from our work on lightweight image super-resolution.

[Watch the Presentation Recording (UCSD SharePoint)](https://ucsdcloud-my.sharepoint.com/:v:/g/personal/vmoparthi_ucsd_edu/EatwsAhNSrNIouYCAYMS82YBXEfm38591dNp7CduyE0CAA?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=44oVJK)

> **Note:** Use your UCSD email (`@ucsd.edu`) to access the recording.


### Presentation Slides

You can view the slides here:  
**[LightSR Project Presentation (PDF/Google Slides)](https://ucsdcloud-my.sharepoint.com/:p:/g/personal/cviraktamath_ucsd_edu/EYLPpfE6z3tEvt7y8OyE4_UB1uw11Mo3LA_QSRXGKqIZsg?e=k2h8Wv)**

> **Note:** Use your UCSD email (`@ucsd.edu`) to access the slide deck.

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

[LightSR Datasets](https://ucsdcloud-my.sharepoint.com/:f:/g/personal/vmoparthi_ucsd_edu/EmXlMFQoLTFJjgIVsoyPwVUBrmTBXueWPH2L-R219uxVQA?e=PGFl2g)

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

[LightSR Checkpoints & Logs](https://ucsdcloud-my.sharepoint.com/:f:/r/personal/vmoparthi_ucsd_edu/Documents/LightSR?csf=1&web=1&e=fR56pw)

# 

# Evaluation and Results

### 1. `evaluate_sr_models.py` – Super-Resolution Evaluation Script

Automates evaluation of pre-trained super-resolution models on standard datasets: **Set5**, **Set14**, **BSD100**, and **Urban100**, across upscaling factors **×2**, **×3**, and **×4**.

**Functionality:**
- Computes PSNR/SSIM for each checkpoint and dataset
- Produces CSV-formatted summaries and aligned tables
- Automatically handles nested dataset ZIP extraction and folder restructuring

▶ **Usage:**
```bash
python evaluate_sr_models.py
```

🔧 **Inputs to Update:**
- `CHECKPOINT_DIRS` – Paths to model weights for 2x, 3x, 4x models
- `DATASET_ZIP` – Path to dataset ZIP file (`21586188.zip` from FigShare)

---

### 2. `gen_psnr_ssim_graph.py` – Performance Graph Generation Script

Generates **PSNR and SSIM vs Epoch** graphs for each dataset and scale using logs or summary files created by `evaluate_sr_models.py`.

▶ **Usage:**
```bash
python gen_psnr_ssim_graph.py
```

**Inputs to Update:**
- `summary_results.txt` – Output log from `evaluate_sr_models.py`

---

### 3. `run_predictions.py` – SRConvNet Inference Script

Performs single-image super-resolution (SISR) inference using pre-trained SRConvNet models. For each input image:
- Generates **model-predicted SR output**
- Produces **baseline bilinear upsampled image** for comparison

▶ **Usage:**
```bash
python run_predictions.py
```

 **Inputs to Update:**
- `CHECKPOINT_IMAGE_PAIRS` – List of tuples with paths to:
  - Model checkpoint (2x, 3x, 4x)
  - Corresponding low-resolution input image

---

### 4. `gen_train_val_loss_graph.py` – Train/Val Graph Generation Script

Generates training and validation loss plots from `.txt` logs generated during model training.

▶**Usage:**
```bash
python gen_train_val_loss_graph.py
```

🔧 **Inputs to Update:**
- `folder` – Folder containing:
  - `train_log.txt`
  - `val_log.txt`



> **Note:** Use your UCSD email (`@ucsd.edu`) to access the shared folder.
