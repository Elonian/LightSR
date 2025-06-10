import kagglehub
import os

os.environ["KAGGLEHUB_DIR"] = "/mntdata/main/light_sr/sr/datasets"

path = kagglehub.dataset_download("yashchoudhary/realsr-v3",
                                  force_download=True)

print("Path to dataset files:", path)