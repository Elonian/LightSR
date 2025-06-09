import kagglehub

# Download latest version

path = kagglehub.dataset_download("anvu1204/df2kdata",
                                  local_dir="/mntdata/main/light_sr/sr/datasets",
                                  force_download=True)

print("Path to dataset files:", path)