import kagglehub

# Download latest version

path = kagglehub.dataset_download("anvu1204/df2kdata",
                                  force_download=True)

print("Path to dataset files:", path)