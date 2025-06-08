from huggingface_hub import snapshot_download

local_dir = "/mntdata/main/light_sr/sr/datasets"  # your desired path

snapshot_download(
    repo_id="tacofoundation/SEN2NAIPv2",
    repo_type="dataset",
    local_dir=local_dir
)

print("Downloaded dataset to:", local_dir)