import os
import urllib.request
import zipfile
import argparse
import urllib.request
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, blocks=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_and_extract(url, download_path, extract_path):
    if not os.path.exists(download_path):
        os.makedirs(download_path, exist_ok=True)
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)

    filename = url.split('/')[-1]
    file_path = os.path.join(download_path, filename)

    if not os.path.exists(file_path):
        print(f"Downloading {filename}...")
        download_url(url, file_path)
        print("Download complete.")
    else:
        print(f"{filename} already downloaded.")

    # Extract if not extracted
    if not os.listdir(extract_path):
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Extraction complete.")
    else:
        print(f"{filename} already extracted.")

def download_coco_stuff(output_dir):
    download_dir = os.path.join(output_dir, 'downloads')
    extract_dir = os.path.join(output_dir, 'coco')

    urls = {
        "train2017": "http://images.cocodataset.org/zips/train2017.zip",
        "val2017": "http://images.cocodataset.org/zips/val2017.zip",
        "cocostuff2017": "https://github.com/nightrome/cocostuff/releases/download/v1.1/cocostuff2017.zip"
    }

    for key, url in urls.items():
        print(f"\nProcessing {key}...")
        download_path = download_dir
        extract_path = os.path.join(extract_dir, key)
        download_and_extract(url, download_path, extract_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download COCO Stuff dataset")
    parser.add_argument('--output_dir', type=str, required=False, default='/mntdata/main/light_sr/sr/datasets/coco',
                        help='Output directory to store the datasets')

    args = parser.parse_args()
    download_coco_stuff(args.output_dir)
