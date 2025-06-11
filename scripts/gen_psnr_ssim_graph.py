import re
import matplotlib.pyplot as plt
from collections import defaultdict

# Initialize storage
results = {
    'x2': defaultdict(list),
    'x3': defaultdict(list),
    'x4': defaultdict(list)
}
epochs = {
    'x2': [],
    'x3': [],
    'x4': []
}

# Read log file
with open('data.txt', 'r') as f:
    lines = f.readlines()

scale = None
epoch = None
pattern_epoch = re.compile(r'model_x\d_(\d+)\.pt')
pattern_dataset = re.compile(r'\[(.*?)\] PSNR: ([\d.]+), SSIM: ([\d.]+)')

for line in lines:
    if 'Evaluating Scale' in line:
        scale_match = re.search(r'Scale x(\d)', line)
        epoch_match = pattern_epoch.search(line)
        if scale_match and epoch_match:
            scale = f'x{scale_match.group(1)}'
            epoch = int(epoch_match.group(1))
            epochs[scale].append(epoch)
    elif '[' in line and 'PSNR' in line:
        match = pattern_dataset.search(line)
        if match and scale is not None:
            dataset, psnr, ssim = match.groups()
            results[scale][dataset].append((epoch, float(psnr), float(ssim)))

# Plotting function
def plot_metric(metric_index, ylabel, filename_prefix):
    for scale in results:
        plt.figure(figsize=(10, 6))
        for dataset in results[scale]:
            sorted_vals = sorted(results[scale][dataset])
            x = [e[0] for e in sorted_vals]
            y = [e[metric_index] for e in sorted_vals]
            plt.plot(x, y, marker='o', label=dataset)
        plt.title(f'{ylabel} over Epochs (Scale {scale})', fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{filename_prefix}_{scale}.png', dpi=300)
        plt.show()

# Plot PSNR (index 1 in tuple)
plot_metric(1, 'PSNR (dB)', 'psnr_plot')

# Plot SSIM (index 2 in tuple)
plot_metric(2, 'SSIM', 'ssim_plot')
