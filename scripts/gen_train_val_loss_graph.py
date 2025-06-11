import os
import re
import matplotlib.pyplot as plt

def parse_train_log(train_path):
    epochs = []
    train_losses = []
    with open(train_path, 'r') as f:
        for line in f:
            match = re.search(r"Epoch (\d+), Loss: ([\d.]+)", line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                epochs.append(epoch)
                train_losses.append(loss)
    return epochs, train_losses

def parse_val_log(val_path):
    epochs = []
    val_losses = []
    with open(val_path, 'r') as f:
        for line in f:
            match = re.search(r"Epoch (\d+), Loss: ([\d.]+),", line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                epochs.append(epoch)
                val_losses.append(loss)
    return epochs, val_losses

def plot_loss(train_epochs, train_losses, val_epochs, val_losses, out_path="loss_plot.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_losses, label="Train Loss", marker='o')
    plt.plot(val_epochs, val_losses, label="Validation Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs (X4)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    folder = "C:\\Users\\chand\\Downloads\\LightSR\\LightSR\\dif2k_checkpoints\\logs_4x"  # replace with your actual path
    train_log = os.path.join(folder, "train_log.txt")
    val_log = os.path.join(folder, "val_log.txt")

    train_epochs, train_losses = parse_train_log(train_log)
    val_epochs, val_losses = parse_val_log(val_log)

    plot_loss(train_epochs, train_losses, val_epochs, val_losses)
