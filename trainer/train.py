import os
import sys
import math
import yaml
import time
import torch
import argparse
import platform
import imageio
from tqdm import tqdm
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchsummary import summary

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataset_utils.data_loader import CustomDataset
from models.model import SRConvnet
from utils.util import calc_psnr, calc_ssim


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_model(path, epoch, model, optimizer, scheduler, stats):
    state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    torch.save({
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'stat_dict': stats
    }, path)


def save_sr_images(sr_batch, epoch, save_dir):
    save_epoch_dir = os.path.join(save_dir, f"results_epoch_{epoch}")
    os.makedirs(save_epoch_dir, exist_ok=True)

    sr_batch = sr_batch.detach().cpu().float()
    for idx in range(sr_batch.size(0)):
        img = sr_batch[idx]
        img_np = img.permute(1, 2, 0).numpy()
        if img_np.shape[2] == 1:
            img_np = img_np[:, :, 0]
        img_np = (img_np * 255).clip(0, 255).astype('uint8')  # <--- fix here
        save_path = os.path.join(save_epoch_dir, f"sr_{idx}.png")
        imageio.imwrite(save_path, img_np)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    backend = "nccl" if platform.system() == "Linux" and torch.cuda.is_available() and dist.is_nccl_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def main_worker(rank, world_size, args):

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"main_worker start: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    if world_size > 1:
        setup(rank, world_size, local_rank)
        print("Setup done")
    else:
        print("Single process, skipping distributed setup")
    config = {}
    if args.config:
        config = load_config(args.config)
    config.update(vars(args))

    device = torch.device(f'cuda:{rank}')

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'results'), exist_ok=True)
    if world_size > 1:
        dist.barrier()

    train_dataset = CustomDataset(
        HR_folder=config['train_HR_folder'],
        LR_folder=config['train_LR_folder'],
        cache_folder=config.get('cache_folder_train', "/mntdata/main/light_sr/sr/cache/realsr/train"),
        scale=config.get('scale', 2),
        colors=config.get('channels', 3),
        patch_size=config.get('patch_size', 64),
        train=True,
        augment=True,
        repeat=1,
        max_samples=500
    )

    val_dataset = CustomDataset(
        HR_folder=config['val_HR_folder'],
        LR_folder=config['val_LR_folder'],
        cache_folder=config.get('cache_folder_val', "/mntdata/main/light_sr/sr/cache/realsr/val"),
        scale=config.get('scale', 2),
        colors=config.get('channels', 3),
        train=False,
        augment=False
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 64),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True
    )

    model = SRConvnet(
        scale=config.get('scale', 2),
        num_kernels=config.get('num_kernels', 8),
        num_acb=config.get('num_acb', 4),
        dimension=config.get('dimension', 64),
        num_heads=config.get('num_heads', 4)
    ).to(device)
    print(f"Device: {device}, Model: {model.__class__.__name__}")

    print("Model Architecture:")
    input_size = (config.get('channels', 3), 64, 64)

    summary(model, input_size=input_size, device='cuda')


    if config.get('fp') == 16:
        model = model.half()

    if config.get('pretrain') and rank == 0:
        checkpoint = torch.load(
            config.get('pretrain', '/mntdata/main/light_sr/sr/results/DF2K/4x/results/model_x4_200.pt'),
            map_location=device,
            weights_only=False
        )
        print(checkpoint.keys())
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained model from {config['pretrain']} for scale {config.get('scale', 2)}")
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    loss_func = getattr(nn, config.get('loss', 'L1Loss'))()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-4), eps=1e-8)
    scheduler = MultiStepLR(optimizer, milestones=config.get('decays', [30, 40]), gamma=config.get('gamma', 0.5))

    start_epoch = 1
    stat_dict = {'losses': [], 'val_losses': [], 'psnrs': [], 'ssims': []}

    epochs = config.get('epochs', 200)
    open(os.path.join(args.train_log_dir, '/mntdata/main/light_sr/sr/results/REALSR/4x/results/train_log.txt'), 'w').close()
    open(os.path.join(args.val_log_dir, '/mntdata/main/light_sr/sr/results/REALSR/4x/results/val_log.txt'), 'w').close()
    for epoch in tqdm(range(start_epoch, epochs + 1), desc="Training Epochs"):

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0

        for i, (lr, hr) in enumerate(train_loader):
            optimizer.zero_grad()
            lr, hr = lr.to(device), hr.to(device)
            if config.get('fp') == 16:
                lr, hr = lr.half(), hr.half()

            sr = model(lr)
            print(f"[Training] lr shape: {lr.shape}, hr shape: {hr.shape}, sr model shape: {sr.shape}")

            loss = loss_func(sr, hr)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if (i + 1) % config.get('log_every', 100) == 0 and rank == 0:
                avg_loss = epoch_loss / (i + 1)
                print(f"Epoch [{epoch}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {avg_loss:.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
        if rank == 0:
            stat_dict['losses'].append(avg_loss)
            with open(os.path.join(args.train_log_dir, '/mntdata/main/light_sr/sr/results/REALSR/4x/results/train_log.txt'), 'a') as f:
                f.write(f"Epoch {epoch}, Loss: {avg_loss:.4f}\n")


        if epoch % config.get('test_every', 5) == 0:
            model.eval()
            avg_psnr = 0
            avg_ssim = 0
            val_loss = 0
            sr_images_to_save = []

            with torch.no_grad():
                for idx, (lr, hr, _) in enumerate(val_loader):
                    lr, hr = lr.to(device), hr.to(device)
                    sr = model(lr).clamp(0, 255)
                    print(f"lr shape: {lr.shape}, hr shape: {hr.shape}, sr model shape: {sr.shape}")
                    psnr = calc_psnr(sr[0], hr[0])
                    ssim = calc_ssim(sr[0], hr[0])
                    
                    loss = loss_func(sr, hr)
                    val_loss += loss.item()
                    avg_psnr += psnr
                    avg_ssim += ssim
                    if rank == 0:
                        sr_images_to_save.append(sr)

            avg_psnr /= len(val_loader)
            avg_ssim /= len(val_loader)
            avg_val_loss = val_loss / len(val_loader)

            if rank == 0:
                print(f"[Validation] Epoch {epoch} | Loss: {avg_val_loss:.4f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")
                stat_dict['psnrs'].append(avg_psnr)
                stat_dict['val_losses'].append(avg_val_loss)
                stat_dict['ssims'].append(avg_ssim)
                with open(os.path.join(args.val_log_dir, '/mntdata/main/light_sr/sr/results/REALSR/4x/results/val_log.txt'), 'a') as f:
                    f.write(f"Epoch {epoch}, Loss: {avg_val_loss:.4f}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}\n")
                if sr_images_to_save:
                    epoch_results_dir = os.path.join(args.save_dir, 'results', f"epoch_{epoch}")
                    for idx, sr_img in enumerate(sr_images_to_save):
                        save_path = os.path.join(epoch_results_dir, f"epoch{epoch}_img{idx}.png")
                        save_sr_images(sr_img, epoch, save_path)
                save_path = os.path.join(args.save_dir, f"model_x{config['scale']}_{epoch}.pt")
                save_model(save_path, epoch, model, optimizer, scheduler, stat_dict)

    if rank == 0:
        print("Training complete.")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Super-Resolution Model with DDP")
    parser = argparse.ArgumentParser(description="Train Super-Resolution Model with DDP")

    parser.add_argument('--resume', type=str, default=None, help='Path to resume checkpoint folder')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3,4,5,6,7', help='Comma separated GPU IDs')
    parser.add_argument('--log_path', type=str, default='./experiments', help='Folder to save logs and models')
    parser.add_argument('--pretrain', type=str, default='/mntdata/main/light_sr/sr/results/DF2K/4x/results/model_x4_200.pt', help='Path to pretrained model')
    parser.add_argument('--model', type=str, default='your_model', help='Model name')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (total across GPUs)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--scale', type=int, default=2, help='Super resolution scale factor')
    parser.add_argument('--loss', type=str, default='L1Loss', help='Loss function name')
    parser.add_argument('--fp', type=int, default=32, help='Floating point precision (16 or 32)')
    parser.add_argument('--log_every', type=int, default=100, help='Steps interval for logging')
    parser.add_argument('--test_every', type=int, default=10, help='Epoch interval for testing/validation')
    parser.add_argument('--cache_folder_train', type=str, default='/mntdata/main/light_sr/sr/cache/realsr/train', help='Path to cache folder for preprocessed npy data')
    parser.add_argument('--cache_folder_val', type=str, default='/mntdata/main/light_sr/sr/cache/realsr/val', help='Path to cache folder for preprocessed npy data')
    parser.add_argument('--train_HR_folder', type=str, required=True, help='Path to training HR images')
    parser.add_argument('--train_LR_folder', type=str, required=True, help='Path to training LR images')
    parser.add_argument('--val_HR_folder', type=str, required=True, help='Path to validation HR images')
    parser.add_argument('--val_LR_folder', type=str, required=True, help='Path to validation LR images')
    parser.add_argument('--channels', type=int, default=1, help='Number of input image channels')
    parser.add_argument('--patch_size', type=int, default=96, help='Patch size for training')
    parser.add_argument('--save_dir', type=str, required=True, default="/mntdata/main/light_sr/sr/results/REALSR/4x/results", help='Directory to save model checkpoints and result images')
    parser.add_argument('--train_log_dir', type=str, default='/mntdata/main/light_sr/sr/results/REALSR/4x/results/train_log.txt', help='Directory to save training logs')
    parser.add_argument('--val_log_dir', type=str, default='/mntdata/main/light_sr/sr/results/REALSR/4x/results/val_log.txt', help='Directory to save validation logs')
    
    args = parser.parse_args()
    print(f"\nParsed arguments: {args}")


    # Set visible GPUs for torch.distributed
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    gpu_list = args.gpu_ids.split(',')
    world_size = len(gpu_list)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"Running on rank {rank} with local rank {local_rank} out of {world_size} total processes.")

    main_worker(rank=rank, world_size=world_size, args=args)
