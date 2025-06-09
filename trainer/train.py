import os
import math
import yaml
import time
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import sys
import os
from torch.cuda.amp import autocast, GradScaler


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataset_utils.data_loader import CustomDataset
from models.model import SRConvnet 
from utils.util import calc_psnr, calc_ssim
import imageio  # for saving images
import platform

def is_linux():
    return platform.system() == "Linux"


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
    """
    Save super-resolved images from batch to disk in 'save_dir/epoch_xxxx/'.
    Assumes sr_batch is a tensor of shape [batch_size, channels, H, W].
    Saves images as PNGs.
    """
    save_epoch_dir = os.path.join(save_dir, f"results_epoch_{epoch}")
    os.makedirs(save_epoch_dir, exist_ok=True)

    sr_batch = sr_batch.detach().cpu().float()
    # If input channels > 1, convert to numpy image accordingly, else squeeze
    for idx in range(sr_batch.size(0)):
        img = sr_batch[idx]
        img_np = img.permute(1, 2, 0).numpy()  # C,H,W -> H,W,C
        # If single channel, squeeze last dim
        if img_np.shape[2] == 1:
            img_np = img_np[:, :, 0]
        # Clamp to [0,255] and convert to uint8
        img_np = img_np.clip(0, 255).astype('uint8')
        save_path = os.path.join(save_epoch_dir, f"sr_{idx}.png")
        imageio.imwrite(save_path, img_np)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup(rank, world_size):
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        # Choose NCCL on Linux with CUDA, otherwise Gloo
        use_nccl = platform.system() == "Linux" and torch.cuda.is_available() and dist.is_nccl_available()
        backend = "nccl" if use_nccl else "gloo"
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def main_worker(rank, world_size, args):
    setup(rank, world_size)

    # Load config
    config = {}
    if args.config:
        config = load_config(args.config)
    config.update(vars(args))  # Override with CLI args

    device = torch.device(f'cuda:{rank}')

    # Create save_dir and results directory on rank 0
    if rank == 0:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir, exist_ok=True)
        results_dir = os.path.join(args.save_dir, 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
    #dist.barrier()  # wait for rank 0 to create dirs
    if world_size > 1:
        dist.barrier()


    train_dataset = CustomDataset(
        HR_folder=config['train_HR_folder'],
        LR_folder=config['train_LR_folder'],
        scale=config.get('scale', 2),
        colors=config.get('channels', 3),
        patch_size=config.get('patch_size', 64),
        train=True,
        augment=True,
        repeat=1
    )

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None


    #train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    '''
    train_loader = DataLoader(train_dataset,
                              batch_size=config.get('batch_size', 16) // world_size,
                              sampler=train_sampler,
                              num_workers=4,
                              pin_memory=True)
    '''
    train_loader = DataLoader(train_dataset,
                            batch_size=config.get('batch_size', 2),
                            shuffle=(train_sampler is None),
                            sampler=train_sampler,
                            num_workers=4,
                            pin_memory=True)

    val_dataset = CustomDataset(
        HR_folder=config['val_HR_folder'],
        LR_folder=config['val_LR_folder'],
        scale=config.get('scale', 2),
        colors=config.get('channels', 3),
        train=False,
        augment=False
    )

#    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
#    val_loader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            sampler=val_sampler,
                            num_workers=2,
                            pin_memory=True)
    model = SRConvnet(
        scale=config.get('scale', 2),
        num_kernels=config.get('num_kernels', 8),
        num_acb=config.get('num_acb', 8),
        dimension=config.get('dimension', 64),
        num_heads=config.get('num_heads', 4)
    ).to(device)
    if config.get('fp') == 16:
        model = model.half()

    if config.get('pretrain') and rank == 0:
        print(f"Loading pretrained model from {config['pretrain']}")
        checkpoint = torch.load(config['pretrain'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    loss_func = getattr(nn, config.get('loss', 'L1Loss'))()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-4), eps=1e-8)
    scheduler = MultiStepLR(optimizer, milestones=config.get('decays', [50, 100]), gamma=config.get('gamma', 0.5))

    start_epoch = 1
    stat_dict = {'losses': [], 'psnrs': [], 'ssims': []}

    if config.get('resume'):
        checkpoint_path = os.path.join(config['resume'], 'models', f"model_x{config['scale']}_latest.pt")
        if os.path.isfile(checkpoint_path):
            if rank == 0:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.module.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                stat_dict = checkpoint.get('stat_dict', stat_dict)
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resumed training from epoch {start_epoch}")
            start_epoch_tensor = torch.tensor(start_epoch, device=device)
            dist.broadcast(start_epoch_tensor, src=0)
            start_epoch = start_epoch_tensor.item()
        dist.barrier()

    epochs = config.get('epochs', 200)
    for epoch in range(start_epoch, epochs + 1):
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
            loss = loss_func(sr, hr)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (i + 1) % config.get('log_every', 100) == 0 and rank == 0:
                avg_loss = epoch_loss / (i + 1)
                print(f"Epoch [{epoch}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {avg_loss:.4f}")

        scheduler.step()

        if epoch % config.get('test_every', 10) == 0:
            model.eval()
            avg_psnr = 0
            avg_ssim = 0
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)
            with torch.no_grad():
                sr_images_to_save = []
            for idx, (lr, hr, _) in enumerate(val_loader):
                lr, hr = lr.to(device), hr.to(device)
                ...
                sr = model(lr).clamp(0, 255)  # sr is [1, C, H, W]

                # Loop over each image in batch (usually just 1)
                for b in range(sr.size(0)):
                    sr_img = sr[b]  # [C, H, W]
                    hr_img = hr[b]
                    psnr = calc_psnr(sr_img, hr_img)
                    ssim = calc_ssim(sr_img, hr_img)
                    avg_psnr += psnr
                    avg_ssim += ssim

                    # Collect sr images for saving (only on rank 0)
                    if rank == 0:
                        sr_images_to_save.append(sr)

                avg_psnr /= len(val_loader)
                avg_ssim /= len(val_loader)

                if rank == 0:
                    print(f"Validation PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")
                    stat_dict['psnrs'].append(avg_psnr)
                    stat_dict['ssims'].append(avg_ssim)

                    # Save SR images from validation batch
                    # Concatenate all images in the validation set and save
                    if len(sr_images_to_save) > 0:
                        sr_images_concat = torch.cat(sr_images_to_save, dim=0)
                        save_sr_images(sr_images_concat, epoch, os.path.join(args.save_dir, 'results'))

                    # Save checkpoint on test_every epochs
                    save_path = os.path.join(args.save_dir, f"model_x{config['scale']}_{epoch}.pt")
                    save_model(save_path, epoch, model, optimizer, scheduler, stat_dict)

        # Save checkpoint every 50 epochs explicitly
        if epoch % 50 == 0:
            if rank == 0:
                save_path_50 = os.path.join(args.save_dir, f"model_x{config['scale']}_{epoch}_checkpoint.pt")
                print(f"Saving checkpoint at epoch {epoch} to {save_path_50}")
                save_model(save_path_50, epoch, model, optimizer, scheduler, stat_dict)

    if rank == 0:
        print("Training complete.")

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Super-Resolution Model with DDP")
    parser = argparse.ArgumentParser(description="Train Super-Resolution Model with DDP")

    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to resume checkpoint folder')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3,4,5,6,7', help='Comma separated GPU IDs')
    parser.add_argument('--log_path', type=str, default='./experiments', help='Folder to save logs and models')
    parser.add_argument('--pretrain', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--model', type=str, default='your_model', help='Model name')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (total across GPUs)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--scale', type=int, default=2, help='Super resolution scale factor')
    parser.add_argument('--loss', type=str, default='L1Loss', help='Loss function name')
    parser.add_argument('--fp', type=int, default=32, help='Floating point precision (16 or 32)')
    parser.add_argument('--log_every', type=int, default=100, help='Steps interval for logging')
    parser.add_argument('--test_every', type=int, default=10, help='Epoch interval for testing/validation')

    parser.add_argument('--train_HR_folder', type=str, required=True, help='Path to training HR images')
    parser.add_argument('--train_LR_folder', type=str, required=True, help='Path to training LR images')
    parser.add_argument('--val_HR_folder', type=str, required=True, help='Path to validation HR images')
    parser.add_argument('--val_LR_folder', type=str, required=True, help='Path to validation LR images')

    parser.add_argument('--channels', type=int, default=1, help='Number of input image channels')
    parser.add_argument('--patch_size', type=int, default=96, help='Patch size for training')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save model checkpoints and result images')

    args = parser.parse_args()

    # Set visible GPUs for torch.distributed
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    gpu_list = args.gpu_ids.split(',')
    world_size = len(gpu_list)


    def run_worker(rank, world_size, args):
        main_worker(rank=rank, world_size=world_size, args=args)

    if world_size > 1:
        # Launch as multiple processes (one per GPU) for DDP
        mp.spawn(
            run_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    else:
        # Single-GPU mode
        main_worker(rank=0, world_size=1, args=args)
