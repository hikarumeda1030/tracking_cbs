'''Train Tiny-ImageNet (200 classes, 64x64) with PyTorch.'''
import argparse
import os
import requests
import zipfile

import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import wandb
from PIL import Image
from tqdm import tqdm

from training import train, test
from utils import config, select_model, lr_scheduler
from utils.sfo_schedule import make_exp_growth_sfo_epochs

MAX_BS = 2048


class TinyImageNetValDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_map, img_dir, class_to_idx, transform=None):
        self.annotations_map = annotations_map
        self.img_dir = img_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.image_filenames = list(annotations_map.keys())

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert('RGB')
        class_label = self.annotations_map[self.image_filenames[idx]]
        label = self.class_to_idx[class_label]
        if self.transform:
            image = self.transform(image)
        return image, label


def download_and_extract_tiny_imagenet(data_dir):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_filename = os.path.join(data_dir, "tiny-imagenet-200.zip")
    extracted_dir = os.path.join(data_dir, "tiny-imagenet-200")

    os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(extracted_dir):
        print("Downloading Tiny ImageNet dataset...")
        response = requests.get(url, stream=True)
        with open(zip_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Extracting Tiny ImageNet dataset...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_filename)
        print("Tiny ImageNet dataset is ready.")
    else:
        print("Tiny ImageNet dataset already exists.")


def load_validation_annotations(val_dir):
    annotations_path = os.path.join(val_dir, "val_annotations.txt")
    annotations_map = {}
    with open(annotations_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                annotations_map[parts[0]] = parts[1]
    return annotations_map


# Command Line Argument
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Tiny-ImageNet Training with Schedulers')
    parser.add_argument('config_path', type=str, help='path of config file(.yaml)')
    parser.add_argument('wandb_project', type=str, help='wandb project name')
    parser.add_argument('--cuda_device', type=int, default=0, help='CUDA device number (default: 0)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    cfg = config.load_yaml(args.config_path)

    epochs = config.require(cfg, "train.epochs")

    # wandb init
    wandb.init(
        project=args.wandb_project,
        config={
            "config_path": args.config_path,
            "cuda_device": args.cuda_device,
            **cfg,
        },
    )

    # Dataset Preparation
    data_dir = config.optional(cfg, "data.data_dir", "../../data")
    download_and_extract_tiny_imagenet(data_dir)

    train_dir = os.path.join(data_dir, "tiny-imagenet-200", "train")
    val_dir = os.path.join(data_dir, "tiny-imagenet-200", "val")

    mean = [0.4802, 0.4481, 0.3975]
    std = [0.2770, 0.2691, 0.2821]

    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)

    trainset_full = torchvision.datasets.ImageFolder(
        root=train_dir,
        transform=transform_val,
    )

    annotations_map = load_validation_annotations(val_dir)
    valset = TinyImageNetValDataset(
        annotations_map=annotations_map,
        img_dir=os.path.join(val_dir, "images"),
        class_to_idx=trainset.class_to_idx,
        transform=transform_val,
    )

    full_batch_size = config.optional(cfg, "train.full_bs", 100)
    full_num_workers = config.optional(cfg, "train.full_num_workers", 8)

    trainloader_full = torch.utils.data.DataLoader(
        trainset_full,
        batch_size=full_batch_size,
        shuffle=True,
        num_workers=full_num_workers,
        pin_memory=True,
        persistent_workers=(full_num_workers > 0),
    )

    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=full_batch_size,
        shuffle=False,
        num_workers=full_num_workers,
        pin_memory=True,
        persistent_workers=(full_num_workers > 0),
    )

    # Device & Model
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    model_name = config.require(cfg, "model.name")
    model = select_model(model_name=model_name, num_classes=200, dataset="tiny_imagenet").to(device)
    print(f"model: {model_name}")

    criterion = nn.CrossEntropyLoss()

    lr = config.require(cfg, "train.lr")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    lr_sched_type = config.optional(cfg, "train.lr_scheduler")
    lr_sched = lr_scheduler(optimizer, cfg)

    bs = config.require(cfg, "train.bs")
    train_num_workers = config.optional(cfg, "train.num_workers", 8)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=bs,
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=True,
        persistent_workers=(train_num_workers > 0),
        drop_last=False,
    )

    bs_sched = config.optional(cfg, "train.bs_scheduler")
    bs_epoch = bs

    if bs_sched == "exp_growth":
        exp_rate = config.optional(cfg, "train.bs_exp_rate", 1.0)
        exp_every0 = config.optional(cfg, "train.exp_every", 1)
        exp_every_power = config.optional(cfg, "train.exp_every_power", 0.0)

        bs_stage_idx = 1
        exp_every = int(exp_every0)
        exp_every = max(1, exp_every)

        next_change_epoch = exp_every

    stages = config.optional(cfg, "train.stages", None)

    # for exp_growth_sfo
    if bs_sched == "exp_growth_sfo":
        assert stages is not None, "train.stages is required for exp_growth_sfo"
        bs_exp_rate = config.optional(cfg, "train.bs_exp_rate", 1.0)
        bs_change_epochs = make_exp_growth_sfo_epochs(
            epochs=epochs,
            stages=stages,
            exp_rate=bs_exp_rate,
            power=1.0,
        )
        bs_stage_idx = 0

    steps = 0
    best_acc = -1.0
    full_grad_every = config.optional(cfg, "train.full_grad_every", 1)

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    for epoch in tqdm(range(epochs), desc="Epoch"):
        if bs_sched == "exp_growth_sfo":
            if bs_stage_idx < len(bs_change_epochs) and epoch == bs_change_epochs[bs_stage_idx]:
                bs_stage_idx += 1

                bs_epoch = int(bs * (bs_exp_rate ** bs_stage_idx))
                if lr_sched_type == "exp_growth" and epoch > 0:
                    if bs_epoch <= MAX_BS:
                        lr_sched.step()
                bs_epoch = min(bs_epoch, MAX_BS)
                bs_epoch = max(1, bs_epoch)

                trainloader = torch.utils.data.DataLoader(
                    trainset,
                    batch_size=bs_epoch,
                    shuffle=True,
                    num_workers=train_num_workers,
                    pin_memory=True,
                    persistent_workers=(train_num_workers > 0),
                    drop_last=False,
                )

        if bs_sched == "exp_growth":
            if epoch == next_change_epoch:
                bs_epoch = int(bs * (exp_rate ** bs_stage_idx))
                if lr_sched_type == "exp_growth" and epoch > 0:
                    if bs_epoch <= MAX_BS:
                        lr_sched.step()
                bs_epoch = min(bs_epoch, MAX_BS)
                bs_epoch = max(1, bs_epoch)

                trainloader = torch.utils.data.DataLoader(
                    trainset,
                    batch_size=bs_epoch,
                    shuffle=True,
                    num_workers=train_num_workers,
                    pin_memory=True,
                    persistent_workers=(train_num_workers > 0),
                    drop_last=False,
                )

                bs_stage_idx += 1

                exp_every = int(round(exp_every0 * (exp_rate ** (exp_every_power * bs_stage_idx))))
                exp_every = max(1, exp_every)

                next_change_epoch = epoch + exp_every

        compute_full_grad = (epoch % full_grad_every == 0)
        norm, train_result = train(model, optimizer, device, criterion, trainloader, trainloader_full, scaler=scaler, amp_enabled=True, compute_full_grad=compute_full_grad)
        val_result = test(model, device, valloader, criterion, amp_enabled=True)

        train_loss, train_acc = train_result
        val_loss, val_acc = val_result

        steps += len(trainloader)
        current_lr = optimizer.param_groups[0]["lr"]

        # wandb log
        log_dict = {
            "epoch": epoch,
            "steps": steps,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "opt/lr": current_lr,
            "opt/bs": bs_epoch,
        }
        if norm is not None:
            log_dict["grad/full_norm"] = float(norm)
        wandb.log(log_dict, step=steps)

        # best model save
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(wandb.run.dir, "best.pt")
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "config": cfg,
            }, ckpt_path)
            wandb.save(ckpt_path)

    wandb.summary["best/val_acc"] = best_acc
    wandb.finish()
