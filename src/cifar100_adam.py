'''Train CIFAR100 with PyTorch.'''
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
from tqdm import tqdm
import wandb

from training import train, test
from utils import config, select_model, lr_scheduler
from utils.sfo_schedule import make_exp_growth_sfo_epochs

MAX_BS = 4096


# Command Line Argument
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training with Schedulers')
    parser.add_argument('config_path', type=str, help='path of config file(.json)')
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
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    trainset = torchvision.datasets.CIFAR100(root='../../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='../../data', train=False, download=True, transform=transform_test)
    trainloader_full = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=4, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)

    # Device Setting
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    model_name = config.require(cfg, "model.name")
    model = select_model(model_name=model_name, num_classes=100).to(device)
    print(f"model: {model_name}")

    criterion = nn.CrossEntropyLoss()

    lr = config.require(cfg, "train.lr")
    beta1 = config.optional(cfg, "train.adam_beta1", 0.9)
    beta2 = config.optional(cfg, "train.adam_beta2", 0.999)
    eps = config.optional(cfg, "train.adam_eps", 1e-8)
    weight_decay = config.optional(cfg, "train.adam_weight_decay", 0.0)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
    )
    lr_sched_type = config.optional(cfg, "train.lr_scheduler")
    lr_sched = lr_scheduler(optimizer, cfg)

    bs = config.require(cfg, "train.bs")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)
    bs_sched = config.optional(cfg, "train.bs_scheduler")
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
    bs_epoch = bs
    full_grad_every = config.optional(cfg, "train.full_grad_every", 1)

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
                    num_workers=2,
                    pin_memory=True,
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
                    num_workers=2,
                    pin_memory=True,
                    drop_last=False,
                )

                bs_stage_idx += 1

                exp_every = int(round(exp_every0 * (exp_rate ** (exp_every_power * bs_stage_idx))))
                exp_every = max(1, exp_every)

                next_change_epoch = epoch + exp_every

        compute_full_grad = (epoch % full_grad_every == 0)
        norm, train_result = train(model, optimizer, device, criterion, trainloader, trainloader_full, compute_full_grad=compute_full_grad)
        test_result = test(model, device, testloader, criterion)

        train_loss, train_acc = train_result
        test_loss, test_acc = test_result

        steps += len(trainloader)
        current_lr = optimizer.param_groups[0]["lr"]

        # wandb log
        log_dict = {
            "epoch": epoch,
            "steps": steps,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "test/loss": test_loss,
            "test/acc": test_acc,
            "opt/lr": current_lr,
            "opt/bs": bs_epoch,
        }
        if norm is not None:
            log_dict["grad/full_norm"] = float(norm)
        wandb.log(log_dict, step=steps)

        # best model save
        if test_acc > best_acc:
            best_acc = test_acc
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

    wandb.summary["best/test_acc"] = best_acc
    wandb.finish()
