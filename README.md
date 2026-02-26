# Tracking the Critical Batch Size for Batch Size and Learning Rate Scaling in SGD
Source code for reproducing our paper's experiments.

## Requirements

- Python 3.9+ (recommended)
- PyTorch + torchvision
- tqdm
- wandb

Example installation:

```bash
pip install torch torchvision tqdm wandb pyyaml
```

## Usage

The script expects two positional arguments:
1.	config_path: path to a YAML config
2.	wandb_project: wandb project name

Optional:
- --cuda_device: GPU index (default: 0)

Example:
```bash
python cifar100.py configs/example.yaml my-wandb-project --cuda_device 0
```

## Configuration

The script loads a YAML config via utils.config.load_yaml() and expects the following keys.

### Required keys
```yaml
model:
  name: resnet18     # example (must be supported by utils.select_model)

train:
  epochs: 200
  lr: 0.1
  bs: 128
```
### Optional keys
#### Learning-rate scheduler
```yaml
train:
  lr_scheduler: exp_growth
```
> Supported scheduler types and parameters depend on utils.lr_scheduler().
#### Batch size scheduler: exp_growth
```yaml
train:
  bs_scheduler: exp_growth
  bs_exp_rate: 2.0         # batch size multiplier per stage
  exp_every: 10            # how often (epochs) to increase batch size (base)
  exp_every_power: 0.0     # optional: changes exp_every over stages
```

### Outputs
- Metrics are logged to wandb each epoch:
  - train/test loss & accuracy
  - learning rate / batch size
  - gradient norm (grad/full_norm)
