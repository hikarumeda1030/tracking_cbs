# Additional Experiments

This directory contains figures for the additional experiments referenced in our rebuttal.

## 1. ResNet-18 on Tiny-ImageNet

Validates the scheduling guideline $\delta \approx \gamma^2$ on a more challenging dataset (200 classes, 64×64 images), isolating the effect of dataset complexity from architectural choices.

- `resnet18_tinyimagenet_lr_growth.pdf`: Comparison of LR growth factors ($\gamma = 1.1, 1.2, 1.3, 1.4$) with fixed BS growth factor ($\delta = 2.0$)
- `resnet18_tinyimagenet_bs_growth.pdf`: Comparison of BS growth factors ($\delta = 2.0, 3.0, 4.0$) with fixed LR growth factor ($\gamma = 1.4$)
- `resnet18_tinyimagenet_epoch_alloc.pdf`: Comparison of stage-wise epoch allocation strategies

## 2. Swin Transformer-Tiny on Tiny-ImageNet

Validates $\delta \approx \gamma^2$ beyond convolutional networks using a hierarchical vision transformer with shifted-window self-attention.

- `swin_tinyimagenet_lr_growth.pdf`: Comparison of LR growth factors ($\gamma = 1.1, 1.2, 1.3, 1.4$) with fixed BS growth factor ($\delta = 2.0$)

## 3. Linear Scaling Rule Comparison (ResNet-18 / CIFAR-100)

Compares the Linear Scaling Rule ($\gamma = \delta = 2.0$) against our proposed schedules ($\delta = 2.0$, $\gamma = 1.1, 1.2, 1.3, 1.4$) on ResNet-18 / CIFAR-100.

- `linear_scaling_comparison.pdf`

## 4. SGD vs Adam (ResNet-18 / CIFAR-100)

Compares SGD and Adam under three scheduling strategies on ResNet-18 / CIFAR-100:

- (a) BS fixed / LR fixed
- (b) BS exp / LR fixed
- (c) BS exp / LR exp

- `sgd_adam_comparison.pdf`
