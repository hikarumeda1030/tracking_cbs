import timm


def vit_tiny_patch8(num_classes=200, **kwargs):
    """ViT-Tiny for 64x64 images (Tiny-ImageNet).
    patch_size=8 gives 8x8=64 tokens, comparable to ViT on 224x224 with patch_size=16.
    """
    return timm.create_model(
        'vit_tiny_patch16_224',
        pretrained=False,
        num_classes=num_classes,
        img_size=64,
        patch_size=8,
        **kwargs,
    )


def vit_tiny_patch4(num_classes=200, **kwargs):
    """ViT-Tiny for 64x64 images (Tiny-ImageNet).
    patch_size=4 gives 16x16=256 tokens.
    """
    return timm.create_model(
        'vit_tiny_patch16_224',
        pretrained=False,
        num_classes=num_classes,
        img_size=64,
        patch_size=4,
        **kwargs,
    )


def deit_small_patch8(num_classes=200, **kwargs):
    return timm.create_model(
        "deit_small_patch16_224",
        pretrained=False,
        num_classes=num_classes,
        img_size=64,
        patch_size=8,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.05,
        **kwargs,
    )


def deit_tiny_patch4(num_classes=200, **kwargs):
    """DeiT-Tiny for 64x64 images.
    patch_size=4 gives 16x16=256 tokens.
    """
    return timm.create_model(
        "deit_tiny_patch16_224",
        pretrained=False,
        num_classes=num_classes,
        img_size=64,
        patch_size=4,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        **kwargs,
    )


def swin_tiny_window4_64(num_classes=200, **kwargs):
    return timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        num_classes=num_classes,
        img_size=64,
        window_size=4,
        patch_size=2,
        **kwargs,
    )


def swin_small_window4_64(num_classes=200, **kwargs):
    default_kwargs = dict(
        pretrained=False,
        num_classes=num_classes,
        img_size=64,
        window_size=4,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
    )
    default_kwargs.update(kwargs)

    return timm.create_model(
        "swin_small_patch4_window7_224",
        **default_kwargs,
    )
