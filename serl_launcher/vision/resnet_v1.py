from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

# Registry mapping model names to corresponding ResNet classes and weight classes
_RESNET_REGISTRY = {
    "resnet18": (models.resnet18, ResNet18_Weights),
    "resnet34": (models.resnet34, ResNet34_Weights),
    "resnet50": (models.resnet50, ResNet50_Weights),
}


class _ResNetBackbone(nn.Module):

    def __init__(self, resnet: nn.Module):
        super().__init__()
        self.resnet = resnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x  # Output shape: [B, C, H, W]


def _make_resnet_backbone(model_name: str, pretrained: bool) -> nn.Module:
    if model_name not in _RESNET_REGISTRY:
        raise ValueError(f"Unsupported model: {model_name}. Supported: {list(_RESNET_REGISTRY.keys())}")

    model_fn, weights_cls = _RESNET_REGISTRY[model_name]
    weights = weights_cls.IMAGENET1K_V1 if pretrained else None
    resnet = model_fn(weights=weights)

    return _ResNetBackbone(resnet)


class PreTrainedResNetEncoder(nn.Module):

    def __init__(
            self,
            model_name: str = "resnet18",
            pooling_method: str = "avg",
            num_spatial_blocks: int = 8,
            bottleneck_dim: Optional[int] = 256,
            freeze_backbone: bool = True,
            pretrained: bool = True,
            image_size: Tuple[int, int] = (128, 128),
            shared_backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.pooling_method = pooling_method
        self.num_spatial_blocks = num_spatial_blocks
        self.bottleneck_dim = bottleneck_dim
        self.image_size = image_size
        self.freeze_backbone = freeze_backbone

        # Use shared backbone or create new one
        if shared_backbone is not None:
            self.backbone = shared_backbone
        else:
            self.backbone = _make_resnet_backbone(model_name, pretrained)
            # Freeze backbone parameters if specified
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                self.backbone.eval()

        # Get backbone output dimensions via dummy forward pass
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, image_size[0], image_size[1])
            dummy_output = self.backbone(dummy_input)
            self.feature_channels = dummy_output.shape[1]
            self.feature_height = dummy_output.shape[2]
            self.feature_width = dummy_output.shape[3]

        # Initialize pooling layer based on selected method
        if pooling_method == "avg":
            self.pooling = None
            self.dropout = None
            pooled_dim = self.feature_channels
        elif pooling_method == "max":
            self.pooling = None
            self.dropout = None
            pooled_dim = self.feature_channels
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")

        # Initialize bottleneck projection layer
        if bottleneck_dim is not None:
            self.bottleneck = nn.Sequential(
                nn.Linear(pooled_dim, bottleneck_dim),
                nn.LayerNorm(bottleneck_dim),
                nn.SiLU(),
            )
            nn.init.xavier_uniform_(self.bottleneck[0].weight)
            nn.init.zeros_(self.bottleneck[0].bias)
            self.output_dim = bottleneck_dim
        else:
            self.bottleneck = None
            self.output_dim = pooled_dim

        # ImageNet normalization parameters (registered as buffer)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, observations: torch.Tensor, train: bool = True) -> torch.Tensor:
        x = observations

        # Convert input shape from [B, H, W, C] to [B, C, H, W] (gym to torch format)
        if x.dim() == 4 and x.shape[-1] in [1, 3]:
            x = x.permute(0, 3, 1, 2)
        elif x.dim() == 3 and x.shape[-1] in [1, 3]:
            x = x.permute(2, 0, 1).unsqueeze(0)

        # Normalize pixel values to [0, 1] and apply ImageNet normalization
        x = x.float() / 255.0
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)

        # Extract features using ResNet backbone
        x = self.backbone(x)

        # Permute to [B, H, W, C] for spatial pooling
        x = x.permute(0, 2, 3, 1)

        # Apply selected pooling method
        if self.pooling_method == "avg":
            x = x.mean(dim=(1, 2))  # Output shape: [B, C]
        elif self.pooling_method == "max":
            x = x.amax(dim=(1, 2))  # Output shape: [B, C]

        # Apply bottleneck projection if enabled
        if self.bottleneck is not None:
            x = self.bottleneck(x)

        return x

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep backbone in eval mode if frozen
        if self.freeze_backbone:
            self.backbone.eval()
        return self


def create_encoder(
        encoder_type: str = "resnet18-pretrained",
        image_keys: Tuple[str, ...] = ("image",),
        image_size: Tuple[int, int] = (128, 128),
        pooling_method: str = "max",
        num_spatial_blocks: int = 8,
        bottleneck_dim: int = 256,
        shared_backbone: Optional[nn.Module] = None,
) -> dict:
    # Map encoder type to ResNet configuration
    if encoder_type == "resnet18-pretrained":
        model_name = "resnet18"
        pretrained = True
        freeze_backbone = False
    elif encoder_type == "resnet18":
        model_name = "resnet18"
        pretrained = False
        freeze_backbone = False
    elif encoder_type == "resnet34-pretrained":
        model_name = "resnet34"
        pretrained = True
        freeze_backbone = False
    elif encoder_type == "resnet34":
        model_name = "resnet34"
        pretrained = False
        freeze_backbone = False
    elif encoder_type == "resnet50-pretrained":
        model_name = "resnet50"
        pretrained = True
        freeze_backbone = False
    elif encoder_type == "resnet50":
        model_name = "resnet50"
        pretrained = False
        freeze_backbone = False
    else:
        raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

    # Create encoder instances for each image key
    encoders = {}
    for image_key in image_keys:
        encoders[image_key] = PreTrainedResNetEncoder(
            model_name=model_name,
            pooling_method=pooling_method,
            num_spatial_blocks=num_spatial_blocks,
            bottleneck_dim=bottleneck_dim,
            freeze_backbone=freeze_backbone,
            pretrained=pretrained,
            image_size=image_size,
            shared_backbone=shared_backbone,
        )

    return encoders
