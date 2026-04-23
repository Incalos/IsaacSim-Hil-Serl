from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

_RESNET_REGISTRY = {
    "resnet18": (models.resnet18, ResNet18_Weights),
    "resnet34": (models.resnet34, ResNet34_Weights),
    "resnet50": (models.resnet50, ResNet50_Weights),
}


class _ResNetBackbone(nn.Module):
    """Wrapper for ResNet backbone to extract convolutional features"""

    def __init__(self, resnet: nn.Module) -> None:
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
        return x


def _make_resnet_backbone(model_name: str, pretrained: bool) -> nn.Module:
    """Create ResNet backbone with specified configuration

    Args:
        model_name: Name of ResNet variant (resnet18/resnet34/resnet50)
        pretrained: Whether to load ImageNet pretrained weights

    Returns:
        Wrapped ResNet backbone module

    Raises:
        ValueError: If model_name is not in supported registry
    """
    if model_name not in _RESNET_REGISTRY:
        raise ValueError(f"Unsupported model: {model_name}. Supported: {list(_RESNET_REGISTRY.keys())}")

    model_fn, weights_cls = _RESNET_REGISTRY[model_name]
    weights = weights_cls.IMAGENET1K_V1 if pretrained else None
    resnet = model_fn(weights=weights)

    return _ResNetBackbone(resnet)


class SpatialLearnedEmbeddings(nn.Module):
    """Spatial learned embeddings layer for feature pooling

    Learns spatial kernel weights to extract discriminative features from 2D feature maps
    """

    def __init__(self, height: int, width: int, channel: int, num_features: int = 5) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features

        self.kernel = nn.Parameter(torch.randn(num_features, channel, height, width))
        nn.init.xavier_uniform_(self.kernel)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass for spatial learned embeddings

        Args:
            features: Input tensor with shape [B, H, W, C]

        Returns:
            Flattened pooled features with shape [B, C * num_features]
        """
        features_reshaped = features.permute(0, 3, 1, 2)
        features_expanded = features_reshaped.unsqueeze(1)
        kernel_expanded = self.kernel.unsqueeze(0)

        weighted = features_expanded * kernel_expanded
        summed = weighted.sum(dim=(3, 4))
        out = summed.reshape(features.shape[0], -1)

        return out


class PreTrainedResNetEncoder(nn.Module):
    """Encoder module using pretrained ResNet with configurable pooling and bottleneck

    Supports multiple pooling methods, frozen/trainable backbone, and bottleneck projection
    """

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
    ) -> None:
        super().__init__()
        self.pooling_method = pooling_method
        self.num_spatial_blocks = num_spatial_blocks
        self.bottleneck_dim = bottleneck_dim
        self.image_size = image_size
        self.freeze_backbone = freeze_backbone

        if shared_backbone is not None:
            self.backbone = shared_backbone
        else:
            self.backbone = _make_resnet_backbone(model_name, pretrained)
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                self.backbone.eval()

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, image_size[0], image_size[1])
            dummy_output = self.backbone(dummy_input)
            self.feature_channels = dummy_output.shape[1]
            self.feature_height = dummy_output.shape[2]
            self.feature_width = dummy_output.shape[3]

        if pooling_method == "avg":
            self.pooling = None
            self.dropout = None
            pooled_dim = self.feature_channels
        elif pooling_method == "max":
            self.pooling = None
            self.dropout = None
            pooled_dim = self.feature_channels
        elif pooling_method == "spatial_learned_embeddings":
            self.pooling = SpatialLearnedEmbeddings(
                height=self.feature_height,
                width=self.feature_width,
                channel=self.feature_channels,
                num_features=self.num_spatial_blocks,
            )
            self.dropout = nn.Dropout(0.1)
            pooled_dim = self.feature_channels * self.num_spatial_blocks
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")

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

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, observations: torch.Tensor, train: bool = True) -> torch.Tensor:
        """Forward pass for ResNet encoder

        Args:
            observations: Input image tensor (3D [H,W,C] or 4D [B,H,W,C]/[B,C,H,W])
            train: Whether in training mode (affects dropout)

        Returns:
            Encoded feature tensor with shape [B, output_dim]

        Raises:
            ValueError: If input dimensions are not 3D/4D or channel format is invalid
        """
        x = observations

        if x.dim() == 4:
            if x.shape[1] == 3:
                pass
            elif x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Unsupported 4D input shape: {x.shape}, expected [B,C,H,W] or [B,H,W,C]")
        elif x.dim() == 3:
            if x.shape[-1] == 3:
                x = x.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError(f"Unsupported 3D input shape: {x.shape}, expected [H,W,C]")
        else:
            raise ValueError(f"Input must be 3D or 4D, got {x.dim()}D")

        x = x.float()
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)

        x = x / 255.0
        x = (x - self.mean) / self.std

        if self.freeze_backbone:
            with torch.no_grad():
                x = self.backbone(x)
        else:
            x = self.backbone(x)

        x = x.permute(0, 2, 3, 1)

        if self.pooling_method == "avg":
            x = x.mean(dim=(1, 2))
        elif self.pooling_method == "max":
            x = x.amax(dim=(1, 2))
        elif self.pooling_method == "spatial_learned_embeddings":
            x = self.pooling(x)
            x = self.dropout(x) if self.training else x

        if self.bottleneck is not None:
            x = self.bottleneck(x)

        return x

    def train(self, mode: bool = True) -> nn.Module:
        """Override train method to keep backbone frozen if specified

        Args:
            mode: Training mode flag

        Returns:
            Self (encoder module)
        """
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
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
    """Create multiple ResNet encoders for different image keys

    Args:
        encoder_type: Type of encoder (e.g., resnet18-pretrained, resnet50)
        image_keys: List of image keys to create encoders for
        image_size: Input image dimensions (H, W)
        pooling_method: Pooling method (avg/max/spatial_learned_embeddings)
        num_spatial_blocks: Number of spatial blocks for learned embeddings
        bottleneck_dim: Dimension of bottleneck projection layer
        shared_backbone: Shared ResNet backbone across encoders

    Returns:
        Dictionary of encoders mapped to image keys

    Raises:
        NotImplementedError: If encoder_type is not supported
    """
    encoder_config = {
        "resnet18-pretrained": ("resnet18", True, False),
        "resnet18": ("resnet18", False, False),
        "resnet34-pretrained": ("resnet34", True, False),
        "resnet34": ("resnet34", False, False),
        "resnet50-pretrained": ("resnet50", True, False),
        "resnet50": ("resnet50", False, False),
    }

    if encoder_type not in encoder_config:
        raise NotImplementedError(f"Unknown encoder type: {encoder_type}, supported: {list(encoder_config.keys())}")

    model_name, pretrained, freeze_backbone = encoder_config[encoder_type]
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
