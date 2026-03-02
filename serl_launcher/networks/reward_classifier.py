import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Dict, List, Tuple
import os
import numpy as np
from serl_launcher.vision.resnet_v1 import PreTrainedResNetEncoder
from serl_launcher.common.encoding import EncodingWrapper


class BinaryClassifier(nn.Module):

    def __init__(self, encoder_def: nn.Module, hidden_dim: int = 256):
        super().__init__()
        self.encoder_def = encoder_def
        self.hidden_dim = hidden_dim
        # MLP head: encoder output -> hidden layer -> scalar logit
        self.mlp = nn.Sequential(
            nn.Linear(encoder_def.output_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Dict[str, torch.Tensor], train: bool = False) -> torch.Tensor:
        self.train(train)
        # Encode input features and pass through MLP classifier
        x = self.encoder_def(x, train=train)
        return self.mlp(x)


class NWayClassifier(nn.Module):

    def __init__(self, encoder_def: nn.Module, hidden_dim: int = 256, n_way: int = 3):
        super().__init__()
        self.encoder_def = encoder_def
        self.hidden_dim = hidden_dim
        self.n_way = n_way
        # MLP head: encoder output -> hidden layer -> n-way classification logits
        self.mlp = nn.Sequential(
            nn.Linear(encoder_def.output_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_way),
        )

    def forward(self, x: Dict[str, torch.Tensor], train: bool = False) -> torch.Tensor:
        self.train(train)
        # Encode input features and pass through MLP classifier
        x = self.encoder_def(x, train=train)
        return self.mlp(x)


def create_classifier(
        image_keys: List[str],
        n_way: int = 2,
        img_size: Tuple[int, int] = (128, 128),
):
    # Build ResNet encoders for each image key
    encoders = {
        image_key:
            PreTrainedResNetEncoder(
                model_name="resnet18",
                pooling_method="spatial_learned_embeddings",
                num_spatial_blocks=8,
                bottleneck_dim=256,
                freeze_backbone=True,
                pretrained=True,
                image_size=img_size,
            ) for image_key in image_keys
    }

    # Wrap encoders into a shared encoding module
    encoder_def = EncodingWrapper(
        encoder=encoders,
        use_proprio=False,
        enable_stacking=True,
        image_keys=image_keys,
    )

    # Initialize binary or multi-class classifier based on n_way
    if n_way == 2:
        model = BinaryClassifier(encoder_def=encoder_def)
    else:
        model = NWayClassifier(encoder_def=encoder_def, n_way=n_way)

    # Attach Adam optimizer to model
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.optimizer = optimizer

    return model


def load_classifier_func(
        image_keys: List[str],
        checkpoint_path: str,
        n_way: int = 2,
        img_size: Tuple[int, int] = (128, 128),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Callable[[Dict], np.ndarray]:
    # Initialize classifier architecture
    model = create_classifier(image_keys, n_way=n_way, img_size=img_size)
    model.to(device)

    # Validate checkpoint path existence
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Classifier checkpoint not found at: {checkpoint_path}")

    # Load trained weights from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Set model to evaluation mode
    model.eval()
    print(f"Classifier loaded from {checkpoint_path}")

    # Define inference function for numpy inputs/outputs
    def func(obs: Dict[str, np.ndarray]) -> np.ndarray:
        with torch.no_grad():
            obs_torch = {}
            for k in image_keys:
                v = obs[k]
                # Convert (H, W, C) to (1, C, H, W) for single image
                if v.ndim == 3:
                    v = np.transpose(v, (2, 0, 1))
                    v = np.expand_dims(v, axis=0)
                # Convert (B, H, W, C) to (B, C, H, W) for batch input
                elif v.ndim == 4 and v.shape[-1] == 3:
                    v = np.transpose(v, (0, 3, 1, 2))

                # Convert numpy array to torch tensor and move to device
                obs_torch[k] = torch.from_numpy(v).float().to(device)

            # Run forward pass and convert output to numpy
            logits = model(obs_torch, train=False)
            return logits.detach().cpu().numpy().item()

    return func
