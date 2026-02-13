import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Dict, List, Tuple
import os
import numpy as np
from serl_launcher.vision.resnet_v1 import PreTrainedResNetEncoder
from serl_launcher.common.encoding import EncodingWrapper


class BinaryClassifier(nn.Module):
    # Binary classifier head on top of a shared encoder
    def __init__(self, encoder_def: nn.Module, hidden_dim: int = 256):
        super().__init__()
        self.encoder_def = encoder_def
        self.hidden_dim = hidden_dim
        # Simple MLP: encoder output -> hidden -> scalar logit
        self.mlp = nn.Sequential(
            nn.Linear(encoder_def.output_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Dict[str, torch.Tensor], train: bool = False) -> torch.Tensor:
        # Optionally switch between train/eval to control dropout etc.
        self.train(train)
        # Encode inputs and apply MLP classifier
        x = self.encoder_def(x, train=train)
        return self.mlp(x)


class NWayClassifier(nn.Module):
    # Multi-class classifier with shared encoder and MLP head
    def __init__(self, encoder_def: nn.Module, hidden_dim: int = 256, n_way: int = 3):
        super().__init__()
        self.encoder_def = encoder_def
        self.hidden_dim = hidden_dim
        self.n_way = n_way
        self.mlp = nn.Sequential(
            nn.Linear(encoder_def.output_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_way),
        )

    def forward(self, x: Dict[str, torch.Tensor], train: bool = False) -> torch.Tensor:
        self.train(train)
        x = self.encoder_def(x, train=train)
        return self.mlp(x)


def create_classifier(
    image_keys: List[str],
    n_way: int = 2,
    img_size: Tuple[int, int] = (128, 128),
):
    # Build encoders and wrap them into a shared EncodingWrapper, then attach a classifier head
    encoders = {
        image_key: PreTrainedResNetEncoder(
            model_name="resnet18",
            pooling_method="spatial_learned_embeddings",
            num_spatial_blocks=8,
            bottleneck_dim=256,
            freeze_backbone=True,
            pretrained=True,
            image_size=img_size,
        )
        for image_key in image_keys
    }
    encoder_def = EncodingWrapper(
        encoder=encoders,
        use_proprio=False,
        enable_stacking=True,
        image_keys=image_keys,
    )
    # Choose binary or multi-class classification head
    if n_way == 2:
        model = BinaryClassifier(encoder_def=encoder_def)
    else:
        model = NWayClassifier(encoder_def=encoder_def, n_way=n_way)
    # Attach an optimizer for convenience (similar to a train state)
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
    # Load a trained classifier from disk and return a NumPy-based inference function
    model = create_classifier(image_keys, n_way=n_way, img_size=img_size)
    model.to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Classifier checkpoint not found at: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Support both raw state_dict and wrapped dict formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"Classifier loaded from {checkpoint_path}")

    # Inference function that takes numpy observations and returns numpy logits
    def func(obs: Dict[str, np.ndarray]) -> np.ndarray:
        with torch.no_grad():
            obs_torch = {}
            for k in image_keys:
                v = obs[k]
                # Convert (H, W, C) to (1, C, H, W)
                if v.ndim == 3:
                    v = np.transpose(v, (2, 0, 1))
                    v = np.expand_dims(v, axis=0)
                # Convert (B, H, W, C) to (B, C, H, W)
                elif v.ndim == 4 and v.shape[-1] == 3:
                    v = np.transpose(v, (0, 3, 1, 2))
                obs_torch[k] = torch.from_numpy(v).float().to(device)
            logits = model(obs_torch, train=False)
            return logits.detach().cpu().numpy().item()

    return func
