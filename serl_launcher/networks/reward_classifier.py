import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Dict, List, Tuple
import os
import numpy as np
from serl_launcher.vision.resnet_v1 import PreTrainedResNetEncoder
from serl_launcher.common.encoding import EncodingWrapper


class BinaryClassifier(nn.Module):
    """Binary classification model with encoder + MLP head.

    Args:
        encoder_def: Encoder module to process input features
        hidden_dim: Dimension of hidden layer in MLP head (default: 256)
    """

    def __init__(self, encoder_def: nn.Module, hidden_dim: int = 256):
        super().__init__()
        self.encoder_def = encoder_def
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(encoder_def.output_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Dict[str, torch.Tensor], train: bool = False) -> torch.Tensor:
        """Forward pass for binary classification.

        Args:
            x: Input dictionary containing tensor features
            train: Whether to set model to training mode (default: False)

        Returns:
            Scalar logit output for binary classification
        """
        self.train(train)
        x = self.encoder_def(x, train=train)
        return self.mlp(x)


class NWayClassifier(nn.Module):
    """N-way classification model with encoder + MLP head.

    Args:
        encoder_def: Encoder module to process input features
        hidden_dim: Dimension of hidden layer in MLP head (default: 256)
        n_way: Number of classification classes (default: 3)
    """

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
        """Forward pass for n-way classification.

        Args:
            x: Input dictionary containing tensor features
            train: Whether to set model to training mode (default: False)

        Returns:
            Logit outputs for n-way classification (shape: [batch_size, n_way])
        """
        self.train(train)
        x = self.encoder_def(x, train=train)
        return self.mlp(x)


def create_classifier(image_keys: List[str], n_way: int = 2, img_size: Tuple[int, int] = (128, 128)) -> nn.Module:
    """Create binary/n-way classifier with ResNet encoders.

    Args:
        image_keys: List of keys for image features in input dict
        n_way: Number of classification classes (2 for binary)
        img_size: Tuple of (height, width) for input images (default: (128, 128))

    Returns:
        Initialized classifier model with attached Adam optimizer
    """
    encoders = {
        image_key: PreTrainedResNetEncoder(
            model_name="resnet18",
            pooling_method="avg",
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

    if n_way == 2:
        model = BinaryClassifier(encoder_def=encoder_def)
    else:
        model = NWayClassifier(encoder_def=encoder_def, n_way=n_way)

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
    """Load trained classifier and return inference function.

    Args:
        image_keys: List of keys for image features in input dict
        checkpoint_path: Path to trained model checkpoint
        n_way: Number of classification classes (2 for binary)
        img_size: Tuple of (height, width) for input images (default: (128, 128))
        device: Computation device (cuda/cpu)

    Returns:
        Inference function that takes numpy dict input and returns numpy output

    Raises:
        FileNotFoundError: If checkpoint path does not exist
    """
    model = create_classifier(image_keys, n_way=n_way, img_size=img_size)
    model.to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Classifier checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"Classifier loaded from {checkpoint_path}")

    def func(obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Inference function for classifier.

        Args:
            obs: Dictionary of numpy arrays (image features)

        Returns:
            Scalar numpy array of classification output
        """
        with torch.no_grad():
            obs_torch = {}
            for k in image_keys:
                v = obs[k]
                if v.ndim == 3:
                    v = np.transpose(v, (2, 0, 1))
                    v = np.expand_dims(v, axis=0)
                elif v.ndim == 4 and v.shape[-1] == 3:
                    v = np.transpose(v, (0, 3, 1, 2))

                obs_torch[k] = torch.from_numpy(v).float().to(device)

            logits = model(obs_torch, train=False)
            return logits.detach().cpu().numpy().item()

    return func
