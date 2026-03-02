from typing import Dict, Iterable
import torch
import torch.nn as nn
from einops import rearrange


class EncodingWrapper(nn.Module):

    def __init__(
            self,
            encoder: Dict[str, nn.Module],
            use_proprio: bool = False,
            proprio_latent_dim: int = 64,
            enable_stacking: bool = False,
            image_keys: Iterable[str] = ("image",),
    ):
        super().__init__()
        self.encoders = nn.ModuleDict(encoder)
        self.use_proprio = use_proprio
        self.proprio_latent_dim = proprio_latent_dim
        self.enable_stacking = enable_stacking
        self.image_keys = list(image_keys)

        # Get output dimension from the first encoder
        first_encoder = list(self.encoders.values())[0]
        self.encoder_output_dim = first_encoder.output_dim

        # Calculate total dimension from all image encoders
        self.total_encoder_dim = self.encoder_output_dim * len(self.image_keys)

        # Configure output dimension with/without proprioception
        if use_proprio:
            self.proprio_encoder = None
            self.proprio_layer_norm = None
            self.output_dim = self.total_encoder_dim + proprio_latent_dim
        else:
            self.proprio_encoder = None
            self.proprio_layer_norm = None
            self.output_dim = self.total_encoder_dim

    def _init_proprio_encoder(self, state_dim: int, device: torch.device):
        # Initialize linear layer for proprioception encoding
        self.proprio_encoder = nn.Linear(state_dim, self.proprio_latent_dim).to(device)
        nn.init.xavier_uniform_(self.proprio_encoder.weight)
        nn.init.zeros_(self.proprio_encoder.bias)

        # Add layer normalization for stable training
        self.proprio_layer_norm = nn.LayerNorm(self.proprio_latent_dim).to(device)

    def forward(self, observations: Dict[str, torch.Tensor], train: bool = False) -> torch.Tensor:
        encoded = []

        # Encode all image inputs
        for image_key in self.image_keys:
            image = observations[image_key]

            # Reshape stacked images (B, T, H, W, C) -> (B, H, W, T*C)
            if self.enable_stacking and len(image.shape) == 5:
                image = rearrange(image, "B T H W C -> B H W (T C)")

            # Extract features from encoder
            features = self.encoders[image_key](image, train=train)
            encoded.append(features)

        # Concatenate features from all image encoders
        encoded = torch.cat(encoded, dim=-1)

        # Process proprioception if enabled
        if self.use_proprio:
            state = observations["state"]

            # Reshape stacked states
            if self.enable_stacking:
                if len(state.shape) == 2:
                    state = rearrange(state, "T C -> (T C)")
                    encoded = encoded.reshape(-1)
                elif len(state.shape) == 3:
                    state = rearrange(state, "B T C -> B (T C)")

            # Lazy initialization of proprio encoder
            if self.proprio_encoder is None:
                state_dim = state.shape[-1]
                self._init_proprio_encoder(state_dim, state.device)

            # Encode proprioceptive state
            state = self.proprio_encoder(state)
            state = self.proprio_layer_norm(state)
            state = torch.tanh(state)

            # Concatenate image features with proprioceptive features
            encoded = torch.cat([encoded, state], dim=-1)

        return encoded
