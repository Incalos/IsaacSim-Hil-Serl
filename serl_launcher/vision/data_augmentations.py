import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def batched_random_crop(img: torch.Tensor, rng: torch.Generator, padding: int, num_batch_dims: int = 1) -> torch.Tensor:
    # Randomly crop each image in a (possibly multi-dimensional) batch with replication padding
    original_shape = img.shape
    batch_size = 1
    for i in range(num_batch_dims):
        batch_size *= original_shape[i]
    H, W, C = original_shape[-3], original_shape[-2], original_shape[-1]
    img = img.reshape(batch_size, H, W, C)
    img_bchw = img.permute(0, 3, 1, 2)
    padded = F.pad(img_bchw, (padding, padding, padding, padding), mode="replicate")
    padded = padded.permute(0, 2, 3, 1)
    offsets = torch.randint(0, 2 * padding + 1, (batch_size, 2), generator=rng).to(img.device)
    h_indices = torch.arange(H, device=img.device).unsqueeze(0).unsqueeze(-1)
    w_indices = torch.arange(W, device=img.device).unsqueeze(0).unsqueeze(0)
    h_offsets = offsets[:, 0].view(batch_size, 1, 1)
    w_offsets = offsets[:, 1].view(batch_size, 1, 1)
    h_idx = (h_indices + h_offsets).expand(batch_size, H, W)
    w_idx = (w_indices + w_offsets).expand(batch_size, H, W)
    b_idx = torch.arange(batch_size, device=img.device).view(batch_size, 1, 1).expand(batch_size, H, W)
    cropped = padded[b_idx, h_idx, w_idx, :]
    return cropped.reshape(original_shape)


def resize(image: torch.Tensor, image_dim: tuple) -> torch.Tensor:
    # Resize image to the given (H, W) dimensions in float [0,1]
    assert len(image_dim) == 2
    return (
        F.interpolate(image.permute(2, 0, 1).unsqueeze(0), size=image_dim, mode="bilinear", align_corners=False)
        .squeeze(0)
        .permute(1, 2, 0)
    )


def _maybe_apply(apply_fn, inputs: torch.Tensor, rng: torch.Generator, apply_prob: float) -> torch.Tensor:
    # Apply a transform with a given probability, otherwise return inputs unchanged
    should_apply = torch.rand(1, generator=rng, device=inputs.device).item() <= apply_prob
    return apply_fn(inputs) if should_apply else inputs


def rgb_to_hsv(r: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> tuple:
    # Convert three RGB channels to HSV color space
    return TF.rgb_to_hsv(torch.stack([r, g, b], dim=-1))


def hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> tuple:
    # Convert HSV channels back to three RGB tensors
    hsv = torch.stack([h, s, v], dim=-1)
    rgb = TF.hsv_to_rgb(hsv)
    return rgb[..., 0], rgb[..., 1], rgb[..., 2]


def adjust_brightness(rgb_tuple: tuple, delta: float) -> tuple:
    # Shift brightness of an RGB tuple by a scalar delta
    return tuple(x + delta for x in rgb_tuple)


def adjust_contrast(image: torch.Tensor, factor: float) -> torch.Tensor:
    # Scale contrast around the per-image mean
    mean = image.mean(dim=(-2, -1), keepdim=True)
    return factor * (image - mean) + mean


def adjust_saturation(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor, factor: float) -> tuple:
    # Scale saturation channel in HSV, clipped to [0,1]
    return h, torch.clamp(s * factor, 0.0, 1.0), v


def adjust_hue(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor, delta: float) -> tuple:
    # Additive hue shift in HSV, wrapped to [0,1)
    return (h + delta) % 1.0, s, v


def color_transform(
    image: torch.Tensor,
    rng: torch.Generator,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
    to_grayscale_prob: float = 0.0,
    color_jitter_prob: float = 1.0,
    apply_prob: float = 1.0,
    shuffle: bool = True,
) -> torch.Tensor:
    # Apply stochastic color jitter and optional grayscale conversion
    def _to_grayscale(image):
        rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=image.device)
        grayscale = (image * rgb_weights).sum(dim=-1, keepdim=True)
        return grayscale.repeat(1, 1, 3)

    should_apply = torch.rand(1, generator=rng, device=image.device).item() <= apply_prob
    should_apply_gs = torch.rand(1, generator=rng, device=image.device).item() <= to_grayscale_prob
    should_apply_color = torch.rand(1, generator=rng, device=image.device).item() <= color_jitter_prob
    if should_apply and should_apply_color:
        transforms = []
        if brightness > 0:
            transforms.append(
                lambda img: TF.adjust_brightness(img, 1 + torch.rand(1, generator=rng).item() * brightness)
            )
        if contrast > 0:
            transforms.append(lambda img: TF.adjust_contrast(img, 1 + torch.rand(1, generator=rng).item() * contrast))
        if saturation > 0:
            transforms.append(
                lambda img: TF.adjust_saturation(img, 1 + torch.rand(1, generator=rng).item() * saturation)
            )
        if hue > 0:
            transforms.append(lambda img: TF.adjust_hue(img, torch.rand(1, generator=rng).item() * hue))
        if shuffle:
            indices = torch.randperm(len(transforms), generator=rng)
            transforms = [transforms[i] for i in indices]
        image_NCHW = image.permute(2, 0, 1).unsqueeze(0)
        for t in transforms:
            image_NCHW = t(image_NCHW)
        image = image_NCHW.squeeze(0).permute(1, 2, 0)
    if should_apply and should_apply_gs:
        image = _to_grayscale(image)
    return torch.clamp(image, 0.0, 1.0)


def gaussian_blur(
    image: torch.Tensor,
    rng: torch.Generator,
    blur_divider: float = 10.0,
    sigma_min: float = 0.1,
    sigma_max: float = 2.0,
    apply_prob: float = 1.0,
) -> torch.Tensor:
    # Apply Gaussian blur with random sigma and image-size-dependent kernel
    kernel_size = int(image.shape[0] / blur_divider) | 1

    def _apply(image):
        sigma = sigma_min + torch.rand(1, generator=rng, device=image.device).item() * (sigma_max - sigma_min)
        return (
            TF.gaussian_blur(
                image.permute(2, 0, 1).unsqueeze(0), kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma]
            )
            .squeeze(0)
            .permute(1, 2, 0)
        )

    return _maybe_apply(_apply, image, rng, apply_prob)


def solarize(image: torch.Tensor, rng: torch.Generator, threshold: float, apply_prob: float) -> torch.Tensor:
    # Invert pixel values above a threshold with some probability
    def _apply(image):
        return torch.where(image < threshold, image, 1.0 - image)

    return _maybe_apply(_apply, image, rng, apply_prob)
