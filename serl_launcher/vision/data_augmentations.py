import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def batched_random_crop(img: torch.Tensor, rng: torch.Generator, padding: int, num_batch_dims: int = 1) -> torch.Tensor:
    """Batched random crop with replicate padding for arbitrary batch dimensions

    Args:
        img: Input tensor with shape [*batch_dims, H, W, C]
        rng: Random number generator
        padding: Padding size for crop augmentation
        num_batch_dims: Number of batch dimensions in input

    Returns:
        Cropped tensor with same shape as input
    """
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
    """Resize HWC image tensor to target dimensions using bilinear interpolation

    Args:
        image: Input image tensor with shape [H, W, C]
        image_dim: Target dimensions (H, W)

    Returns:
        Resized image tensor with shape [image_dim[0], image_dim[1], C]
    """
    image_nchw = image.permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(image_nchw, size=image_dim, mode="bilinear", align_corners=False)
    return resized.squeeze(0).permute(1, 2, 0)


def _maybe_apply(apply_fn, inputs: torch.Tensor, rng: torch.Generator, apply_prob: float) -> torch.Tensor:
    """Conditionally apply transformation based on probability

    Args:
        apply_fn: Transformation function to apply
        inputs: Input tensor
        rng: Random number generator
        apply_prob: Probability (0-1) to apply the transformation

    Returns:
        Transformed or original input tensor
    """
    should_apply = torch.rand(1, generator=rng, device=inputs.device).item() <= apply_prob
    return apply_fn(inputs) if should_apply else inputs


def rgb_to_hsv(r: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> tuple:
    """Convert RGB channels to HSV color space

    Args:
        r: Red channel tensor
        g: Green channel tensor
        b: Blue channel tensor

    Returns:
        Tuple of (hue, saturation, value) tensors
    """
    rgb = torch.stack([r, g, b], dim=-1)
    hsv = TF.rgb_to_hsv(rgb)
    return hsv[..., 0], hsv[..., 1], hsv[..., 2]


def hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> tuple:
    """Convert HSV channels back to RGB color space

    Args:
        h: Hue channel tensor
        s: Saturation channel tensor
        v: Value channel tensor

    Returns:
        Tuple of (red, green, blue) tensors
    """
    hsv = torch.stack([h, s, v], dim=-1)
    rgb = TF.hsv_to_rgb(hsv)
    return rgb[..., 0], rgb[..., 1], rgb[..., 2]


def adjust_brightness(rgb_tuple: tuple, delta: float) -> tuple:
    """Adjust brightness of RGB channels by adding delta

    Args:
        rgb_tuple: Tuple of (R, G, B) tensors
        delta: Brightness adjustment value

    Returns:
        Tuple of adjusted RGB tensors
    """
    return tuple(x + delta for x in rgb_tuple)


def adjust_contrast(image: torch.Tensor, factor: float) -> torch.Tensor:
    """Adjust contrast of image tensor

    Args:
        image: Input image tensor (HWC format)
        factor: Contrast adjustment factor

    Returns:
        Contrast-adjusted image tensor
    """
    mean = image.mean(dim=(-2, -1), keepdim=True)
    return factor * (image - mean) + mean


def adjust_saturation(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor, factor: float) -> tuple:
    """Adjust saturation channel of HSV image

    Args:
        h: Hue channel tensor
        s: Saturation channel tensor
        v: Value channel tensor
        factor: Saturation adjustment factor

    Returns:
        Tuple of (hue, adjusted_saturation, value) tensors
    """
    s_adjusted = torch.clamp(s * factor, 0.0, 1.0)
    return h, s_adjusted, v


def adjust_hue(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor, delta: float) -> tuple:
    """Adjust hue channel of HSV image with wrapping

    Args:
        h: Hue channel tensor
        s: Saturation channel tensor
        v: Value channel tensor
        delta: Hue adjustment value

    Returns:
        Tuple of (adjusted_hue, saturation, value) tensors
    """
    h_adjusted = (h + delta) % 1.0
    return h_adjusted, s, v


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
    """Comprehensive color transformation with jitter and grayscale augmentation

    Args:
        image: Input image tensor (HWC format, 0-1 range)
        rng: Random number generator
        brightness: Brightness jitter strength (0 = no jitter)
        contrast: Contrast jitter strength (0 = no jitter)
        saturation: Saturation jitter strength (0 = no jitter)
        hue: Hue jitter strength (0 = no jitter)
        to_grayscale_prob: Probability to convert to grayscale
        color_jitter_prob: Probability to apply color jitter
        apply_prob: Overall probability to apply any color transform
        shuffle: Whether to shuffle order of color jitter operations

    Returns:
        Transformed image tensor (clamped to 0-1 range)
    """

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
    """Apply Gaussian blur with random sigma to image tensor

    Args:
        image: Input image tensor (HWC format)
        rng: Random number generator
        blur_divider: Factor to calculate kernel size (kernel_size = image_height / blur_divider)
        sigma_min: Minimum sigma value for Gaussian kernel
        sigma_max: Maximum sigma value for Gaussian kernel
        apply_prob: Probability to apply blur

    Returns:
        Blurred or original image tensor
    """
    kernel_size = int(image.shape[0] / blur_divider) | 1

    def _apply(image):
        sigma = sigma_min + torch.rand(1, generator=rng, device=image.device).item() * (sigma_max - sigma_min)
        image_nchw = image.permute(2, 0, 1).unsqueeze(0)
        blurred = TF.gaussian_blur(image_nchw, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
        return blurred.squeeze(0).permute(1, 2, 0)

    return _maybe_apply(_apply, image, rng, apply_prob)


def solarize(image: torch.Tensor, rng: torch.Generator, threshold: float, apply_prob: float) -> torch.Tensor:
    """Solarize image by inverting pixels above threshold

    Args:
        image: Input image tensor (0-1 range)
        rng: Random number generator
        threshold: Threshold for solarization (0-1)
        apply_prob: Probability to apply solarization

    Returns:
        Solarized or original image tensor
    """

    def _apply(image):
        return torch.where(image < threshold, image, 1.0 - image)

    return _maybe_apply(_apply, image, rng, apply_prob)
