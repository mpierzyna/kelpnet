import cv2
import numpy as np
import numba


def upsample_mask(mask):
    """Upsample prediction again for submission"""
    mask = cv2.resize(mask, (350, 350))
    return mask


@numba.njit(cache=True)
def compute_kelp_distr(mask):
    """
    COG gives rough position of kelp,
    inertia gives extend of patch around COG (not necessarily connected!)
    """
    # Area can always be computed
    ni, nj = mask.shape
    mask_area = np.sum(mask)
    mask_area /= ni * nj

    # If null, don't bother with rest
    if mask_area == 0:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 0])

    # Find the coordinates of the true values (where shape is marked by 1)
    true_coords = np.nonzero(mask)

    # Calculate the moments with respect to the origin
    m00 = len(true_coords[0])
    m10 = np.sum(true_coords[0])
    m01 = np.sum(true_coords[1])

    # Calculate centroid coordinates (i = row, j = col)
    i = m10 / m00
    j = m01 / m00

    # Calculate std of distances to the centroid (measure of size)
    dist_i = true_coords[0] - i
    dist_j = true_coords[1] - j

    std_dist_i = np.std(dist_i)
    std_dist_j = np.std(dist_j)

    # Calculate polar area moment of inertia (tells if mass is close or far away from center)
    mom = np.sum(dist_i**2 + dist_j**2)

    # Normalize
    i /= ni
    j /= nj
    std_dist_i /= ni
    std_dist_j /= nj
    mom /= (ni * nj) * (ni**2 + nj**2) / 12  # normalize with mom of full rect

    return np.array([i, j, std_dist_i, std_dist_j, mom, mask_area])


def inverse_log_scale_effect(img, scale):
    h, w = img.shape

    # Create meshgrid
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Normalize x and y to be in the range of -1 to 1
    x_normalized = 2.0 * x / (w - 1) - 1
    y_normalized = 2.0 * y / (h - 1) - 1

    # Apply an inverse log-scale-like transformation
    # The transformation is adjusted to expand the center and compress the edges
    x_transformed = (
        w * (np.sign(x_normalized) * (np.exp(np.abs(x_normalized) * scale) - 1) / np.exp(scale)) / 2 + w / 2
    )
    y_transformed = (
        h * (np.sign(y_normalized) * (np.exp(np.abs(y_normalized) * scale) - 1) / np.exp(scale)) / 2 + h / 2
    )

    # Ensure the transformed coordinates are within image bounds
    x_transformed = np.clip(x_transformed, 0, w - 1).astype(np.float32)
    y_transformed = np.clip(y_transformed, 0, h - 1).astype(np.float32)

    # Apply the custom remapping
    img_tf = cv2.remap(
        img, x_transformed, y_transformed, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
    )

    return img_tf


def compute_scaled_fft2(x):
    """Compute scaled 2D fft"""
    # Compute fft
    ft = np.fft.ifftshift(x)
    ft = np.fft.fft2(ft)
    ft = np.fft.fftshift(ft)
    ft = np.log(abs(ft))

    # Normalize and convert to uint16
    ft_min = ft.min()
    ft_max = ft.max()
    ft = (ft - ft_min) / (ft_max - ft_min)
    ft *= 2**16
    ft = ft.astype(np.uint16)

    # Stretch out center to emphasize low frequencies
    ft = inverse_log_scale_effect(ft, 2.75)

    # Invert normalization
    ft = ft / 2**16 * (ft_max - ft_min) + ft_min
    return ft


if __name__ == "__main__":
    img = np.random.random((256, 256))
    compute_ch_fft2(img)
