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
    mask_area /= (ni * nj)

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
    mom /= ((ni * nj) * (ni**2 + nj**2) / 12)  # normalize with mom of full rect

    return np.array([i, j, std_dist_i, std_dist_j, mom, mask_area])


