import numpy as np


def sin_func(x, a, b, c, d):
    # sin function
    return a * np.sin(b * (x + c)) + d


def gauss_func(x, a, x0, sigma):
    # Gaussian function
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def create_circular_mask(h, w, center, radius):
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask
