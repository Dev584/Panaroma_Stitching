import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
import numpy as np
import cv2
import random

def load_img(img_path):
    """Loads an image from a file path."""
    images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in img_path]
    if any(img is None for img in images):
        raise ValueError("Error: One or more images could not be loaded.")
    return images

def show_img(img, title="Image"):
    """Displays an image using matplotlib."""
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title(title)
    plt.show()

def cylindrical_projection(img, f):
    """Applies cylindrical projection to the image."""
    h, w = img.shape[:2]
    xc, yc = w // 2, h // 2  # Image center
    cyl_img = np.zeros_like(img)

    # Create new coordinates grid

    y_idx, x_idx = np.indices((h, w))

    # Compute cylindrical coordinates
    x_prime = f * np.tan((x_idx - xc) / f) + xc
    y_prime = ((y_idx - yc) / np.cos((x_idx - xc) / f)) + yc

    # Ensure valid pixel locations
    valid = (x_prime >= 0) & (x_prime < w) & (y_prime >= 0) & (y_prime < h)

    # Use bilinear interpolation to avoid gaps
    cyl_img[valid] = cv2.remap(img, x_prime.astype(np.float32), y_prime.astype(np.float32), cv2.INTER_LINEAR)

    return cyl_img