import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
import numpy as np
import cv2
import random

def warp_and_blend(img1, img2, H):
    """Warps img2 using homography H and blends it with img1."""

    # Get image dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Define the four corners of img2
    corners_img2 = np.array([
        [0, 0], [w2, 0], [w2, h2], [0, h2]
    ], dtype=np.float32).reshape(-1, 1, 2)

    # Transform img2's corners using H
    transformed_corners = cv2.perspectiveTransform(corners_img2, H)

    # Compute bounding box for the final stitched image
    min_x = int(min(transformed_corners[:, 0, 0].min(), 0))
    max_x = int(max(transformed_corners[:, 0, 0].max(), w1))
    min_y = int(min(transformed_corners[:, 0, 1].min(), 0))
    max_y = int(max(transformed_corners[:, 0, 1].max(), h1))

    # Translation matrix to align coordinates to positive values
    translation = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ], dtype=np.float32)

    H_translated = translation @ H

    # Warp both images
    warped_img1 = cv2.warpPerspective(img1, translation, (max_x - min_x, max_y - min_y))
    warped_img2 = cv2.warpPerspective(img2, H_translated, (max_x - min_x, max_y - min_y))

    # Blend images (simple feathering)
    mask1 = (warped_img1 > 0).astype(np.uint8)
    mask2 = (warped_img2 > 0).astype(np.uint8)
    
    # Fix shape mismatch (Ensure blend_mask has correct dimensions)
    blend_mask = np.clip(mask1 + mask2, 0, 1).astype(np.uint8)[..., None]

    # Ensure all images have the same shape
    assert warped_img1.shape == warped_img2.shape, "Shape mismatch in blending!"

    # Create a blended panorama
    alpha = np.float32(mask2) / (mask1 + mask2 + 1e-6)  # Avoid division by zero
    blended = np.uint8(alpha * warped_img1 + (1 - alpha) * warped_img2)

    return blended