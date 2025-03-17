import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
import numpy as np
import cv2
import random

def warp_and_blend(img1, img2, H):
    # Applies homography to warp img1 and blends it smoothly with img2.
    
    # Get dimensions of both images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Define the four corners of img2
    corners_img2 = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32).reshape(-1, 1, 2)

    # Transform corners of img2 using homography
    transformed_corners = cv2.perspectiveTransform(corners_img2, H)

    # Compute the bounding box
    points = np.vstack((corners_img2, transformed_corners))
    [x_min, y_min] = np.int32(points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(points.max(axis=0).ravel() + 0.5)

    # Compute translation matrix to shift the result into positive coordinates
    translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
    H_translated = translation_matrix @ H

    # Warp img1 using homography
    panorama_width = x_max - x_min
    panorama_height = y_max - y_min
    warped_img1 = cv2.warpPerspective(img1, H_translated, (panorama_width, panorama_height))

    # Warp img2 into the same coordinate space
    warped_img2 = np.zeros_like(warped_img1)
    warped_img2[-y_min:h2 - y_min, -x_min:w2 - x_min] = img2

    # Create a mask to define valid regions in both images
    mask1 = (warped_img1 > 0).astype(np.float32)
    mask2 = (warped_img2 > 0).astype(np.float32)

    # Compute the overlap mask (feathered blending)
    overlap_mask = mask1 + mask2
    alpha = np.clip(mask2 / (overlap_mask + 1e-6), 0, 1)  # Normalize blend weights

    # Blend images using weighted averaging in the overlap region
    blended = (warped_img1 * (1 - alpha) + warped_img2 * alpha).astype(np.uint8)

    return blended