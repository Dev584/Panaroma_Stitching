import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
import numpy as np
import cv2
import random

def load_img(img_path):
    """Loads images from the given path."""
    images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in img_path]
    if any(img is None for img in images):
        raise ValueError("Error: One or more images could not be loaded.")
    return images

def show_img(img, title="Image"):
    """Displays the image."""
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
    theta = np.arctan((x_idx - xc) / f)
    h_ = (y_idx - yc) / np.sqrt((x_idx - xc)**2 + f**2)

    x_prime = f * theta + xc
    y_prime = f * h_ + yc

    # Ensure valid pixel locations
    x_prime = np.clip(x_prime, 0, w - 1).astype(np.float32)
    y_prime = np.clip(y_prime, 0, h - 1).astype(np.float32)

    # Apply remapping using bilinear interpolation
    cyl_img = cv2.remap(img, x_prime, y_prime, interpolation=cv2.INTER_LINEAR)

    return cyl_img

def plot_matches(img1, keypoints1, img2, keypoints2, matches):
    """Draws matched feature points between two images."""
    # Create a blank image to show matches
    img_matches = np.hstack((img1, img2))
    offset = img1.shape[1]  # Offset to shift points from img2

    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))

    for m in matches:
        pt1 = keypoints1[m[0]]  # Keypoint in Image 1
        pt2 = keypoints2[m[1]]  # Keypoint in Image 2

        pt2_shifted = (pt2[0] + offset, pt2[1])  # Shift x-coordinates for img2

        plt.plot([pt1[0], pt2_shifted[0]], [pt1[1], pt2_shifted[1]], color=np.random.rand(3,), linewidth=1.5)

    plt.axis("off")
    plt.title(f"Feature Matching (Total Matches: {len(matches)})")
    plt.show()

def plot_after_ransac(img1, img2, inliers1, inliers2):
    """Draws matched feature points between two images after RANSAC."""
    img_matches = np.hstack((img1, img2))
    offset = img1.shape[1]  # Offset to shift points from img2

    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))

    for pt1, pt2 in zip(inliers1, inliers2):
        pt2_shifted = (pt2[0] + offset, pt2[1])  # Adjust x-coordinates for second image
        plt.plot([pt1[0], pt2_shifted[0]], [pt1[1], pt2_shifted[1]], color=np.random.rand(3,), linewidth=1.5)

    plt.axis("off")
    plt.title(f"Feature Matching After RANSAC (Total Inliers: {len(inliers1)})")
    plt.show()