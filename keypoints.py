import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
import numpy as np
import cv2
import random

def detect_corner(img, N_Strong, method="HARRIS"):
    """
    Detects corner points using Harris, Shi-Tomasi, or SIFT.
    Returns:
        - keypoints: List of detected corner points.
        - corner_response: Corner strength map for ANMS.
    """
    # mask = mask = np.ones(gray.shape, dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    corner_response = None  # Initialize response map

    if method.upper() == "HARRIS":
        corner_response = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        keypoints = np.argwhere(corner_response > 0.01 * corner_response.max())[:N_Strong]
    elif method.upper() == "SHI-TOMASI":
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=N_Strong, qualityLevel= 0.01, minDistance=5)
        keypoints = corners.reshape(-1, 2).astype(np.intp)
        corner_response = np.zeros_like(gray, dtype=np.float32)  # Create empty response map
        for x, y in keypoints:
            corner_response[y, x] = 1  # Mark detected keypoints
    elif method.upper() == "SIFT":
        sift = cv2.SIFT_create()
        keypoints = sift.detect(gray, None)
        keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:N_Strong]
         # Convert `cv2.KeyPoint` objects to NumPy (x, y) array
        keypoints = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.intp)

        # Create an empty response map for consistency
        corner_response = np.zeros_like(gray, dtype=np.float32)
        for x, y in keypoints:
            corner_response[y, x] = 1  # Mark detected keypoints
    else:
        raise  ValueError("Invalid method. Choose 'HARRIS', 'SHI-TOMASI', or 'SIFT'.")
    return keypoints, corner_response

def anms(keypoints, N_Best = 500):
    """Applies Adaptive Non-Maximal Suppression (ANMS)."""
    if len(keypoints) < N_Best:
        return keypoints
    keypoints = np.array(keypoints, dtype=np.float32)
    r = np.full(len(keypoints), np.inf)

    for i in range(len(keypoints)):
        for j in range(len(keypoints)):
            dist = np.linalg.norm(keypoints[i] - keypoints[j])
            if dist < r[i]:
                r[i] = dist

    best_indices = np.argsort(r)[-N_Best:]
    return keypoints[best_indices]

def feature_desc(img, keypoints, patch_size=40):
    """Extracts feature descriptors from keypoints."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    desc = []
    valid_keypoints = []  # To store only keypoints that have valid descriptors
    half_patch = patch_size // 2

    for x, y in keypoints:
        if x - half_patch < 0 or y - half_patch < 0 or x + half_patch >= img.shape[1] or y + half_patch >= img.shape[0]:
            continue  # Skip keypoints too close to edges

        patch = img[int(y-half_patch):int(y+half_patch), int(x-half_patch):int(x+half_patch)]
        patch = cv2.resize(cv2.GaussianBlur(patch, (5, 5), sigmaX=1), (8, 8))
        feature_vector = (patch.flatten() - np.mean(patch)) / (np.std(patch) + 1e-7)

        desc.append(feature_vector)
        valid_keypoints.append([x, y])  # Store only valid keypoints

    return np.array(desc), np.array(valid_keypoints)
