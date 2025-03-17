import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
import numpy as np
import cv2
import random

def feature_matching(desc1, desc2, thershold=0.3):
    matches = []
    for i, d1 in enumerate(desc1):
        ssd = np.sum((desc2 - d1) ** 2, axis=1)
        best_match = np.argmin(ssd)
        second_best = np.argsort(ssd)[1]
        if ssd[best_match] < thershold * ssd[second_best]:
            matches.append((i, best_match))
    return matches

def est_homography(src_pts, dst_pts):
    """
    Estimates the homography matrix H using Direct Linear Transform (DLT).
    """
    H, _ = cv2.findHomography(src_pts, dst_pts, method=0)  # Use exact method
    return H

def apply_homography(H, pts):
    """
    Applies the homography matrix H to a set of points.
    """
    pts_homog = np.hstack((pts, np.ones((pts.shape[0], 1))))  # Convert to homogeneous coordinates
    transformed_pts = (H @ pts_homog.T).T
     # Avoid division by zero (check last coordinate)
    transformed_pts[:, 2] = np.where(transformed_pts[:, 2] == 0, 1e-7, transformed_pts[:, 2])

    transformed_pts = transformed_pts[:, :2] / transformed_pts[:, 2, np.newaxis]  # Convert back to 2D
    return transformed_pts

def RANSAC(match_kp1, match_kp2, iter=1000, threshold=4):
    """
    Performs RANSAC to estimate the best homography matrix H.

    Parameters:
        match_kp1: Keypoints from image 1
        match_kp2: Corresponding keypoints from image 2
        iter: Number of iterations
        N: Minimum inliers required to consider a good model
        thr: Distance threshold for inliers

    Returns:
        H_best: Best homography matrix found
        inliers1: Inliers from image 1
        inliers2: Inliers from image 2
    """

    if len(match_kp1) < 4 or len(match_kp2) < 4:
        print(f"Error: Not enough matches for RANSAC. Found only {len(match_kp1)} matches.")
        return None, None, None  # Not enough points for homography estimation
    
    best_H = None
    max_inlier = 0
    inliers1, inliers2 = None, None

    for _ in range(iter):
        # Randomly Select 4 pairs
        idx = random.sample(range(len(match_kp1)), 4)
        src_pts = np.float32([match_kp1[i] for i in idx]).reshape(-1, 1, 2)
        dst_pts = np.float32([match_kp2[i] for i in idx]).reshape(-1, 1, 2)

        # Estimate Homography using these points
        H = est_homography(src_pts, dst_pts)

        if H is None:
            continue

        # Apply the homography to all keypoints
        transformed_pts = apply_homography(H, np.array(match_kp1))

        # Compute SSD distance and find inliers
        distances = np.linalg.norm(transformed_pts - np.array(match_kp2), axis=1)
        inlier_idx = np.where(distances < threshold)[0]

        # Update best homography if we found more inliers
        if len(inlier_idx) > max_inlier:
            max_inlier = len(inlier_idx)
            best_H = H
            inliers1 = [match_kp1[i] for i in inlier_idx]
            inliers2 = [match_kp2[i] for i in inlier_idx]

        # Stop early if we have 90% inliers
        if len(inlier_idx) > 0.9 * len(match_kp1):
            break

    if best_H is None:
        print("RANSAC failed: No good homography found.")
        return None, None, None
    
    # Recompute final Homography using all inliers
    final_H = est_homography(np.float32(inliers1).reshape(-1, 1, 2), np.float32(inliers2).reshape(-1, 1, 2))

    return final_H, inliers1, inliers2