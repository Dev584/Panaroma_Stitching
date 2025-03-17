from img_utils import load_img, show_img, cylindrical_projection, plot_matches, plot_after_ransac
from keypoints import detect_corner, anms, feature_desc
from matching import feature_matching, RANSAC
from warp_blend import warp_and_blend
import cv2
import numpy as np
import os

def main():
    # User-defined parameters
    focal_length = int(input("Enter focal length: "))  # Example: 700
    N_Strong = int(input("Enter number of strong keypoints: "))  # Example: 1000
    method = input("Enter keypoint detection method (HARRIS/SHI-TOMASI/SIFT): ").upper()
    N_Best = int(input("Enter number of best keypoints after ANMS: "))  # Example: 500

    image_dir = "images"
    if not os.path.exists(image_dir):
        print(f"Error: The '{image_dir}' folder does not exist. Please create it and add images.")
        return

    # Load images from 'images' folder
    image_paths = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.lower().endswith(('.jpg', '.png', '.jpeg'))])

    if len(image_paths) < 2:
        print("Error: At least two images are required for stitching.")
        return
    
    # Ensure left-most image is first (Sort based on X-coordinates if possible)
    img_objects = [cv2.imread(img, cv2.IMREAD_COLOR) for img in image_paths]
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_objects]
    sift = cv2.SIFT_create()
    keypoints_list = [sift.detect(gray, None) for gray in gray_images]
    x_means = [np.mean([kp.pt[0] for kp in keypoints]) for keypoints in keypoints_list]

    # Sort images based on mean X-coordinate of keypoints
    sorted_indices = np.argsort(x_means)
    image_paths = [image_paths[i] for i in sorted_indices]
    
    img1, img2 = load_img(image_paths[:2])

    show_img(img1, "Image 1")
    show_img(img2, "Image 2")

    # Apply Cylindrical Projection
    img1 = cylindrical_projection(img1, focal_length)
    img2 = cylindrical_projection(img2, focal_length)

    # Detect Keypoints
    keypoints1, _ = detect_corner(img1, N_Strong, method)
    keypoints2, _ = detect_corner(img2, N_Strong, method)

    # Apply ANMS
    keypoints1 = anms(keypoints1, N_Best)
    keypoints2 = anms(keypoints2, N_Best)

    # Extract Feature Descriptors
    descriptors1, keypoints1 = feature_desc(img1, keypoints1)
    descriptors2, keypoints2 = feature_desc(img2, keypoints2)

    # Match Features
    matches = feature_matching(descriptors1, descriptors2, thershold=0.3)

    # Show the number of matches found
    print(f"Found {len(matches)} feature matches.")

    # Convert matched keypoints for RANSAC
    match_kp1 = np.array([keypoints1[m[0]] for m in matches])
    match_kp2 = np.array([keypoints2[m[1]] for m in matches])

    # Estimate Homography using RANSAC
    H, inliers1, inliers2 = RANSAC(match_kp1, match_kp2, iter=2000, threshold=3)

    # Call the function to plot feature matches
    plot_matches(img1, keypoints1, img2, keypoints2, matches)

    if H is not None:
        print(f"RANSAC found {len(inliers1)} inliers and computed the best Homography matrix.")

        # PLot matches after RANSAC
        plot_after_ransac(img1, img2, inliers1, inliers2)
        # Test Warp and Blend**
        print("Warping and blending images...")

        # Step 8: Warp and Blend Images
        stitched_image = warp_and_blend(img1, img2, H)

        # Step 9: Display and Save Final Panorama
        show_img(stitched_image, "Final Panorama")
        cv2.imwrite("stitched_panorama.jpg", stitched_image)
    else:
        print("No valid Homography found.")

    

if __name__ == "__main__":
    main()