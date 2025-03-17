from img_utils import load_img, show_img, cylindrical_projection
from keypoints import detect_corner, anms, feature_desc
from matching import feature_matching, RANSAC
from warp_blend import warp_and_blend

img1 = load_img(["images/image1.jpg"])[0]
img2 = load_img(["images/image2.jpg"])[0]

img1 = cylindrical_projection(img1, 700)
img2 = cylindrical_projection(img2, 700)

stitched = warp_and_blend(img1, img2, RANSAC(...))

show_img(stitched, "Final Panorama")