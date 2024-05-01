
import cv2
import numpy as np
from matplotlib import pyplot as plt
from util.homography_points import get_warpPerspective, warpImg
from util.imageProcessing import  rotate_bbox, rotate_bound,resizeImage, get_angle_and_box_coord
from util.faces import  findFaces, is_two_image_same, siftMatching

def main():
    
    # template = cv2.imread("test/testcard.png")
    # sample = cv2.imread("croppedFaces/crop_face_1.png")
    
    template = cv2.imread(r"test\teguh_ktp2.jpg")
    sample = cv2.imread(r"train\Tumbal1.PNG")
    
    MIN_MATCH_COUNT = 12

    img1 = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)         # trainImage
    img2 = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)           # queryImage
    
    img1 = resizeImage(img1)
    kp1, kp2, good = siftMatching(img1, img2)

    if len(good) >= MIN_MATCH_COUNT:

        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        M2, mask2 = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)
        h,w,_ = img1.shape
    
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        border = cv2.polylines(img2, [np.int32(dst)], True, (0, 255, 0), 3)

        warped_image = get_warpPerspective(img2, M2, dst)
       
        (heigth_q, width_q) = img2.shape[:2]
        (cx, cy) = (width_q // 2, heigth_q // 2)

        angle, box = get_angle_and_box_coord(dst)
        
        rotated_img = rotate_bound(img2, angle)
        
        new_bbox = rotate_bbox(box, cx, cy, heigth_q, width_q, angle)
       
        warp_image = warpImg(rotated_img, new_bbox ,  heigth_q, width_q)
        
        face_crop_img_query = findFaces(warped_image)
        face_crop_img_target = findFaces(img1)
        
        if(img1 is not None):
            plt.title("rotated_image")
            plt.imshow(rotated_img)
            # plt.show()
        
        if(warped_image  is not None):
            plt.title("homography transformed image")
            plt.imshow(warped_image)
            # plt.show()
        
        if(face_crop_img_query is not None):
            plt.title("face_crop")
            plt.imshow(face_crop_img_query)
            # plt.show()
        
        if(face_crop_img_target is not None or face_crop_img_query is not None):
            plt.title("face_crop_target")
            plt.imshow(face_crop_img_target)
            plt.show()
            is_two_image_same(face_crop_img_target, img2, 10)
        # Show Matched Data 
        fig, axes = plt.subplots(1, 2, figsize=(8, 6))
        fig.suptitle("Matched Image")
        # Display the template image
        axes[0].imshow(img1)
        axes[0].set_title('Template Image')
        axes[0].axis('off')
        
        # Display the sample image
        axes[1].imshow(img2)
        axes[1].set_title('Sample Image')
        axes[1].axis('off')
        plt.show()
        
    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        
        # Show Unmatched Data 
        fig, axes = plt.subplots(1, 2, figsize=(8, 6))
        fig.suptitle("Unmatched Image")
        # Display the template image
        axes[0].imshow(img1)
        axes[0].set_title('Template Image')
        axes[0].axis('off')
        
        # Display the sample image
        axes[1].imshow(img2)
        axes[1].set_title('Sample Image')
        axes[1].axis('off')
        plt.show()

if __name__ == '__main__':
    main()