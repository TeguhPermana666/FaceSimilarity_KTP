import cv2 
import numpy as np
def get_homography_points(M, points):
    new_bb = np.zeros_like(points)
    
    for i, coord in enumerate(points):
        v = [coord[0, coord[1], 1]]
        calculated = np.dot(M, v)
        calculated_scaled = calculated / calculated[2]
        new_bb[i] (calculated_scaled[0], calculated_scaled[1])

def reorder(myPoints):
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4,2))
    add = myPoints.sum(1)
    # Take the minimum point
    myPointsNew[0] = myPoints[np.argmin(add)]
    # Take the maximum point
    myPointsNew[3]  = myPoints[np.argmax(add)]

    # Calculate the different data each column
    diff = np.diff(myPoints, axis = 1)
    
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    
    return myPointsNew
    
def warpImg(img, poitns, w, h):
    poitns = reorder(poitns)
    pts1 = np.float32(poitns)
    pts2 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w,h))
    return imgWarp

def get_warpPerspective(img, M,dst):
    img_homog = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    dst_h = dst.reshape(-1, 1, 2)
    # Get the Homography points
    new_bb = get_homography_points(M, dst_h)
    warp_image = warpImg(img_homog, new_bb, img.shape[0], img.shape[1])
    return warp_image


    