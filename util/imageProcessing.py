import cv2
import numpy as np

# Get the angle and box coooard
def get_angle_and_box_coord(dst):
# (center(x,y), (width, height), angle of rotation) = cv2.minAreaRect(c)
    rect = cv2.minAreaRect(dst)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Retrieve the key parameters of the rotated bounding box
    box_center = (int(rect[0][0]), int(rect[0][1]))
    box_width = int(rect[1][0])
    box_height = int(rect[1][1])
    angle = int(rect[2])
    
    if box_width < box_height:
        angle = 90 - angle
    else:
        angle = -angle
    
    print("Rotation Angle:" + str(angle) +"degress")
    return -angle, box


def rotate_bbox(bb,cx,cy,h,w,theta):
    new_bb = np.zeros_like(bb)     
    for i, coord in enumerate(bb):
        # Calculate the standard transformation matriks
        M = cv2.getRotationMatrix2D((cx,cy),theta, 1.0)
        #Grab the rotation components of the matriks
        cos = np.abs(M[0,0])
        sin = np.abs(M[0,1])
        #Compute new bounding dimensions of the image
        nW = int((h*sin) + (w*cos))
        nH = int((h*cos) + (w*sin))
        #Adjust the rotation matriks to take into account translation
        M[0,2] += (nW / 2) - cx
        M[0,1] += (nH / 2) - cy
        #Prepare the vector to be transform
        v = [coord[0, coord[1], 1]]
        #Perform the actual rotation and return the image
        calculated = np.dot(M, v)
        new_bb[i] = (calculated[0] / calculated[1])
    return new_bb


def rotate_bound(image,angle):
    # grab the dimensions of the image and then determine the center
    (h,w) = image.shape[:2] #0,1
    (cX, cY) = (w // 2, h // 2) #1,2

    # grab the rotation matriks (applying the negative of the angle to rotate clockwise)
    # Then grab the sin and cos 
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])

    #compute the new bounding dimensions of the image
    nW = int((h*sin) + (w*cos))
    nH = int((h*cos) + (w*sin))

    #adjust the rotation matriks to take into account translation
    M[0,2] += (nW / 2) - cX
    M[1,2] += (nH / 2) - cY

    # Perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def applyBlur(image):
    return cv2.blur(image,(3,3))

def resizeImage(image):
    h,w = image.shape[:2]
    return cv2.resize(image, (w+100, h+100), cv2.INTER_LINEAR)