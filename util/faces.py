import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt
def findFaces(image):
    detector = dlib.get_frontal_face_detector() 
    faces = detector(image)
    num_of_faces = len(faces)
    print("Number of Faces:", num_of_faces)
    if (not num_of_faces):
        return None
    
    for face in faces:
        x1 = face.left() - 30
        y1 = face.top() - 70
        x2 = face.right() + 10
        y2 = face.bottom() + 30
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        face_crop = image[y1:y2, x1:x2]
        return face_crop

def cropFaceRegions(image,x1,y1,x2,y2):
    face_crop = image[y1:y2, x1:x2]
    plt.imsave("croppedFaces/crop_face.png", face_crop)
    return face_crop



def is_two_image_same(img1,img2, face_match_count):
    
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm  =FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2, k=2)

    #Store all the good matches as per Lowe's ratio test
    good = []
    for m,n in matches:
        if m.distance < n.distance:
            good.append.m
    print("Total good matches", len(good))
    good = good[:face_match_count]
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,flags=2)
    plt.title('Face Matches')
    plt.imshow(img3, 'gray'), plt.show()
    print("Matches are found - %d/%d" % (len(good),face_match_count))

    if len(good) > face_match_count:
        print("Faces are similar")
        return True
    else:
        print("Faces are not similar")
        return False

def siftMatching(img1, img2):
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance <= 0.70*n.distance:
            good.append(m)
    print("Total good matches:", len(good))       
    # good = good[:20]
    return kp1, kp2, good