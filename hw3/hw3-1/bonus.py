import numpy as np
import cv2

print('Running bonus...')

cap = cv2.VideoCapture('input/ar_marker.mp4')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
new_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 5
new_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 5
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('bonus.mp4',fourcc, fps, (new_width,new_height))

MIN_MATCH_COUNT = 10
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

img1 = cv2.imread('input/marker.png')
kp1, des1 = sift.detectAndCompute(img1,None)

my_img = cv2.imread('input/mona_lisa.jpg')

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
#         frame = cv2.flip(frame,0)
        img2 = cv2.resize(frame, (new_width, new_height))
        # find the keypoints and descriptors with SIFT        
        kp2, des2 = sift.detectAndCompute(img2,None)
        matches = flann.knnMatch(des1,des2,k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        
        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
           
            my_img_warped = cv2.warpPerspective(my_img, M, (img2.shape[1],img2.shape[0]))
            overlaid = np.where(my_img_warped != 0, my_img_warped, img2)
            out.write(overlaid)   
        else: # write original frame
            print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
            out.write(frame)   
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
print('DONE!')
