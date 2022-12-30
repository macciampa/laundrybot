import cv2
import numpy as np
import imutils
 
cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,360)
res_height = 320                         # Resolution of camera (height)

# Initial framerate value
fps = 0

# Start webcam
while True:
    
    # Get timestamp for calculating actual framerate
    timestamp = cv2.getTickCount()

    ret, image = cam.read()

    # Crop
    img = image[20:340, 160:480]

    # Convert to grayscale
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Edge detection
    img = cv2.GaussianBlur(img, (7,7), 0) 
    img = cv2.Canny(img,80,200)

    # Corner Detection
    img = np.float32(img)
    dst = cv2.cornerHarris(img,30,13,0.04)
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    img[dst>0.1*dst.max()]=[0,0,255]

    # Blob Centroid Detection
    img_corners = np.zeros(img.shape, np.float32)
    img_corners = cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB)
    img_corners[dst>0.1*dst.max()]=[0,0,255]
    img_corners = np.uint8(img_corners)
    img_corners = cv2.cvtColor(img_corners, cv2.COLOR_BGR2GRAY)
    img_corners = cv2.GaussianBlur(img_corners, (5, 5), 0)
    img_corners = cv2.threshold(img_corners, 60, 255, cv2.THRESH_BINARY)[1]
    # find contours in the thresholded image
    cnts = cv2.findContours(img_corners.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    bl = (img.shape[0], 0)
    tl = (img.shape[0], img.shape[1])
    br = (0, 0)
    tr = (0, img.shape[1])
    # loop over the contours, keep bottom left and top left coordinates
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        # Find corners
        if (cX + (img.shape[1]-cY)) < (bl[0] + (img.shape[1]-bl[1])):
            bl = (cX, cY)
        if (cX + cY) < (tl[0] + tl[1]):
            tl = (cX, cY)
        if (cX + cY) > (br[0] + br[1]):
            br = (cX, cY)
        if ((img.shape[0]-cX) + cY) < (img.shape[0]-(tr[0]) + tr[1]):
            tr = (cX, cY)

    cv2.circle(img, bl, 7, (0, 0, 255), -1)
    cv2.circle(img, tl, 7, (0, 0, 255), -1)
    cv2.circle(img, br, 7, (0, 0, 255), -1)
    cv2.circle(img, tr, 7, (0, 0, 255), -1)

    # Draw framerate on frame
    cv2.putText(img, 
                "FPS: " + str(round(fps, 2)), 
                (0, 12),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255,255,255))
    
    # Calculate framerate
    frame_time = (cv2.getTickCount() - timestamp) / cv2.getTickFrequency()
    fps = 1 / frame_time

    # Show the frame
    cv2.imshow('Imagetest',img)
    k = cv2.waitKey(1)
    if k != -1:
        break

# End
#cv2.imwrite('image1.jpg', image)
cam.release()
cv2.destroyAllWindows()
