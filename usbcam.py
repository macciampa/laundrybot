import cv2

cam = cv2.VideoCapture(0)

cam.set(3,640)
cam.set(4,360)

while True:
	ret, image = cam.read()
	cv2.imshow('Imagetest',image)
	k = cv2.waitKey(1)
	if k != -1:
		break
cv2.imwrite('image1.jpg', image)
cam.release()
cv2.destroyAllWindows()
