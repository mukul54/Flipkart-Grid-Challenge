import cv2

x1, x2, y1, y2 = 78, 446,83,422  # given co-ordinates for the image JPEG_20160517_140621_1000651031832.png  

a1, a2, b1, b2 = 65, 460, 70, 410 # our prediction on the image JPEG_20160517_140621_1000651031832.png

image = cv2.imread('/home/mukul/Desktop/machine_learning/flipKartGrid/JPEG_20160517_140621_1000651031832.png')

cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0), thickness = 2)

cv2.rectangle(image,(a1,b1),(a2,b2),(0,0,255), thickness=2)

cv2.imwrite('pred_det.png',image)

cv2.imshow('image',image)

cv2.waitKey(0)

cv2.destroyAllWindows()
