import cv2
from matplotlib import pyplot as plt
import numpy

def rotateImage(image, angle):
    h, w = image.shape[:2]
    center = w // 2, h // 2
    rotationMatrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotatedImage = cv2.warpAffine(image, rotationMatrix, (w, h))
    cv2.imwrite('rotatedImage.jpg', rotatedImage)
    return rotatedImage

def addBorderToImage(image):
    h, w = image.shape[:2]
    borderedImage = cv2.copyMakeBorder(image, w // 4, w // 4, h // 4, h // 4, cv2.BORDER_CONSTANT)
    return borderedImage

def colorImageWithMask(image, mask):
    image[:, :, 2] = numpy.maximum(image[:, :, 2], mask)
    return image

img = cv2.imread('/home/timmy/python/crater/images/craters1.jpg')
img = addBorderToImage(img)
img = rotateImage(img, -43)
 

h, w = img.shape[:2]
img  = img[ 0 : 100 * h // 100, 51 * w // 100 : 75 * w // 100]
cv2.imwrite('rotated.jpg' , img)

b, g, r = cv2.split(img)

green_gauss = cv2.GaussianBlur(g, (7, 7), 0)

g_eq = cv2.equalizeHist(green_gauss)

plt.hist(g_eq.ravel(),100,[0,256])
# plt.savefig('green_histogram.png')

_, inv_bin = cv2.threshold(g_eq, 15, 255, cv2.THRESH_BINARY_INV)
_, bin = cv2.threshold(g_eq, 15, 255, cv2.THRESH_BINARY)

maskedImage = colorImageWithMask(img, inv_bin)

# blob detection
params = cv2.SimpleBlobDetector_Params()

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio  = 0.05
# Filter by Area
params.filterByArea = True
params.maxArea = 1000
params.minArea = 0

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(bin)
print("Craters: " + str(len(keypoints)))
im_with_keypoints = cv2.drawKeypoints(img, keypoints, numpy.zeros((1, 1)), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
final = cv2.putText(im_with_keypoints, "Craters: " + str(len(keypoints)), (10 ,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
cv2.imwrite('blobs.jpg' , final)
