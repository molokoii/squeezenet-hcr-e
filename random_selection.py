import cv2
import numpy

# Saves 50x50 images in an entire image with a step of 50

path = "c\\02\\049\\"
name = "c02-049"
src = cv2.imread(path + name + ".png")

y, x, _ = src.shape

var = 1
for j in range(0, y, 50):
    for i in range(0, x, 50):
        img = src[j:j+50, i:i+50]
        cv2.imwrite(path + name + "-no-e" + str(var) + ".png", img)
        var += 1
