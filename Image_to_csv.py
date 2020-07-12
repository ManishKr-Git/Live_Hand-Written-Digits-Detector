import cv2
import os
import numpy as np
import csv
import glob
import time

label = "3"
path = 'D:\\MCA Study Material\\Coding_Practice\\MachineLearning\\Udemy_Projects\\Hand-Written Digits Recognization\\Images\\Three\\'
imageList = glob.glob(path+'*.png')
for img in imageList:
    im = cv2.imread(img)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)
    roi = cv2.resize(im_gray, (28, 28), interpolation=cv2.INTER_AREA)
    data = []
    data.append(label)
    rows, cols = roi.shape
    # cv2.imshow("window", roi)
    # time.sleep(5)

    # Adding pixels to data array
    for i in range(rows):
        for j in range(cols):
            k = roi[i, j]
            if k > 100:
                k = 1
            else:
                k = 0
            data.append(k)
    f = open("D:\\MCA Study Material\\Coding_Practice\\MachineLearning\\Udemy_Projects\\Hand-Written Digits Recognization\\dataset.csv", 'a')
    writer = csv.writer(f)
    writer.writerow(data)

    cv2.waitKey()
