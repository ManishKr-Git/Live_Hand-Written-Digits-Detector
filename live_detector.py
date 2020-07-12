import joblib
import pyscreenshot as ss
import time
import cv2

model = joblib.load(
    "D:\\MCA Study Material\\Coding_Practice\\MachineLearning\\Udemy_Projects\\Hand-Written Digits Recognization\\model.lib")

image_folder = 'D:\\MCA Study Material\\Coding_Practice\\MachineLearning\\Udemy_Projects\\Hand-Written Digits Recognization\\Images\\'
for i in range(15):
    time.sleep(5)
    im = ss.grab(bbox=(15, 152, 180, 332))
    print("saved.........", i)
    im.save(image_folder+'testing_image.png')
    im = cv2.imread(image_folder+'testing_image.png')
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)

    # Threshold the image
    ret, im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY)
    roi = cv2.resize(im_th, (28, 28), interpolation=cv2.INTER_AREA)
    cv2.imwrite(image_folder+'segmented.png', roi)
    rows, cols = roi.shape
    data = []
    for _ in range(rows):
        for j in range(cols):
            k = roi[_, j]
            if k >= 100:
                k = 1
            else:
                k = 0
            data.append(k)
    print(model.predict([data]))
    print("clear screen")
