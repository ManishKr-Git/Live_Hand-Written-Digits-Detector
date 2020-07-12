import pyscreenshot as ss
import time
image_folder = 'D:\\MCA Study Material\\Coding_Practice\\MachineLearning\\Udemy_Projects\\Hand-Written Digits Recognization\\Images\\Three\\'
for i in range(30):
    time.sleep(5)
    im = ss.grab(bbox=(6, 152, 184, 336))
    print("saved.........", i)
    im.save(image_folder+str(i)+'.png')
    print("Clear screen")
