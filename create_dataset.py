from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np
import math
import time
import tensorflow as tensorflow

cap = cv2.VideoCapture(0) 
# to turn on video camerassssssss
detector = HandDetector(maxHands=2)
# to ensure we only capture one hand at a time

spacing = 20
# to give our hands an offset at the sides to not miss any movement

imgWhiteSize = 300
# to ensure our han ds are the correct size and always a square, making classification easier.

folder = "Data/Help"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgWhiteSize, imgWhiteSize, 3), np.uint8) * 250
        imgCrop = img[y-spacing:y + h + spacing, x - spacing:x + w + spacing]
        imgCropShape = imgCrop.shape


        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgWhiteSize / h
            wCalculated = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCalculated,imgWhiteSize))
            imgResizeShape = imgResize.shape
            widthGap = math.ceil((imgWhiteSize-wCalculated)/2)
            imgWhite[:, widthGap:wCalculated+widthGap] = imgResize

        else:
            k = imgWhiteSize / w
            hCalculated = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgWhiteSize,hCalculated))
            imgResizeShape = imgResize.shape
            heightGap = math.ceil((imgWhiteSize-hCalculated)/2)
            imgWhite[heightGap:hCalculated + heightGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImgWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)





    success, img = cap.read()
    img = cv2.resize(img, (300,300))
    image = np.expand_dims(img,axis=0)
    image = image/255



success, img = cap.read()   
    
    hands, image2 = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgWhiteSize, imgWhiteSize, 3), np.uint8) * 250
        imgCrop = img[y-spacing:y + h + spacing, x - spacing:x + w + spacing]
        
        imgCropShape = imgCrop.shape
        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgWhiteSize / h
            wCalculated = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCalculated,imgWhiteSize))
            imgResizeShape = imgResize.shape
            widthGap = math.ceil((imgWhiteSize-wCalculated)/2)
            imgWhite[:, widthGap:wCalculated+widthGap] = imgResize

        else:
            k = imgWhiteSize / w
            hCalculated = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgWhiteSize,hCalculated))
            imgResizeShape = imgResize.shape
            heightGap = math.ceil((imgWhiteSize-hCalculated)/2)
            imgWhite[heightGap:hCalculated + heightGap, :] = imgResize


            
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    image = cv2.resize(img, (300,300))
    image = np.expand_dims(image,axis=0)
    image = img/255
    cv2.imshow("Image", image)
    
    #predictions = model.predict(image)
    #class_label= np.argmax(predictions)
   
    #cv2.putText(img, f'Class:{folder[class_label]}', (10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow("Img", img)
    cv2.imshow("Image2", image2)



    image = cv2.resize(img, (300,300))
    image = np.expand_dims(img,axis=0)
    image = image/255