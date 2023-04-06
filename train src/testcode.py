import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import pyttsx3
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("model\converted_keras\keras_model.h5", "model\converted_keras\labels.txt")
engine = pyttsx3.init()

offset = 20
imgSize = 300

folder = "Data/0"
counter = 0

labels = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 4)
        
        fingers = detector.fingersUp(hand)
        if detector.fingersUp(hand) == [0, 0, 0, 0, 0]:
            engine.say("a")
            engine.runAndWait()
        if detector.fingersUp(hand) == [1, 0, 0, 0, 0]:
            engine.say("b")
            engine.runAndWait()
        if detector.fingersUp(hand) == [1, 1, 0, 0, 0]:
            engine.say("c")
            engine.runAndWait()
        if detector.fingersUp(hand) == [1, 1, 1, 0, 0]:
            engine.say("d")
            engine.runAndWait()
        if detector.fingersUp(hand) == [1, 1, 1, 1, 0]:
            engine.say("e")
            engine.runAndWait()
        if detector.fingersUp(hand) == [1, 1, 1, 1, 1]:
            engine.say("f")
            engine.runAndWait()
        if detector.fingersUp(hand) == [0, 1, 0, 0, 0]:
            engine.say("g")
            engine.runAndWait()
        if detector.fingersUp(hand) == [0, 1, 1, 0, 0]:
            engine.say("h")
            engine.runAndWait()
        if detector.fingersUp(hand) == [0, 1, 1, 1, 0]:
            engine.say("i")
            engine.runAndWait()
        if detector.fingersUp(hand) == [0, 1, 1, 1, 1]:
            engine.say("j")
            engine.runAndWait()
        if detector.fingersUp(hand) == [0, 0, 1, 1, 0]:
            engine.say("k")
            engine.runAndWait()
        if detector.fingersUp(hand) == [0, 0, 1, 1, 1]:
            engine.say("l")
            engine.runAndWait()
        if detector.fingersUp(hand) == [0, 0, 0, 1, 0]:
            engine.say("m")
            engine.runAndWait()
        if detector.fingersUp(hand) == [0, 0, 0, 1, 1]:
            engine.say("n")
            engine.runAndWait()
        if detector.fingersUp(hand) == [0, 0, 0, 0, 1]:
            engine.say("o")
            engine.runAndWait()
        if detector.fingersUp(hand) == [1, 1, 0, 0, 1]:
            engine.say("p")
            engine.runAndWait()
        if detector.fingersUp(hand) == [1, 1, 1, 0, 1]:
            engine.say("q")
            engine.runAndWait()
        if detector.fingersUp(hand) == [0, 1, 0, 0, 1]:
            engine.say("r")
            engine.runAndWait()
        if detector.fingersUp(hand) == [0, 1, 1, 0, 1]:
            engine.say("s")
            engine.runAndWait()
        if detector.fingersUp(hand) == [0, 0, 1, 0, 1]:
            engine.say("t")
            engine.runAndWait()
        if detector.fingersUp(hand) == [1, 0, 1, 0, 0]:
            engine.say("u")
            engine.runAndWait()
        if detector.fingersUp(hand) == [1, 0, 1, 1, 0]:
            engine.say("v")
            engine.runAndWait()
        if detector.fingersUp(hand) == [1, 0, 1, 1, 1]:
            engine.say("w")
            engine.runAndWait()
        if detector.fingersUp(hand) == [0, 1, 0, 1, 0]:
            engine.say("x")
            engine.runAndWait()
        if detector.fingersUp(hand) == [1, 0, 0, 0, 1]:
            engine.say("y")
            engine.runAndWait()
        if detector.fingersUp(hand) == [0, 1, 0, 1, 1]:
            engine.say("z")
            engine.runAndWait()
        
        
       


       #cv2.imshow("ImageCrop", imgCrop)
       #cv2.imshow("ImageWhite", imgWhite)  

    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) == ord('q'):
        break
    
        