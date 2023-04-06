import cv2
from flask import Flask, request, jsonify
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import pyttsx3
import math

app = Flask(__name__)

detector = HandDetector(maxHands=1)
classifier = Classifier("model\converted_keras\keras_model.h5", "model\converted_keras\labels.txt")
engine = pyttsx3.init()

offset = 20
imgSize = 300

labels = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

@app.route('/', methods=['GET'])
def gesture_recognition():
    # Get the webcam input
    cap = cv2.VideoCapture(0)

    # Read frames from the webcam and process them
    while True:
        ret, img = cap.read()

        # Get the hand detection results
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

            output = {
                "prediction": labels[index]
            }

        else:
            output = {
                "prediction": "No hands detected"
            }

        # Release the resources
        cap.release()

        return jsonify(output)

if __name__ == '__main__':
    app.run()
