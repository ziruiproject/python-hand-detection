import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands # type: ignore
mpDraw = mp.solutions.drawing_utils # type: ignore

class HandDetection:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.hands = mpHands.Hands(max_num_hands = max_num_hands,
                                   min_detection_confidence = min_detection_confidence,
                                   min_tracking_confidence = min_tracking_confidence
                                   )
    def findHandLandmarks(self, image, handNumber=0, draw=False):
        originalImage = image
        Image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        results = self.hands.process(image)

        landMarkList = []

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[handNumber]
            for id, landmark in enumerate(hand.landmark):
                imgHeigh, imgWidth, imgChannel = originalImage.shape
                xPos, yPos = int(landmark.x*imgWidth), int(landmark.x*imgHeigh)
                landMarkList.append([id, xPos, yPos])
            if draw:
                mpDraw.draw_landmarks(originalImage, hand, mpHands.HAND_CONNECTIONS)
        return landMarkList

