import cv2
import mediapipe as mp
from HandTrackingMinimum import HandDetection

wCam = 1280
hCam = 720

webcam = cv2.VideoCapture()
webcam.set(3, wCam)
webcam.set(4, hCam)
webcam.open(0, cv2.CAP_DSHOW)

hand = HandDetection()

while True:
    status, frame = webcam.read()
    frame = cv2.flip(frame, 1)
    handLandmarks = hand.findHandLandmarks(image=frame, draw=True)
    cv2.imshow("hand Landmark", frame)
    if cv2.waitKey(1) == ord("q"):
        break
cv2.destroyAllWindows
webcam.release()
