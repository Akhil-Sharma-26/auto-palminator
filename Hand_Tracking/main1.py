import cv2
import mediapipe as mp
import time
import HandtrackingModule as hmodule

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = hmodule.handDetec()
while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=False)  # To draw/not the lines
    lmList = detector.findPosition(img, draw=False)  # To draw/not the specific nodes2624
    if len(lmList) >= 5:
        print(lmList[4])
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (5, 70), cv2.FONT_HERSHEY_PLAIN,
                2, (255, 0, 255), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
