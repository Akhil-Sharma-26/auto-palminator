import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)  # Chooosing the webcam we're going to use,

mpHands = mp.solutions.hands
hands = mpHands.Hands()  # Default parameters are already given here
mpDrw = mp.solutions.drawing_utils  # Draws the points on the hand
pTime = 0
cTime = 0
while True:
    success, img = cap.read()  # Give success as cap in reading
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)  # Shows coordinates on the console
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h) # extracting some node of the wireframe over the hand
                print(id, cx, cy)
                if id == 5:
                    cv2.circle(img, (cx,cy) , 20, (255,0,255) , cv2.FILLED)
            mpDrw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  # HandConnection draws lines, connecting dots
    cTime = time.time()  # gives us the current time
    fps = 1/(cTime - pTime)  # logic for fps
    pTime = cTime

    cv2.putText(img, str(int(fps)), (5, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    cv2.imshow("Image", img)  # For Image showing
    cv2.waitKey(1)
