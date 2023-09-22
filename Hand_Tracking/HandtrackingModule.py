# import cv2
# import mediapipe as mp
# import time
#
#
# class handDetec():
#     def __init__(self, mode=False, maxHands=2, complexity=1, detectioncon=0.5, trackingco=0.5):
#         self.mode = mode
#         self.maxHands = maxHands
#         self.complexity = complexity
#         self.detectioncon = detectioncon
#         self.trackingco = trackingco
#
#         self.mpHands = mp.solutions.hands
#         self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectioncon,
#                                         self.trackingco)  # Default parameters are already given here
#         self.mpDrw = mp.solutions.drawing_utils
#
#     def findHands(self, img, draw=True):
#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         self.results = self.hands.process(imgRGB)
#         # print(results.multi_hand_landmarks)  # Shows coordinates on the console
#         if self.results.multi_hand_landmarks:
#             for handLms in self.results.multi_hand_landmarks:
#                 if draw:
#                     self.mpDrw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS,
#                           self.mpDrw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2), # change the color here for DOTS
#                           self.mpDrw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)) # change color here for LINES
#
#         return img
#     def findPosition(self, img, handNo=0, draw=True):
#         lmList = []
#         if self.results.multi_hand_landmarks:
#             myHand = self.results.multi_hand_landmarks[handNo]
#             for id, lm in enumerate(myHand.landmark):  # what's enumerate??
#                 # print(id, lm)
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)  # extracting some node of the wireframe over the hand
#                 # print(id, cx, cy)
#                 lmList.append(id ,cx,  cy)
#                 if draw:
#
#                 # if id == 5:
#                     cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
#         return lmList
# def main():
#     pTime = 0
#     cTime = 0
#     cap = cv2.VideoCapture(0)
#     detector = handDetec()
#     while True:
#         success, img = cap.read()
#         img = detector.findHands(img)
#         lmList = detector.findPosition(img)
#         if len(lmList) !=0:
#             print(lmList[4])
#         cTime = time.time()  # gives us the current time
#         fps = 1 / (cTime - pTime)  # logic for fps
#         pTime = cTime
#
#         cv2.putText(img, str(int(fps)), (5, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
#         cv2.imshow("Image", img)  # For Image showing
#         cv2.waitKey(1)
#
#
# if __name__ == "__main__":
#     main()

# ! 640x480 image
# 20-30fps
import cv2
import mediapipe as mp
import time


class handDetec():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectioncon=0.5, trackingco=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectioncon = detectioncon
        self.trackingco = trackingco

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.complexity, self.detectioncon,
                                        self.trackingco)
        self.mpDrw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDrw.draw_landmarks(img, handLms,
                                              self.mpHands.HAND_CONNECTIONS,
                                              self.mpDrw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                                              self.mpDrw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2))
        return img

    def findPosition(self, img, handNo=0):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id ,cx,  cy])
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmList

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetec()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) >= 5:
            print(lmList[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (5, 70), cv2.FONT_HERSHEY_PLAIN,
                    2,(255, 0, 255), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()

