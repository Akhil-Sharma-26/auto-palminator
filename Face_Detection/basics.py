import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)  # Chooosing the webcam we're going to use
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
face = mpFaceDetection.FaceDetection(0.75) # just like hand tracker # we can also cahnge the min confidence
pTime = 0
cTime = 0
while True:
    success, img = cap.read()  # Give success as cap in reading
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face.process(imgRGB)
    # print(results)
    if results.detections:
        for id, detections in enumerate(results.detections):
            # mpDraw.draw_detection(img,detections)
            # print(id,detections)
            # print(detections.score, detections.location_data.relative_bounding_box)
            bboxC = detections.location_data.relative_bounding_box #bounding box from class
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img,bbox,(0,255,0),2)
            cv2.putText(img, f'{(int(detections.score[0]*100))}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)




    cTime = time.time()  # gives us the current time
    fps = 1 / (cTime - pTime)  # logic for fps
    pTime = cTime
    cv2.putText(img, f'FPS: {str(int(fps))}', (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
    cv2.imshow("Image", img)  # For Image showing
    cv2.waitKey(1)