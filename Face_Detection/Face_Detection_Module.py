import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self,minDetConf=0.75):
        self.minDetectionConf = minDetConf
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.face = self.mpFaceDetection.FaceDetection(self.minDetectionConf)
    def findFaces(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face.process(imgRGB)
        bboxes = []
        if self.results.detections:
            for id, detections in enumerate(self.results.detections):
                bboxC = detections.location_data.relative_bounding_box  # bounding box from class
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                bboxes.append([id,bbox,detections.score])
                if draw:
                    self.fancyDraw(img,bbox)
                # cv2.rectangle(img, bbox, (0, 255, 0), 2)
                    cv2.putText(img, f'{(int(detections.score[0] * 100))}%', (bbox[0], bbox[1] - 20),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        return img, bboxes
    def fancyDraw(self,img,bbox,len=30,t=5):
        x, y, w, h =bbox
        x1, y1 = x+w ,y+h

        cv2.rectangle(img, bbox, (0, 255, 0), 1)
        # for top left
        cv2.line(img, (x,y),(x+len,y),(255,255,0),t)
        cv2.line(img, (x, y), (x, y+len), (255, 255, 0), t)
        # for bottom right
        cv2.line(img, (x1, y1), (x1 - len, y1), (255, 255, 0), t)
        cv2.line(img, (x1, y1), (x1, y1 - len), (255, 255, 0), t)
        # for top right
        cv2.line(img, (x1 ,y), (x1 - len, y), (255, 255, 0), t)
        cv2.line(img, (x1, y), (x1, y + len), (255, 255, 0), t)
        # for bottom left
        cv2.line(img, (x, y1), (x + len, y1), (255, 255, 0), t)
        cv2.line(img, (x, y1), (x, y1 - len), (255, 255, 0), t)
        return img
def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()  # Give success as cap in reading
        img, bboxs = detector.findFaces(img) # add draw=false for not drawing
        # print(bboxs)
        cTime = time.time()  # gives us the current time
        fps = 1 / (cTime - pTime)  # logic for fps
        pTime = cTime
        cv2.putText(img, f'FPS: {str(int(fps))}', (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
        cv2.imshow("Image", img)  # For Image showing
        cv2.waitKey(1)
if __name__=="__main__":
    main()