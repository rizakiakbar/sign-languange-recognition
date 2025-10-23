import cv2
import time
import PoseModule as pm

cap = cv2.VideoCapture('PoseVideo/1.mp4')
pTime = 0
detector = pm.poseDetector()
while True:
    succes, img = cap.read()
    img = detector.findPose(img)
    lmlist = detector.findPosition(img)
    if len(lmlist) !=0:
        print(lmlist[14])
        cv2.circle(img, (lmlist[14][1], lmlist[14][2]), 5, (0, 0, 255), cv2.FILLED)
    
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)