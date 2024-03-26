import cv2
import numpy as np
import time

def read_vid():
    
    img = cv2.VideoCapture(r"C:\Users\Atif Traders\Music\Lane_detection_and_steering_module-main\drive.mp4")
    return img


img = read_vid()

while True:
    ret, frame = img.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    canny = cv2.Canny(gray, threshold1 = 170,  threshold2 = 250)
    
    width, height = frame.shape[1], frame.shape[0]
   
    pts1 = np.float32([[590, 440],
                      [690, 440],
                      [200, 640],
                      [1000, 640]])
    
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    
    pts1_ = np.array([[590, 440],
                      [690, 440],
                      [200, 640],
                      [1000, 640]])
    
    pts2_ = np.array([[0,0],[width,0],[0,height],[width,height]])
    
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    birds_eye = cv2.warpPerspective(frame, matrix, (width, height))
    
    birds_eye_r =  birds_eye[:, int(width/2):]
    birds_eye_l = birds_eye[:,:int(width/2)]
    
    for i in range(0,4):
        cv2.circle(frame,(pts1_[i][0], pts1_[i][1]),5,(0,0,255),2)
    
    roi = frame[400:600,350:900]
    #           y      ,   x
    
    gray = cv2.cvtColor(birds_eye, cv2.COLOR_BGR2GRAY)
    
    gray_r = cv2.cvtColor(birds_eye_r, cv2.COLOR_BGR2GRAY)
    gray_l = cv2.cvtColor(birds_eye_l, cv2.COLOR_BGR2GRAY)
    
    ret, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY) 
    
    ret, thresh_r = cv2.threshold(gray_r, 170, 255, cv2.THRESH_BINARY) 
    ret, thresh_l = cv2.threshold(gray_l, 170, 255, cv2.THRESH_BINARY) 
    
    
    linesP = cv2.HoughLinesP(thresh, 1, np.pi / 180, 50, None, 50, 10)
    
    linesP_r = cv2.HoughLinesP(thresh_r, 1, np.pi / 180, 50, None, 50, 10)
    linesP_l = cv2.HoughLinesP(thresh_l, 1, np.pi / 180, 50, None, 50, 10)
    
    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         cv2.line(birds_eye, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    #         # cv2.line(birds_eye, (l[0], l[1]), (l[2], l[3]), (0,255,0), 3, cv2.LINE_AA)
            
    for i in range(0,len(linesP_r)):
        l = linesP_r[i][0]
        
        cv2.line(birds_eye, (l[0]+int(width/2), l[1]), (l[2]+int(width/2), l[3]), (0,0,255), 3, cv2.LINE_AA)
     
    for i in range(0,len(linesP_l)):
        l = linesP_l[i][0]
        cv2.line(birds_eye, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    
    for i in range(0,len(linesP_r)): 
        try:
            ll = linesP_l[i][0]
            x1 = int((l[0]+width/2+ll[0])/2)
            x2 = int((l[2]+width/2+ll[2])/2)
            
            cv2.line(birds_eye,  (640,50),(x1, ll[1]), (0,255,0), 3, cv2.LINE_AA)
            print("x1 : {}  ll : {}".format(x1,ll[1]))
        except IndexError:
            continue
    
    drawing_pts = np.array([[[590, 440],[690, 440],[1000, 640],[200, 640]]], np.int32)

    cv2.polylines(frame, [drawing_pts],True, (0,255,),1)
    
    
    cv2.imshow("frame", frame)
    cv2.imshow("birdsEye", birds_eye)
    
    
    if ret == False or cv2.waitKey(60) == ord("q"):
        break
    
img.release()
cv2.destroyAllWindows()


