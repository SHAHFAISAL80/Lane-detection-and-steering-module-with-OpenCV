import cv2
import numpy as np
import math


cap = cv2.VideoCapture(r"C:\Users\Atif Traders\Music\Lane_detection_and_steering_module-main\drive.mp4")

# def frame_detect_line(frame):
    
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     lower = np.uint8([180, 18, 255])
#     upper = np.uint8([0, 0, 231])

#     mask = cv2.inRange(hsv, lower, upper)
#     edges = cv2.Canny(mask, 75, 250)
    
#     cv2.imshow("frame_edges", mask)
    
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=50)

#     if lines is not None:
#         for i in range(0, len(lines)):
#             l = lines[i][0]
#             cv2.line(frame, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
        
while 1:
    
    ret,frame = cap.read()
    if ret:
        
        width, height = frame.shape[1], frame.shape[0]
        
        # ------------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.uint8([180, 18, 255])
        upper = np.uint8([0, 0, 231])

        mask = cv2.inRange(hsv, lower, upper)
        edges = cv2.Canny(mask, 75, 250)
        
        cv2.imshow("mask", edges)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=50)

        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv2.line(frame, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
        # ----------
        
        # change perspective
        # perspective points
        pts1 = np.float32([[560, 440],[710, 440],[200, 640],[1000, 640]])
        pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
        # end perspective points
        
        # perspective points draw circle
        pts1_ = np.array([[560, 440],[710, 440],[200, 640],[1000, 640]])
        pts2_ = np.array([[0,0],[width,0],[0,height],[width,height]])
        
        # for i in range(0,4):
        #     cv2.circle(frame,(pts1_[i][0], pts1_[i][1]),5,(0,0,255),2)
        # end perspective points draw circle
        
        # change perspective 
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
        birds_eye = cv2.warpPerspective(frame, matrix, (width, height))
        
        # end change birds eye perspective
        
        
        grayscale = cv2.cvtColor(birds_eye, cv2.COLOR_BGR2GRAY)
        
        # smooth image
        kernel_size = 5
        blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
        
        # canny edge dedection
        low_t = 50
        high_t = 95
        edges = cv2.Canny(blur, low_t, high_t)
        
        # detect line with houg line transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=50)
        
        # draaw lines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # classification right and left line
            if x1 < 640 or x2 < 640:
                x1_left = x1
                x2_left = x2
                y1_left = y1
                y2_left = y2
            
            elif x1 > 640 or x2 > 640:
                x1_right = x1
                x2_right = x2
                y1_right = y1
                y2_right = y2
            
            
            try:
                # calculate middle points
    
                x1_mid = int((x1_right + x1_left)/2)
                x2_mid = int((x2_right + x2_left)/2)
            
                # y1_mid = int((y1_right + y1_left)/2)
                # y2_mid = int((y2_right + y2_left)/2)
            
                cv2.line(birds_eye, (640, 300), (x2_mid, 420), (0, 255, 0), 2)
                
                
                # create straight pipe line in middle of the frame
                x_1, x_2 = 640, 640
                y_1, y_2 = 300, 420
                cv2.line(birds_eye, (x_1,y_1), (x_2, y_2), (0, 0, 255), 2)
            
                
                # calculate 3 point beetween angle

                point_1 = [x_1, y_1]
                point_2 = [x_2, y_2]
                point_3 = [x2_mid, 420]
                
                radian = np.arctan2(point_2[1] - point_1[1], point_2[0] - point_1[0]) - np.arctan2(point_3[1] - point_1[1], point_3[0] - point_1[0])
                angle = (radian *180 / np.pi)
                
                print("omega : ", angle)
                
                # cv2.putText(frame, str(int(angle)), (15,35),cv2.FONT_HERSHEY_SIMPLEX,1 , (255,0,0), 2, cv2.LINE_AA )
                
                if angle < -30:
                    cv2.putText(frame, "LEFT", (15,35),cv2.FONT_HERSHEY_SIMPLEX,1 , (255,0,0), 2, cv2.LINE_AA )
                elif angle > 25:
                    cv2.putText(frame, "RIGHT", (1150,35),cv2.FONT_HERSHEY_SIMPLEX,1 , (255,0,0), 2, cv2.LINE_AA )
                
                elif angle > -25 and angle < 25:
                    # cv2.putText(frame, "DUZ", (600,35),cv2.FONT_HERSHEY_SIMPLEX,1 , (255,0,0), 2, cv2.LINE_AA )
                    continue
            except NameError:
                continue
            # lane draw line red
            cv2.line(birds_eye, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # print("x1 : {} y1 : {} \nx2 : {} y2 : {}".format(x1,y1, x2 ,y2))
            # print("------------------------")
            
            
        
        
        
        drawing_pts = np.array([[[550, 440],[690, 440],[1000, 640],[200, 640]]], np.int32)
        cv2.polylines(frame, [drawing_pts],True, (0,255,),3)
        # cv2.fillPoly(frame, [drawing_pts], (0,255,0))
            
        birds_eye = cv2.resize(birds_eye, (640,360))
        
        cv2.imshow("birds_eye", birds_eye)
        cv2.imshow("frame", frame)
        
    
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    
    elif ret == False:
        break

cap.release()
cv2.destroyAllWindows()