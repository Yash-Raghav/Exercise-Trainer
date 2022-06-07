import cv2
import mediapipe as mp
import numpy as np
import time
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles

def calculate_angle(a,b,c):
    '''Calculating angles between joints/pose landmarks'''
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

    
def posture(video):
    cap = cv2.VideoCapture(video) 

    # Set timer to zero
    timer = int(0)
    prev_time = time.time()

    # Dimensions and frames per second for saving output video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
    print(frame_height)
    print(frame_width)

    # For displaying the exercise position and rep count
    start = "START"
    stage = ""
    counter = 0


    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                # Calculate angle
                angle = calculate_angle(hip, knee, ankle)
                
                # Visualize angle
                angle_display = f"Knee bend angle: {angle.round(0)}"
                cv2.putText(image, angle_display, tuple(np.multiply(knee, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Knee bend logic
                if angle < 140:       # Exercise starts 
                    start = ""
                    stage = "HOLD"    
                    
                if timer==9 and stage=="HOLD":
                    stage = "STRAIGHTEN"
                    
                if timer<9 and stage=="HOLD" and angle>140:
                    stage = "PAUSE"
                    
                if stage=="PAUSE":
                    hold_bend_display = "KEEP YOUR KNEE BENT"        
                    cv2.putText(image,hold_bend_display,(240,150), font, 1,(0,0,255),2,cv2.LINE_AA)
                        
                if angle>170 and stage=='STRAIGHTEN':
                    stage = "BEND"
                    timer=0         # Reset the hold counter    
                    counter+=1

            except:
                pass
            
            # Heads up box for display
            cv2.rectangle(image, (900,10), (0,80), (245,0,0), -1)
            
            # Display
            cv2.putText(image, start,(330,70),font, 2, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, stage,(350,60),font, 1, (255,255,255), 2, cv2.LINE_AA)
            counter_text = f"Rep Count: {counter}"
            cv2.putText(image, counter_text, 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Timer function
            if stage=="HOLD":
                period = "{:02d}".format(timer)
                cv2.putText(image,period,(480,60), font, 1,(255,255,255),2,cv2.LINE_AA)
                curr_time = time.time()
                if curr_time-prev_time >= 1:
                    prev_time = curr_time
                    timer = timer+1
                    
            # For showing connections only for lower body
            connection_list = []
            for connection in mp_pose.POSE_CONNECTIONS:
                connection_list.append(connection)
            connection_list.sort(key = lambda t: t[0])
            connection_list
        
            # Display pose landmark lines and connections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, connection_list[23:],
                                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())               
            
            cv2.imshow('MediaPipe Pose', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
            # out.write(image)
            
        cap.release()
        cv2.destroyAllWindows()

posture('KneeBendVideo.mp4')

