import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles

#
key_path = "save_json"
video_path = "keypoints"
target_video = "swf2test.mp4"

cap = cv2.VideoCapture(os.path.join(video_path, target_video))
# cap = cv2.VideoCapture(0)
os.makedirs(os.path.join(key_path, target_video), exist_ok=True)

# Curl counter variables
warning = False
count = 0
pTime = 0
sTime = time.time()
FPS = cap.get(cv2.CAP_PROP_FPS)

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    i=0
    while cap.isOpened():
        ret, frame = cap.read()
        
        # get frame time and FPS
        frame_time=cap.get(cv2.CAP_PROP_POS_MSEC)
        resize_frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
        # Recolor image to RGB
        image = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try :
            landmarks = results.pose_landmarks.landmark
        except : 
            continue
        # Get coordinate
        left_hip = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility,
        ]
        left_knee = [
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility,
        ]
        left_ankle = [
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility,
        ]
        right_hip = [
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility,
        ]
        right_knee = [
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility,
        ]
        right_ankle = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z,
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility,
        ]
        left_shoulder = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility,
        ]
        left_elbow = [
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z,
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility,
        ]
        left_wrist = [
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z,
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility,
        ]
        right_shoulder = [
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility,
        ]
        right_elbow = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z,
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility,
        ]
        right_wrist = [
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility,
        ]
        nose = [
            landmarks[mp_pose.PoseLandmark.NOSE.value].x,
            landmarks[mp_pose.PoseLandmark.NOSE.value].y,
            landmarks[mp_pose.PoseLandmark.NOSE.value].z,
            landmarks[mp_pose.PoseLandmark.NOSE.value].visibility,
        ]
        left_eye = [
            landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y,
            landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].z,
            landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].visibility,
        ]
        right_eye = [
            landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y,
            landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].z,
            landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].visibility,
        ]
        left_ear = [
            landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y,
            landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].z,
            landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].visibility,
        ]
        right_ear = [
            landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y,
            landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].z,
            landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].visibility,
        ]

        time_stamp = str(time.time() - sTime)
        
        # save keypoints
        keypoints = {
            "1. left_hip": left_hip,
            "2. left_knee": left_knee,
            "3. left_ankle": left_ankle,
            "6. right_hip": right_hip,
            "7. right_knee": right_knee,
            "8. right_ankle": right_ankle,
            "14. left_shoulder": left_shoulder,
            "15. left_elbow": left_elbow,
            "16. left_wrist": left_wrist,
            "19. right_shoulder": right_shoulder,
            "20. right_elbow": right_elbow,
            "21. right_wrist": right_wrist,
            "24. nose": nose,
            "25. left_eye": left_eye,
            "26. right_eye": right_eye,
            "27. left_ear": left_ear,
            "28. right_ear": right_ear,
            "time stamp": frame_time,
        }
        
        with open(os.path.join(key_path, target_video, f'{i:0>3}.json'), "w") as f:
            json.dump(keypoints, f, indent=4)
        i=i+1
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime


        cv2.putText(image, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Mediapipe Feed", image)
        #'q'누르면 캠 꺼짐
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()