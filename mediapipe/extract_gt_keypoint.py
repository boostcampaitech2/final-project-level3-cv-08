import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os
import metric
import config
import argparse

x0,y0,x1,y1 = 1e+9,1e+9,-1,-1

# args 정의
# parser = argparse.ArgumentParser()

# parser.add_argument('--key_path', type=int, default="save_json")
# parser.add_argument('--video_path', type=str, default="keypoints")
# parser.add_argument('--target_video', type=str, default="hot_prac.mp4")

# args = parser.parse_args()

pTime = 0
sTime = time.time()
# mediapipe 설정
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles

# 경로 설정
key_path = "save_json"
video_path = "keypoints"
target_video = "hot_gt.mp4"

# video 정보 저장 파일 생성
os.makedirs(os.path.join(key_path, target_video), exist_ok=True)

# video 불러오기 및 video 설정 저장
cap = cv2.VideoCapture(os.path.join(video_path, target_video))
video_inform = {
    'frame_width' : int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    'frame_height' : int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    'video_fps' : int(cap.get(cv2.CAP_PROP_FPS)),
    'total_frame' : int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
}
with open(os.path.join(key_path, target_video, f'_info.json'), "w") as f:
    json.dump(video_inform, f, indent='\t')

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    i=0
    while cap.isOpened():
        ret, frame = cap.read()
        
        # get frame time and FPS
        frame_time=cap.get(cv2.CAP_PROP_POS_MSEC)
        resize_frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        
        # save keypoints
        keypoints = config.make_keypoints(landmarks, mp_pose, video_inform)
        
        with open(os.path.join(key_path, target_video, f'{i:0>4}.json'), "w") as f:
            json.dump(keypoints, f, indent='\t')
            
        for x,y,z,k in list(keypoints.values())[:-1]:
            if x0>x: x0=x
            if y0>y: y0=y
            if x1<x: x1=x
            if y1<y: y1=y
        if x0<0: x0=0
        if y0<0: y0=0
        if x1>1: x1=1
        if y1>1: y1=1
            
        i=i+1
        
        if i%1000==0:
            print(i)
        #'q'누르면 캠 꺼짐
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
        
    video_inform['gt_bbox'] = [x0,y0,x1,y1]
    
    with open(os.path.join(key_path, target_video, f'_info.json'), "w") as f:
        json.dump(video_inform, f, indent='\t')
    cap.release()
    cv2.destroyAllWindows()