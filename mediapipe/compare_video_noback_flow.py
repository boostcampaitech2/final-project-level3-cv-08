import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os
import metric
import config
import argparse

# gt_bbox 추가 필요


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
gt_path = "hot_gt.mp4"
target_video = "hot_prac.mp4"

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

# gt information 가져오기
with open(os.path.join(key_path, gt_path, f'_info.json')) as json_file:
    gt_inform = json.load(json_file) # frame_width, frame_height, video_fps, total_frame, #gt_bbox

gt_resize = (int(gt_inform['frame_width']*video_inform['frame_height']/gt_inform['frame_height']), video_inform['frame_height'])

hot_gt = metric.VideoMetric(gt_resize[0],gt_resize[1])
hot_prac = metric.VideoMetric(video_inform['frame_width'],video_inform['frame_height'])

# gt와 비교할 Frame 수 선정
compare_frame = 10
# x0,y0,x1,y1 = gt_inform['gt_bbox']
# gt_bbox = [x0*gt_resize[0],y0*gt_resize[1],x1*gt_resize[0],y1*gt_resize[1]]

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
            
        # gt와 비교 구간, target 1개 frame 시점에 대해서 앞뒤로 'compare_frame' 수 만큼의 gt frame과 비교하여 최고점 추출 및 박자 확인
        s_p = max(i-compare_frame,0) # start point
        e = min(i+compare_frame, gt_inform['total_frame']) # end point
        
        # body part별로(왼다리, 오른다리, 왼팔, 오른팔, 몸통) normalize된 값 vector 추출
        prac = hot_prac.extract_vec_norm_by_part(keypoints)
        total = [[],[],[],[],[],[]]
        for j in range(s_p,e,1):
            with open(f'save_json/hot_gt.mp4/{j:0>4}.json') as json_file:
                hot_gt_json = json.load(json_file)
                if i==j: gt_mid_point = hot_gt_json
            gt = hot_gt.extract_vec_norm_by_part(hot_gt_json)
            s = 0
            for part in range(5):
                temp = metric.l2_normalize(gt[part], prac[part])#*metric.cosine_similar(gt[part], prac[part])
                total[part].append(temp) # l2_normalize로 비교
                s+=temp
            total[5].append(s)
        speed_metric = []
        for part in range(6):
            good_point = np.argmin(total[part])
            if good_point-compare_frame+5>0: speed_metric.append("fast")
            elif good_point-compare_frame+5<0: speed_metric.append("slow")
            else : speed_metric.append("good")
        
        prac_image = hot_prac.visual(keypoints)
        gt_image = hot_gt.visual(gt_mid_point)
        image = cv2.hconcat([gt_image,prac_image])
        l=len(total[0])//2
        cv2.putText(image, f"speed  : {speed_metric[5]}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, f"left_leg  : {total[0][l]:0.2f}_{speed_metric[0]}", (10, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, f"right_leg : {total[1][l]:0.2f}_{speed_metric[1]}", (10, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, f"left_arm  : {total[2][l]:0.2f}_{speed_metric[2]}", (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, f"right_arm : {total[3][l]:0.2f}_{speed_metric[3]}", (10, 150), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, f"body      : {total[4][l]:0.2f}_{speed_metric[4]}", (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)        
        
        i=i+1
        
        cv2.imshow("Mediapipe Feed", image)
        #'q'누르면 캠 꺼짐
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()