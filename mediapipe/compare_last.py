import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os
import metric
import config
import argparse
import matplotlib.pyplot as plt

'''
update
'21.12.23

Position vector(위치벡터) : extract_vec_norm_by_small_part
displacemnet Vector(변위벡터) : extract_vec_norm_by_small_part_diff

position evaluation : (cosine_similarity/2+1) * OKS
displacement evaluation : if L2_norm(extract_vec_norm_by_small_part_diff)>threshold/3: (cosine_similarity/2+1)

1. position evaluation < threshold -> NG
2. displacement evaluation < threshold/3 
    2-1. Fast : GT 진행 후 prac이 먼저 출 경우
    2-2. Slow : GT 진행 후 prac이 따라갈 경우
    2-3. Good
3. (cosine_similarity/2+1) > threshold_cs
    3-1. Fast : GT 진행 후 prac이 먼저 출 경우
    3-2. Slow : GT 진행 후 prac이 따라갈 경우
    3-3. Good
4. NG
'''
def compare_video(music_name):
    pTime = 0

    # mediapipe 설정
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic
    mp_drawing_styles = mp.solutions.drawing_styles

    # 경로 설정
    key_path = "save_json"
    video_path = "keypoints"
    gt_path = f"{music_name}_gt.mp4"
    target_video = f"{music_name}_prac.mp4"

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

    # gt_resize = (int(gt_inform['frame_width']*video_inform['frame_height']/gt_inform['frame_height']), video_inform['frame_height'])
    # gt_video = metric.VideoMetric(gt_resize[0],gt_resize[1])
    # prac_video = metric.VideoMetric(video_inform['frame_width'],video_inform['frame_height'])
    
    prac_resize = (int(video_inform['frame_width']*gt_inform['frame_height']/video_inform['frame_height']), gt_inform['frame_height'])
    gt_video = metric.VideoMetric(gt_inform['frame_width'],gt_inform['frame_height'])
    prac_video = metric.VideoMetric(prac_resize[0],prac_resize[1])   
    

    # gt와 비교할 Frame 수 선정
    compare_frame = 30
    before_frame = 5
    threshold = 0.6 # 위치 vector 평가 기준
    threshold_cs = 0.8 # 변위 vector Cosine Similarity 평가 기준
    accept_frame = 20 # OK 로 평가하는 Frame 수 
    prac_temp = []
    eval_metric = ["normal"]*10  # 시작 후 compare_frame+before_frame 동안 평가 진행 X
    eval_graph = []
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        i=0
        while cap.isOpened():
            ret, frame = cap.read()
            
            if ret is False: break
            
            # get frame time and FPS
            # frame_time=cap.get(cv2.CAP_PROP_POS_MSEC)
            resize_frame = cv2.resize(frame, dsize=prac_resize, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
            
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
            
            # save keypoints
            keypoints = config.make_keypoints(landmarks, mp_pose, video_inform)
            if len(prac_temp)>before_frame:
                prac_temp = prac_temp[1:]
            prac_temp.append(keypoints)
            
            with open(os.path.join(key_path, gt_path,f'{i:0>4}.json')) as json_file:
                gt_json = json.load(json_file)
            
            if i%(compare_frame*2) == compare_frame and i>= compare_frame+before_frame:
                s_p = max(i-compare_frame,0) # start point
                e_p = min(i+compare_frame, gt_inform['total_frame']) # end point

                # body part별로(왼다리, 오른다리, 왼팔, 오른팔, 몸통) normalize된 값 vector 추출
                prac = prac_video.extract_vec_norm_by_small_part(keypoints)
                prac_displace_prac = prac_video.extract_vec_norm_by_small_part_diff(prac_temp[0],keypoints)
                    
                total_eval = [[] for _ in range(len(prac)+1)]
                total_eval_diff = [[] for _ in range(len(prac))]
                for j in range(s_p,e_p,1):
                    with open(os.path.join(key_path, gt_path,f'{j:0>4}.json')) as json_file:
                        gt_temp = json.load(json_file)
                    with open(os.path.join(key_path, gt_path,f'{j-before_frame:0>4}.json')) as json_file:
                        displace_gt_temp = json.load(json_file)
                    gt = gt_video.extract_vec_norm_by_small_part(gt_temp)
                    gt_displace_prac = prac_video.extract_vec_norm_by_small_part_diff(displace_gt_temp,gt_temp)

                    s = 0
                    for part in range(len(prac)):
                        eval = (metric.cosine_similar(gt[part], prac[part])/2+1)*metric.coco_oks(gt[part], prac[part], part)
                        total_eval[part].append(eval)
                        total_eval_diff[part].append((gt_displace_prac[part],prac_displace_prac[part])) #metric.cosine_similar(gt_displace_prac[part], prac_displace_prac[part])/2+1)
                        s+=eval
                    eval_graph.append(s/len(prac)) # 평균 계산!
                    
                eval_metric = []
                for part in range(len(prac)):
                    best_point = np.argmax(total_eval[part])
                    worst_point = np.argmin(total_eval[part])
                    if total_eval[part][best_point] < threshold: # threshold
                        eval_metric.append("NG")
                    else :
                        if np.linalg.norm(total_eval_diff[part][best_point][0])< threshold/5:
                            if best_point-compare_frame-accept_frame>0 : eval_metric.append("fast")
                            elif best_point-compare_frame+accept_frame<0 : eval_metric.append("slow")
                            else : eval_metric.append("good")
                        else : 
                            if metric.cosine_similar(total_eval_diff[part][best_point][0], total_eval_diff[part][best_point][1])/2+1 > threshold_cs:
                                if best_point-compare_frame-accept_frame>0 : eval_metric.append("fast")
                                elif best_point-compare_frame+accept_frame<0 : eval_metric.append("slow")
                                else : eval_metric.append("good")
                            else :
                                eval_metric.append("NG")
                                
            array = (np.zeros((gt_inform['frame_height'],gt_inform['frame_width'],3))+255).astype(np.uint8)
            prac_image = prac_video.visual_back_color(image, keypoints, eval_metric)
            gt_image = gt_video.visual_back_color(array, gt_json, eval_metric)
            
            print(prac_image.shape,gt_image.shape)
            image = cv2.hconcat([gt_image,prac_image])
            
            # l=compare_frame
            # cv2.putText(image, f"speed  : {eval_metric[-1]}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
            # if i>= compare_frame+before_frame:
            #     for txt in range(len(eval_metric)-1):
            #         cv2.putText(image, f"{metric.small_name[txt]} : {total_eval[0][txt]:0.2f}_{eval_metric[txt]}", (10, 60+30*txt), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2) 
            
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(image, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3) # FPS 삽입
            
            i=i+1
            
            cv2.imshow("Mediapipe Feed", image)

            #'q'누르면 캠 꺼짐
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":      
    compare_video('hotbb')