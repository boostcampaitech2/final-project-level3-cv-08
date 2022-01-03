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
import seaborn as sns

small_parts = {
            "left thigh" : [1,2],
            "left calf" : [2,3],
            "right thigh" : [6,7],
            "right calf" : [7,8],
            "left arm" : [14,15],
            "left forearm" : [15,16],
            "right arm" : [19,20],
            "right forearm" : [20,21],
            "body1" : [6,14],
            "body2" : [1,19],
            "total":[1,1]}

small_name = list(small_parts.keys())

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
def compare_video(music_name, sync_frame):
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
        'video_fps' : cap.get(cv2.CAP_PROP_FPS),
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
    
    
    # print( gt_inform['video_fps'],video_inform['video_fps'],gt_inform['video_fps']/video_inform['video_fps'])
    # gt와 비교할 Frame 수 선정
    compare_frame = 15
    before_frame = 5
    match_frame = gt_inform['video_fps']/video_inform['video_fps']# 비디오 두 프레임이 다를 경우에 Sync를 맞춰줌
    threshold = 0.2 # 위치 vector 평가 기준
    threshold_cs = 0.8 # 변위 vector Cosine Similarity 평가 기준
    accept_frame = 5 # OK 로 평가하는 Frame 수
    sync_frame = sync_frame # Sync를 위한 frame
    prac_temp = []
    eval_metric = ["normal"]*10  # 시작 후 compare_frame+before_frame 동안 평가 진행 X
    eval_graph_y = [[] for _ in range(11)]
    eval_graph_x = []
    dtw_temp = []
    dtw_gt = []
    dtw_prac = []
    for j in range(0,gt_inform['video_fps']*3):
        with open(os.path.join(key_path, gt_path,f'{j:0>4}.json')) as json_file:
            dtw_gt_temp = json.load(json_file)
        dtw_gt.append(gt_video.extract_vec_norm_by_small_part(dtw_gt_temp))
        
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        i=0
        while cap.isOpened():
            ret, frame = cap.read()
            # print(i,int(i*match_frame))
            if ret is False: break
            if i>=gt_inform['total_frame']-1: break
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
            
            keypoints = config.make_keypoints(landmarks, mp_pose, video_inform)
            if len(prac_temp)>compare_frame:
                prac_temp = prac_temp[1:]
            prac_temp.append(keypoints)
            
            with open(os.path.join(key_path, gt_path,f'{int(i*match_frame+sync_frame):0>4}.json')) as json_file:
                gt_json = json.load(json_file)
            
            if i%(compare_frame) == 0 and i>=gt_inform['video_fps']*3:
                s_p = max(int(i*match_frame+sync_frame)-compare_frame,before_frame) # start point
                e_p = min(int(i*match_frame+sync_frame)+compare_frame, gt_inform['total_frame']-1) # end point

                # body part별로(왼다리, 오른다리, 왼팔, 오른팔, 몸통) normalize된 값 vector 추출
                prac = prac_video.extract_vec_norm_by_small_part(keypoints)
                prac_displace_prac = prac_video.extract_vec_norm_by_small_part_diff(prac_temp[i-before_frame],keypoints)
                    
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
                        eval = metric.coco_oks(gt[part], prac[part], part) * (metric.cosine_similar(gt[part], prac[part])/2+0.5)
                        total_eval[part].append(eval)
                        total_eval_diff[part].append((gt_displace_prac[part],prac_displace_prac[part])) #metric.cosine_similar(gt_displace_prac[part], prac_displace_prac[part])/2+1)
                        s+=eval
                    total_eval[-1].append(s/len(prac))
                    
                eval_graph_y[-1].append(total_eval[-1][np.argmax(total_eval[-1])]) # 평균 계산
                eval_graph_x.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
                eval_metric = []
                for part in range(len(prac)):
                    eval_graph_y[part].append(total_eval[part][np.argmax(total_eval[part])])
                    best_point = np.argmax(total_eval[part])
                    if total_eval[part][best_point] < threshold: # threshold
                        eval_metric.append("NG")
                    else :
                        if np.linalg.norm(total_eval_diff[part][best_point][0])< threshold or np.linalg.norm(total_eval_diff[part][best_point][1])<threshold :
                            if best_point-compare_frame-accept_frame>0 : eval_metric.append(best_point)#.append("fast")
                            elif best_point-compare_frame+accept_frame<0 : eval_metric.append(best_point)#.append("slow")
                            else : eval_metric.append(best_point)#.append("good")
                        else : 
                            if metric.cosine_similar(total_eval_diff[part][best_point][0], total_eval_diff[part][best_point][1]) > 0:
                                if best_point-compare_frame-accept_frame>0 : eval_metric.append(best_point)#.append("fast")
                                elif best_point-compare_frame+accept_frame<0 : eval_metric.append(best_point)#.append("slow")
                                else : eval_metric.append(best_point)#.append("good")
                            else :
                                eval_metric.append("NG")
                         
            elif i < gt_inform['video_fps']*3 and i%(compare_frame) == 0:
                if i >=compare_frame-1:
                    dtw_prac=dtw_prac[1:]
                    dtw_prac.append(gt_video.extract_vec_norm_by_small_part(dtw_gt_temp))
                    
                    for s_f in range(0,gt_inform['video_fps']*3-compare_frame):
                        
                        for g,p in zip(dtw_gt[s_f:s_f+compare_frame], dtw_prac):
                            for part, gg, pp in enumerate(zip(g,p)):
                                dtw_temp.append(metric.coco_oks(gg, pp, part))
                    
            array = (np.zeros((gt_inform['frame_height'],gt_inform['frame_width'],3))+255).astype(np.uint8)
            prac_image = prac_video.visual_back_color(image, keypoints, eval_metric)
            gt_image = gt_video.visual_back_color(array, gt_json, eval_metric)
            
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
            cv2.putText(image, 'GT', (gt_inform['frame_width']//2-25, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
            i=i+1
            
            cv2.imshow("Mediapipe Feed", image)
            
            #'q'누르면 캠 꺼짐
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        return eval_graph_x,eval_graph_y
    
if __name__ == "__main__":      
    eval_graph_x,eval_graph_y=compare_video('hotbb')
    # print(len(eval_graph_x))
    # print(len(eval_graph_y[-1]))
    # print(len(eval_graph_y))
    plt.figure(figsize = (10,6), constrained_layout=True)
    for i in range(11):
        ax = plt.subplot(6,2,i+1) #)
        plt.ylim(0,1)
        sns.lineplot(x=eval_graph_x,y=eval_graph_y[i])
        # print(len(eval_graph_y[i]))
        ax.set_title(f"{small_name[i]}")
    plt.savefig("result.png")