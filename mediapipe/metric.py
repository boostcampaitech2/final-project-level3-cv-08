import json
import cv2
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


info_dict = {1 : 0, #left_hip
             2 : 1, #left_knee
             3 : 2, #left_ankle
             6 : 3, #right_hip
             7 : 4, #right_knee
             8 : 5, #right_ankle
             14 : 6, #left_shoulder
             15 : 7, #left_elbow
             16 : 8, #left_wrist
             19 : 9, #right_shoulder
             20 : 10, #right_elbow
             21 : 11, #right_wrist
             24 : 12, #nose
             25 : 13, #left_eye
             26 : 14, #right_eye
             27 : 15, #left_ear
             28 : 16} #right_ear

connect_point = [[1,2,3], #왼쪽다리
                 [6,7,8], #오른쪽다리
                 [14,15,16], #왼쪽팔
                 [19,20,21], #오른쪽팔
                 [1,6,19,14,1], #몸통
                 [28,26,24,25,27]] #눈코입

connect_point_parts = [[1,2], #left thigh
                       [2,3], #left calf
                       [6,7], #right thigh
                       [7,8], #right calf
                       [14,15], #left arm
                       [15,16], #left forearm
                       [19,20], #right arm
                       [20,21], #right forearm
                       [1,6,19,14,1], #body
                       [28,26,24,25,27]] #face

vec_point = [[1,2],[2,3], #왼쪽다리
             [6,7],[7,8], #오른쪽다리
             [14,15],[15,16], #왼쪽팔
             [19,20],[20,21], # 오른쪽팔
             [24,25],[24,26],[24,27],[24,28],# 눈코입
             [6,14],[1,19]] #몸통

vec_part = {'left_leg' : [[1,2],[2,3]], #왼쪽다리
            'right_leg': [[6,7],[7,8]], #오른쪽다리
            'left_arm' : [[14,15],[15,16]], #왼쪽팔
            'right_arm': [[19,20],[20,21]], # 오른쪽팔
            'body' : [[6,14],[1,19]]} #몸통

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
            "body2" : [1,19]}

small_name = list(small_parts.keys())

vec_part_key = ['left_leg', 'right_leg', 'left_arm', 'right_arm', 'body', 'total']

sigma = {
"left_leg": [0.087, 0.089], 
"right_leg":[0.087, 0.089],
"left_arm": [0.072, 0.062], 
"right_arm":[0.072, 0.062],
"body":[0.079,0.079] #shoulder
}

color_map = {
    'fast': (255,69,0),
    'slow': (138,43,226),
    'good': (0,255,0),
    'NG' : (255,0,0)
    }

class VideoMetric():
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def extract_vec_norm_by_small_part(self, point_json, info_dict = info_dict, vec_point=small_parts): ## 추가
        point = list(point_json.values())
        left_shoulder = info_dict[14] # left_shoulder
        right_hip = info_dict[6] # right_hip
        normalize_value = (point[left_shoulder][0]*self.width-point[right_hip][0]*self.width, point[left_shoulder][1]*self.height-point[right_hip][1]*self.height)
        normalize_value = np.linalg.norm(normalize_value)
        if normalize_value < 1.0:
            normalize_value = 1
        output = []
        for parts in vec_point.values():
            output_part = []
            p = info_dict[parts[0]]
            q = info_dict[parts[1]]
            x1,y1 = point[p][0]*self.width/normalize_value, point[p][1]*self.height/normalize_value
            x2,y2 = point[q][0]*self.width/normalize_value, point[q][1]*self.height/normalize_value
            output_part.append((x2-x1,y2-y1))
            output.append(output_part)
        return output #left_leg, right_leg, left_arm, right_arm, body
    
    def extract_vec_norm_by_small_part_diff(self,before_point_json, point_json, info_dict = info_dict, vec_point=small_parts):  ## 추가
        before_point = list(before_point_json.values())
        point = list(point_json.values())
        left_shoulder = info_dict[14] # left_shoulder
        right_hip = info_dict[6] # right_hip
        normalize_value = (point[left_shoulder][0]*self.width-point[right_hip][0]*self.width, point[left_shoulder][1]*self.height-point[right_hip][1]*self.height)
        normalize_value = np.linalg.norm(normalize_value)
        if normalize_value < 1.0:
            normalize_value = 1
        output = []
        for parts in vec_point.values():
            output_part = []
            p = info_dict[parts[0]]
            q = info_dict[parts[1]]
            bx1,by1 = before_point[p][0]*self.width/normalize_value, before_point[p][1]*self.height/normalize_value
            bx2,by2 = before_point[q][0]*self.width/normalize_value, before_point[q][1]*self.height/normalize_value                    
            x1,y1 = point[p][0]*self.width/normalize_value, point[p][1]*self.height/normalize_value
            x2,y2 = point[q][0]*self.width/normalize_value, point[q][1]*self.height/normalize_value
            output_part.append((x1-bx1+x2-bx2,y1-by1+y2-by2))
            output.append(output_part)
        return output 
        
    def visual_back_color(self,frame, point_json, speed_metric, connect_point = connect_point_parts, info_dict = info_dict):
        '''
        keypoint -> numpy skeleton image
        '''
        point = list(point_json.values())
        img = frame
        for i, parts in enumerate(connect_point[:-1]):
            c = color_map[speed_metric[i]]
            for i in range(len(parts)-1):
                p = info_dict[parts[i]]
                q = info_dict[parts[i+1]]
                cv2.line(img, (int(point[p][0]*self.width), int(point[p][1]*self.height)), (int(point[q][0]*self.width), int(point[q][1]*self.height)), c, 3)
        return img

def l2_normalize(gt, target):
    output = []
    for i in range(len(gt)):
        x1,y1 = np.abs(gt[i][0] - target[i][0]), np.abs(gt[i][1] - target[i][1])
        output.append(np.linalg.norm((x1,y1)))
    return np.average(output)

def cosine_similar(gt, target):
    output = []
    for i in range(len(gt)):
        if np.linalg.norm(gt[i])!=0 and np.linalg.norm(target[i]) != 0:
            c_s = np.dot(gt[i],target[i])/(np.linalg.norm(gt[i])*np.linalg.norm(target[i]))
            output.append(c_s)
    return np.average(output)

def coco_oks(gt, target, part):
    output = []
    for i in range(len(gt)):
        gx,gy = gt[i][0],gt[i][1]
        tx,ty = target[i][0],target[i][1]
        dx = gx-tx
        dy = gy-ty
        kp_c = sigma[vec_part_key[part]][i]
        oks = np.exp(-(dx**2+dy**2)/(2*(kp_c**2)))
        output.append(oks)
    return np.average(output)