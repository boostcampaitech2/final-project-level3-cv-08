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
                 [19,20,21], # 오른쪽팔
                 [28,26,24,25,27],# 눈코입
                 [1,6,19,14,1]] #몸통

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

class VideoMetric():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
    def drawimg(self,point_json, connect_point=connect_point, info_dict = info_dict, save = False):
        point = list(point_json.values())
        img = np.zeros([self.height,self.width])+255
        for parts in connect_point:
            for i in range(len(parts)-1):
                p = info_dict[parts[i]]
                q = info_dict[parts[i+1]]
                cv2.line(img, (int(point[p][0]*self.width), int(point[p][1]*self.height)), (int(point[q][0]*self.width), int(point[q][1]*self.height)), (0,0,0), 3)
        if save:
            os.makedirs(save, exist_ok=True)
            plt.imsave(save+f'/{point[-1]}.png', img, cmap='gray')
        else : plt.imshow(img, cmap='gray')

    def extract_vec(self, point_json, info_dict = info_dict, vec_point=vec_point):
        point = list(point_json.values())
        output_vecs = []
        for parts in vec_point:
            for i in range(len(parts)-1):
                p = info_dict[parts[i]]
                q = info_dict[parts[i+1]]
                x1,y1 = int(point[p][0]*self.width), int(point[p][1]*self.height)
                x2,y2 = int(point[q][0]*self.width), int(point[q][1]*self.height)
                output_vecs.append((x2-x1,y2-y1))
        return output_vecs

    def extract_vec_norm_pck(self, point_json, info_dict = info_dict, vec_point=vec_point):
        point = list(point_json.values())
        left_eye = info_dict[25] # left_eye
        right_eye = info_dict[26] # right_eye
        normalize_value = (point[left_eye][0]*self.width-point[right_eye][0]*self.width, point[left_eye][1]*self.height-point[right_eye][1]*self.height)# 왼쪽 어깨~ 오른쪽 골반 
        normalize_value = np.linalg.norm(normalize_value)
        if normalize_value < 1.0:
            normalize_value = 1
        output_vecs = []
        for parts in vec_point:
            for i in range(len(parts)-1):
                p = info_dict[parts[i]]
                q = info_dict[parts[i+1]]
                x1,y1 = int(point[p][0]*self.width/normalize_value), int(point[p][1]*self.height/normalize_value)
                x2,y2 = int(point[q][0]*self.width/normalize_value), int(point[q][1]*self.height/normalize_value)
                output_vecs.append((x2-x1,y2-y1))
        return output_vecs


    def extract_vec_norm(self, point_json, info_dict = info_dict, vec_point=vec_point):
        point = list(point_json.values())
        left_shoulder = info_dict[14] # left_shoulder
        right_hip = info_dict[6] # right_hip
        normalize_value = (point[left_shoulder][0]*self.width-point[right_hip][0]*self.width, point[left_shoulder][1]*self.height-point[right_hip][1]*self.height)
        normalize_value = np.linalg.norm(normalize_value)
        if normalize_value < 1.0:
            normalize_value = 1
        output_vecs = []
        for parts in vec_point:
            for i in range(len(parts)-1):
                p = info_dict[parts[i]]
                q = info_dict[parts[i+1]]
                x1,y1 = int(point[p][0]*self.width/normalize_value), int(point[p][1]*self.height/normalize_value)
                x2,y2 = int(point[q][0]*self.width/normalize_value), int(point[q][1]*self.height/normalize_value)
                output_vecs.append((x2-x1,y2-y1))
        return output_vecs    

    def extract_vec_norm_by_part(self, point_json, info_dict = info_dict, vec_point=vec_part):
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
            for part in parts:
                for i in range(len(part)-1):
                    p = info_dict[part[i]]
                    q = info_dict[part[i+1]]
                    x1,y1 = point[p][0]*self.width/normalize_value, point[p][1]*self.height/normalize_value
                    x2,y2 = point[q][0]*self.width/normalize_value, point[q][1]*self.height/normalize_value
                    output_part.append((x2-x1,y2-y1))
            output.append(output_part)
        return output #left_leg, right_leg, left_arm, right_arm, body
    
    def extract_vec_norm_by_part_pck(self, point_json, info_dict = info_dict, vec_point=vec_part):
        point = list(point_json.values())
        left_eye = info_dict[25] # left_eye
        right_eye = info_dict[26] # right_eye
        normalize_value = (point[left_eye][0]*self.width-point[right_eye][0]*self.width, point[left_eye][1]*self.height-point[right_eye][1]*self.height)# 왼쪽 어깨~ 오른쪽 골반 
        normalize_value = np.linalg.norm(normalize_value)
        if normalize_value < 1.0:
            normalize_value = 1
        output = []
        for parts in vec_point.values():
            output_part = []
            for part in parts:
                for i in range(len(part)-1):
                    p = info_dict[part[i]]
                    q = info_dict[part[i+1]]
                    x1,y1 = point[p][0]*self.width/normalize_value, point[p][1]*self.height/normalize_value
                    x2,y2 = point[q][0]*self.width/normalize_value, point[q][1]*self.height/normalize_value
                    output_part.append((x2-x1,y2-y1))
            output.append(output_part)
        return output #left_leg, right_leg, left_arm, right_arm, body    
    

    def visual(self, point_json, connect_point = connect_point, info_dict = info_dict, save = False):
        '''
        keypoint -> numpy skeleton image
        
        '''
        point = list(point_json.values())
        img = img = np.zeros([self.height,self.width])+255
        for parts in connect_point:
            for i in range(len(parts)-1):
                p = info_dict[parts[i]]
                q = info_dict[parts[i+1]]
                cv2.line(img, (int(point[p][0]*self.width), int(point[p][1]*self.height)), (int(point[q][0]*self.width), int(point[q][1]*self.height)), (0, 0, 0), 3)
        return img

    def visual_back(self,frame, point_json, connect_point = connect_point, info_dict = info_dict, save = False):
        '''
        keypoint -> numpy skeleton image
        
        '''
        point = list(point_json.values())
        img = frame
        for parts in connect_point:
            for i in range(len(parts)-1):
                p = info_dict[parts[i]]
                q = info_dict[parts[i+1]]
                cv2.line(img, (int(point[p][0]*self.width), int(point[p][1]*self.height)), (int(point[q][0]*self.width), int(point[q][1]*self.height)), (0, 0, 0), 3)
        return img

def compare_visual(background, gt_size, prac_size, gt_keypoints, prac_keypoints, connect_point = connect_point, info_dict = info_dict):
    '''
    keypoint -> numpy skeleton image
    '''
    # [6,14],[1,19]
    points = [list(gt_keypoints.values()),list(prac_keypoints.values())]
    img = np.zeros([prac_size[1],prac_size[0]])+255
    for t in range(2):
        for parts in connect_point:
            for i in range(len(parts)-1):
                p = info_dict[parts[i]]
                q = info_dict[parts[i+1]]
                if t%2:
                    cv2.line(img, (int(points[t][p][0]*prac_size[0]), int(points[t][p][1]*prac_size[1])), (int(points[t][q][0]*prac_size[0]), int(points[t][q][1]*prac_size[1])), (245, 66, 230), 3)
                else :    
                    cv2.line(img, (int(points[t][p][0]*prac_size[0]), int(points[t][p][1]*prac_size[1])), (int(points[t][q][0]*prac_size[0]), int(points[t][q][1]*prac_size[1])), (245, 66, 230), 3)
    return img

def flow_vec(before_point_json, point_json, vec_point = vec_point):
    befo_point = before_point_json
    now_point = point_json
#     befo_point = extract_vec(before_point_json, vec_point)
#     now_point = extract_vec(point_json, vec_point)
    output_flow = []
    for i in range(len(before_point_json)):
        output_flow.append((now_point[i][0]-befo_point[i][0],now_point[i][1]-befo_point[i][1]))
    return output_flow

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

    
# def flow_part_vec(before_point_json, point_json, vec_point = vec_point):
#     point_before = list(before_point_json.values())
#     point = list(point_json.values())
#     output_vecs = []
#     for parts in vec_point:
#         for i in range(len(parts)-1):
#             p = info_dict[parts[i]]
#             q = info_dict[parts[i+1]]
#             x1,y1 = int(point[p][0]*self.width), int(point[p][1]*self.height)
#             x2,y2 = int(point[q][0]*self.width), int(point[q][1]*self.height)
#             output_vecs.append((x2-x1,y2-y1))
            
#     output_flow = []
#     for i in range(len(before_point_json)):
#         output_flow.append((now_point[i][0]-befo_point[i][0],now_point[i][1]-befo_point[i][1]))
#     return output_flow