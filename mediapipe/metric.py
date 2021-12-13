import json
import cv2
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



w,h= 640, 480
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




def drawimg(point_json, connect_point = connect_point, save = None): #Json 파일 point 연결
    os.makedirs(os.path.join(os.getcwd(),save), exist_ok=True)
    point = list(point_json.values())
    img = np.zeros([h,w])+255
    for parts in connect_point:
        for i in range(len(parts)-1):
            p = info_dict[parts[i]]
            q = info_dict[parts[i+1]]
            cv2.line(img, (int(point[p][0]*w), int(point[p][1]*h)), (int(point[q][0]*w), int(point[q][1]*h)), (0,0,0), 3)
    plt.imshow(img, cmap='gray')
    if save != None :
        plt.imsave(save+f'/{point[-1]}.png',img, cmap='gray')
    else : plt.imshow(img, cmap='gray')

def extract_vec(point_json, vec_point = vec_point): # 
    point = list(point_json.values())
    output_vecs = []
    for parts in vec_point:
        for i in range(len(parts)-1):
            p = info_dict[parts[i]]
            q = info_dict[parts[i+1]]
            x1,y1 = int(point[p][0]*w), int(point[p][1]*h)
            x2,y2 = int(point[q][0]*w), int(point[q][1]*h)
            output_vecs.append((x2-x1,y2-y1))
    return output_vecs

def flow_vec(before_point_json,point_json, vec_point):
    befo_point = extract_vec(before_point_json, vec_point)
    now_point = extract_vec(point_json, vec_point)
    output_flow = []
    for i in range(len(befo_point)):
        output_flow.append((now_point[i][0]-befo_point[i][0],now_point[i][1]-befo_point[i][1]))
    return output_flow

def l2_normalize(gt, target):
    for i in range(len(gt)):
        output = []
        x1,y1 = np.abs(gt[i][0] - target[i][0]), np.abs(gt[i][1] - target[i][1])
        output.append(np.linalg.norm((x1,y1)))
    return np.average(output)

def cosine_similar(gt, target):
    for i in range(len(gt)):
        output = []
        c_s = np.dot(gt[i],target[i])/(np.linalg.norm(gt[i])*np.linalg.norm(target[i]))
        output.append(c_s)
    return np.average(output)