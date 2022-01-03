# DanceFit : Pose Estimation을 이용한 댄스 자세 교정

## 목차

- [팀소개](#팀소개)
- [프로젝트 소개](#프로젝트-소개)
- [프로젝트 구조](#프로젝트-구조)
- [세부 설명](#세부-설명)
- [시연 결과](#시연-결과)


## 팀소개


<table>
  <tr>
    <td align="center">
      <a href="https://github.com/hanlyang0522">
        <img src="https://avatars.githubusercontent.com/u/67934041?v=4" width="100px;" alt=""/>
        <br />
        <sub>박범수</sub>
      </a>
        <br>
        <sub>PM</sub>
    </td>
    <td align="center">
      <a href="https://github.com/WonsangHwang">
        <img src="https://avatars.githubusercontent.com/u/49892621?v=4" width="100px;" alt=""/>
        <br />
        <sub>황원상</sub>
      </a>
        <br>
        <sub>Modeling</sub>
    </td>
    <td align="center">
      <a href="https://github.com/sala0320">
        <img src="https://avatars.githubusercontent.com/u/49435163?v=4" width="100px;" alt=""/>
        <br />
        <sub>조혜원</sub>
      </a>
        <br>
        <sub>Front-End</sub>
    </td>
    <td align="center">
      <a href="https://github.com/hongsusoo">
        <img src="https://avatars.githubusercontent.com/u/77658029?v=4" width="100px;" alt=""/>
        <br />
        <sub>홍요한</sub>
      </a>
        <br>
        <sub>Metric</sub>
    </td>
    <td align="center">
      <a href="https://github.com/Junhyuk93">
        <img src="https://avatars.githubusercontent.com/u/61610411?v=4" width="100px;" alt=""/>
        <br />
        <sub>박준혁</sub>
        </a>
          <br>
        <sub>Model Research</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/GunwooHan">
        <img src="https://avatars.githubusercontent.com/u/76226252?v=4" width="100px;" alt=""/>
        <br />
        <sub>한건우</sub>
      </a>
        <br>
        <sub>Metric</sub>
    </td>
  </tr>
  <tr>
    </td>
  </tr>
</table>
<br>  


## 프로젝트 소개

### 선정 이유
---
<img src="https://i.imgur.com/eeoWHer.png"  width="70%" height="70%"/>

- 춤에 대한 관심은 커지고 있으나, 시공간의 제약으로 직접 배우러 가기 힘든 상황
- 코로나로 인해 오프라인 활동이 제약됨

### 문제 정의

---
<img src="https://i.imgur.com/jZvI5xc.png"  width="70%" height="70%"/>

- 홈트레이닝의 대부분은 헬스, 요가 등 정적인 운동 

<img src="https://images.velog.io/images/hanlyang0522/post/e944cffd-6f54-44fd-8473-2550679ddbe5/%EA%B7%B8%EB%A6%BC1.gif"  width="70%" height="70%"/>

- 대부분의 metric은 동작의 일부만 비교하기 때문에 keypoint간 비교가 어려움
- 따라서 각 keypoint별로 정확도를 측정하는 것이 중요


### 데이터 소개

---

<img src="https://i.imgur.com/0855U3t.png"  width="70%" height="70%"/>

- [aihub K-pop 안무 영상](https://aihub.or.kr/aidata/34116)
- 기타 수집 데이터
- coco keypoint annotation을 기준으로 17개의 keypoint만 추출해 사용



## 프로젝트 구조


<img src="https://i.imgur.com/NVmVImq.png"  width="70%" height="70%"/>

- 웹서비스를 위해 streamlit 사용
1. 사용자가 추고 싶은 영상을 선택
2. 해당 영상에 맞춰 웹캠으로 실시간으로 촬영해서 업로드 OR 따로 촬영하여 영상 업로드
3. groudtruth 춤 영상과 사용자가 업로드 한 춤 영상 비교 피드백 제공


## 세부 설명

<img src="https://i.imgur.com/Dw2teYK.png"  width="70%" height="70%"/>
<img src="https://user-images.githubusercontent.com/77658029/147928665-0252ca77-28d7-4bd3-917f-3da1835ef64a.gif"  width="70%" height="70%"/>

### Model
---
-  PCK@0.2으로 모델 평가를 하였을 때, 춤과 같은 task에서 **Blazepose**의 성능이 가장 높게 나타남

<img src="https://i.imgur.com/crNklun.png"  width="70%" height="70%"/>


- 동일한 하드웨어 사양에 대해서 Openpose와 BlazePose는 대략 25배의 FPS 차이가 있음
<img src="https://i.imgur.com/CDFKYpL.png" width="500" height="200"/>


### Metric
---
- Model이 예측한 17개의 Keypoint를 이용하여 춤 비교
- **문제 상황** : 신체 사이즈, 촬영 환경에 따라서 좌표값이 달라져 정확한비교가 어려움
🔑 Keypoint 전처리 필요!
<img src="https://user-images.githubusercontent.com/77658029/147926164-6295e111-0fce-41ff-a653-32f6fa3e23c1.png"  width="70%" height="70%"/>

- **데이터 전처리**
    - 위치 벡터 추출 : 신체의 연결부 기준 중심부에서 멀어지는 방향으로 벡터 산출
    
    ![image](https://user-images.githubusercontent.com/77658029/147926597-828fc1da-8eda-44f3-926f-2ca9e5396766.png)
    
    - 방향 벡터 추출 : 신체 부위 keypoints의 변위를 벡터로 변환
    
    ![image](https://user-images.githubusercontent.com/77658029/147926627-2e698fdf-e936-47cd-b9ea-dd5617785a44.png)

- **춤 비교** : 춤을 비교하기 위해서는 동작과 박자가 맞아야함
    - 동작 정확도
        - OKS(Object Keypoint Similarity)
        - Cosine Similiarity
        <img src="https://user-images.githubusercontent.com/77658029/147927125-d5088815-cd25-4ed0-ae35-2e5dc4cdc985.png" width="70%" height="70%"/>
        
    - 박자 정확도
        - 두 시계열의 유사도가 높은 Alignment Point를 찾아 춤의 빠르고 느림을 평가하는데 사용
        <img src="https://user-images.githubusercontent.com/77658029/147927618-918de2f6-8833-404e-90ad-dc929228b4b1.png" width="70%" height="70%"/>
        

## 시연 결과
<img src="https://i.imgur.com/l0fz8MH.gif" width="70%" height="70%"/>

### 후속 개발 및 연구

---
<img src="https://i.imgur.com/tmeq7nc.png"  width="70%" height="70%"/>

- 현재 안무 영상과 User Video에서 사람이 여러 명일 경우 영상에서 가장 왼쪽에 있는 사람을 기준으로 Keypoint를 계산하고 있으나, 이후 Face Detection으로 인원을 미리 계산하고 어떤 사람을 기준으로 keypoint를 매길지 결정한다면 군무 영상에서도 안무 연습이 가능 할 것이라 생각.
-  또한 영상을 보면서 안무를 배울 때 일반적으로 배속을 느리게 하여 연습을 하곤 하는데, 이 배속에 대한 시스템도 개발하게 된다면 활용성이 증대될 수 있을 것이라 생각.

<img src="https://i.imgur.com/FL35Qgx.png"  width="70%" height="70%"/>

- 사용자가 영상을 업로드해서 Keypoint를 뽑아둔 data를 미리 로컬이나 서버에 저장시켜 둔다면 준비된 영상 뿐만 아니라 사용자가 연습하고 싶은 영상을 직접 고를 수 있을 것.
- 미숙한 부분을 중점으로 연습할 수 있는 구간별 연습모드나, 오락성을 위해 여러 사람이 같은 영상에 대해서 평가를 받고 점수로 경쟁을 하거나 스테이지를 깨는 식의 게임모드 등으로 서비스를 확장할 수 있을 것.


## License

Dataset : CC-BY-SA  
Streamlit : Apache license 2.0  
Pytorch : Facebook Copyright   
MediaPipe : Apache License 2.0  
