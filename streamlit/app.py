import streamlit as st
import cv2
import io
import os
import yaml
import time
import pandas as pd
import numpy as np
from webcam import webcam_pose
# from config import click_video
from compare import compare_video

UPLOAD = 'upload video'
WEBCAM = 'filmed with webcam'
MUSIC_OPTIONS = ['call me baby', 'next level', 'hope']
SIDEBAR_OPTIONS = [UPLOAD, WEBCAM]

# 비디오 보여주기
def show_video(music_name):
    video_file = open(music_name, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes, format="video/mp4", start_time=0)

#비디오 업로드 후 폴더에 저장
def upload_video(music_name):
    sync_frame = st.sidebar.slider("Sync", -30, 30, 10, 5)
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        g = io.BytesIO(uploaded_file.read())
        temporary_location = "./dataset/target/" + music_name + "_prac.mp4"

        with open(temporary_location, 'wb') as out:
            out.write(g.read())

        out.close()
        # 업로드 완료되면 평가받기
        with st.container():
            with st.spinner('Wait...'):
                compare_video(music_name, sync_frame)
                # time.sleep(30)
                # st.header("당신의 댄스 실력은")
                # video_file = open('./dataset/result/' + music_name + '.mp4', 'rb')
                # video_bytes = video_file.read()
                # st.video(video_bytes, format="video/mp4", start_time=0)

#웹탬으로 촬영 후 폴더 에 저장
def upload_webcam(music_name):
    sync_frame = st.sidebar.slider("Sync", -30, 30, 10, 5)
    # run = st.checkbox('Start/Stop Webcam')
    col1, col2 = st.columns(2)
    start = col1.button("start")
    finish = col2.button("finish")
    # Webcam mediapipe랑 같이 돌리기
    # webcam_pose(run, music_name)
    if start:
        # click_video()
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        fps = camera.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter("./dataset/target/" + music_name + "_prac.mp4", fourcc, fps, (640, 480))

        while (camera.isOpened()):
            flag = 1
            ret, frame = camera.read()
            out.write(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

        camera.release()
        out.release()

    if finish:
        with st.container():
            with st.spinner('Wait...'):
                compare_video(music_name, sync_frame)


def main():
    st.set_page_config(layout="wide")

    #사용 설명
    with st.sidebar.expander("사용설명서"):
        st.write('1. 추고싶은 노래를 선택')
        st.write('2. 평가 방식을 선택 '
                 '\n  - 웹캠 선택 시 실시간으로 웹캠으로 촬영 후 평가 '
                 '\n  - 업로드 선택시 동영상 업로드 후 평가 ')
        # st.write('3. 평가받기 버튼 클릭')

    st.sidebar.title("Dancing Pose")

    #음악 선택 - 음악 추가하기
    music_name = st.sidebar.selectbox('Please select Music', MUSIC_OPTIONS)

    #경로에 동영상 있으면 띄우고 없으면 에러
    data_path = './dataset/video/' + music_name + '.mp4'
    if os.path.exists(data_path) == True:
        show_video(data_path)
    else:
        st.error('no video')

    #업로드 방식 선택
    dance_mode = st.sidebar.selectbox("Please select evaluation method", SIDEBAR_OPTIONS)
    if dance_mode == UPLOAD:
        upload_video(music_name)
    elif dance_mode == WEBCAM:
        upload_webcam(music_name)

    #평가 받기 버튼
    # if st.button('평가받기'):
    #     with st.container():
    #         st.header("당신의 댄스 실력은")
    #         #평가 되는 동안 wait
    #         with st.spinner('Wait...'):
    #             time.sleep(5)
    #             score = 100
    #             st.subheader(str(score) + "점")

if __name__ == "__main__":
    main()
