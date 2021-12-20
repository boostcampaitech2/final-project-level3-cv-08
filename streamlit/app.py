import streamlit as st
import cv2
import io
import os
import yaml
import time
import pandas as pd
import numpy as np
from datetime import datetime
import pyautogui

UPLOAD = 'upload video'
WEBCAM = 'filmed with webcam'
MUSIC_OPTIONS = ['call me baby', 'next level','Fire']
SIDEBAR_OPTIONS = [UPLOAD, WEBCAM]
now = datetime.now()

def show_video(music_name):
    video_file = open(music_name, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes, format="video/mp4", start_time=0)



#TODO 비디오 mediapipe
def upload_video():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        st.write(uploaded_file)
        # bytes_data = uploaded_file.getvalue()
        # st.write(bytes_data)



# TODO 웹캠 mediaipe
def upload_webcam():
    now = datetime.now()
    run = st.checkbox('Start/Stop Webcam')
    if run:
        pyautogui.keyDown('shift')
        pyautogui.keyDown('tab')
        pyautogui.keyDown('tab')
        pyautogui.keyDown('tab')
        pyautogui.keyDown('tab')
        pyautogui.keyDown('tab')
        pyautogui.keyDown('tab')
        pyautogui.keyUp('shift')
        pyautogui.keyUp('tab')
        pyautogui.keyDown('space')
        pyautogui.keyUp('space')

        st.text(f"Start time is {now.hour}:{now.minute}:{now.second}.")
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('./dataset/save.mp4', fourcc, 25.0, (640, 480))

        while(camera.isOpened()):
            ret, frame = camera.read()
            out.write(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

        camera.release()
        out.release()



def main():
    #사용 설명
    with st.expander("사용설명서"):
        st.write('1. 추고싶은 노래를 선택')
        st.write('2. 평가 방식을 선택 '
                '\n  - 웹캠 선택 시 실시간으로 웹캠으로 촬영 후 평가 '
                '\n  - 업로드 선택시 동영상 업로드 후 평가 ')
        st.write('3. 평가받기 버튼 클릭')

    st.sidebar.title("Dancing Pose")

    #음악 선택 - 음악 추가하기
    music_name = st.sidebar.selectbox('Please select Music', MUSIC_OPTIONS)

    #경로에 동영상 있으면 띄우고 없으면 에러
    data_path = './dataset/training/' + music_name + '.mp4'
    if os.path.exists(data_path) == True:
        show_video(data_path)
    else:
        st.error('no video')

    #업로드 방식 선택
    dance_mode = st.sidebar.selectbox("Please select evaluation method", SIDEBAR_OPTIONS)
    if dance_mode == UPLOAD:
        upload_video()
    elif dance_mode == WEBCAM:
        upload_webcam()

    #평가 받기 버튼
    if st.button('평가받기'):
        with st.container():
            st.header("당신의 댄스 실력은")
            #평가 되는 동안 wait
            with st.spinner('Wait...'):
                time.sleep(5)
                # TODO score return 받기
                score = 100
                st.subheader(str(score) + "점")
                # TODO score 그래프
                chart_data = pd.DataFrame(
                    np.random.randn(20, 3),
                    columns = ['a', 'b', 'c'])
                st.line_chart(chart_data)

if __name__ == "__main__":
    main()