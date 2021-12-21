import cv2
import mediapipe as mp
import time
import streamlit as st
import config

def webcam_pose(run, music_name):

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    pTime = 0
    sTime = time.time()

    if run:
        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter("./dataset/target/" + music_name + ".mp4", fourcc, fps, (w, h))
        video_inform = {
            'frame_width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'frame_height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'video_fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'total_frame': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }


        # Setup mediapipe instance

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()

                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = True

                # Make detection
                results = pose.process(image)

                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                except:
                    continue

                keypoints = config.make_keypoints(landmarks, mp_pose, video_inform)
                # with open(os.path.join('dataset/target/keypoint/', time_stamp), "w") as f:
                #     json.dump(keypoints, f, indent=4)

                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
                )
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime

                FRAME_WINDOW.image(image)
                out.write(frame)

            cap.release()