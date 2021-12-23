# import pyautogui

def make_keypoints(landmarks, mp_pose, video_inform):
    left_hip = [
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z,
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility,
    ]
    left_knee = [
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z,
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility,
    ]
    left_ankle = [
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z,
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility,
    ]
    right_hip = [
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z,
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility,
    ]
    right_knee = [
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z,
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility,
    ]
    right_ankle = [
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z,
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility,
    ]
    left_shoulder = [
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z,
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility,
    ]
    left_elbow = [
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z,
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility,
    ]
    left_wrist = [
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z,
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility,
    ]
    right_shoulder = [
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z,
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility,
    ]
    right_elbow = [
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z,
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility,
    ]
    right_wrist = [
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z,
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility,
    ]
    nose = [
        landmarks[mp_pose.PoseLandmark.NOSE.value].x,
        landmarks[mp_pose.PoseLandmark.NOSE.value].y,
        landmarks[mp_pose.PoseLandmark.NOSE.value].z,
        landmarks[mp_pose.PoseLandmark.NOSE.value].visibility,
    ]
    left_eye = [
        landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y,
        landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].z,
        landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].visibility,
    ]
    right_eye = [
        landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y,
        landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].z,
        landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].visibility,
    ]
    left_ear = [
        landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y,
        landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].z,
        landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].visibility,
    ]
    right_ear = [
        landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y,
        landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].z,
        landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].visibility,
    ]

    return {
        "1. left_hip": left_hip,
        "2. left_knee": left_knee,
        "3. left_ankle": left_ankle,
        "6. right_hip": right_hip,
        "7. right_knee": right_knee,
        "8. right_ankle": right_ankle,
        "14. left_shoulder": left_shoulder,
        "15. left_elbow": left_elbow,
        "16. left_wrist": left_wrist,
        "19. right_shoulder": right_shoulder,
        "20. right_elbow": right_elbow,
        "21. right_wrist": right_wrist,
        "24. nose": nose,
        "25. left_eye": left_eye,
        "26. right_eye": right_eye,
        "27. left_ear": left_ear,
        "28. right_ear": right_ear,
        "time stamp": video_inform['video_fps'],
    }
#
# def click_video():
#     pyautogui.keyDown('shift')
#     pyautogui.keyDown('tab')
#     pyautogui.keyDown('tab')
#     pyautogui.keyDown('tab')
#     pyautogui.keyDown('tab')
#     pyautogui.keyDown('tab')
#     pyautogui.keyDown('tab')
#     pyautogui.keyUp('shift')
#     pyautogui.keyUp('tab')
#     pyautogui.keyDown('space')
#     pyautogui.keyUp('space')
