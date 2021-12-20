
metric 사용법

1. keypoints에 gt, prac 영상 넣기 
2. extract_gt_keypoints.py내에 target_video에 gt 영상 이름 변경
3. python extract_gt_keypoints.py 실행 -> save_json/{영상이름}/.json keypoints 파일 생성
4. compare_norm_veccos_10frame.py내에 gt_path, target_video 영상 이름 변경
5. python compare_norm_veccos_10frame.py

영상 list 로 제공
