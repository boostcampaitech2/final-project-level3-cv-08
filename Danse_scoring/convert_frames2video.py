import glob
import cv2
import os
import numpy as np
import subprocess as sp
from moviepy.editor import *

def createVideoClip(clip, folder, name, size=[1920, 1000]):

    vf = clip.shape[0]
    command = ['ffmpeg',
               '-y',  # overwrite output file if it exists
               '-f', 'rawvideo',
               '-s', '%dx%d' % (size[1], size[0]),  # '256x256', # size of one frame
               '-pix_fmt', 'rgb24',
               '-r', '25',  # frames per second
               '-an',  # Tells FFMPEG not to expect any audio
               '-i', '-',  # The input comes from a pipe
               '-vcodec', 'libx264',
               '-b:v', '1500k',
               '-vframes', str(vf),  # 5*25
               '-s', '%dx%d' % (size[1], size[0]),  # '256x256', # size of one frame
               folder + '/' + name]
    # sfolder+'/'+name
    pipe = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
    out, err = pipe.communicate(clip.tostring())
    pipe.wait()
    pipe.terminate()
    print(err)

def createSoundClip(path):
    print(path)
    videoclip = VideoFileClip(path + '.mp4')
    audioclip = AudioFileClip(path + '.mp3').subclip(0, 120)

    videoclip.audio = audioclip
    videoclip.write_videofile(path + '_audio.mp4')

if __name__ == '__main__':
    out_frames = []
    video_name = 'callmebaby'
    for path in sorted(glob.glob(os.path.join('/opt/ml/Pose_project/dataset/training/callmebaby/source/012_B00_001_F', '*.jp*'))):
        print(path)
        out_frame = cv2.imread(path)
        shape = out_frame.shape
        out_frames.append(out_frame[:, :, ::-1])

    final_clip = np.stack(out_frames)
    video_path = '/opt/ml/Pose_project/dataset/training/callmebaby/'
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    createVideoClip(final_clip, video_path, '%s.mp4' % (video_name), [shape[0], shape[1]])
    createSoundClip(video_path + video_name)


















