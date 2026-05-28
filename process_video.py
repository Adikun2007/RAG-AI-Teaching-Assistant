# Here i will convert the videos into mp3 using ffmpeg

import os 
import subprocess

os.makedirs('audios', exist_ok=True)

files = os.listdir('videos')
for file in files:
    video_number = file.split('_')[0]
    video_name = '_'.join(file.split('_')[1:]).replace('.mp4', '')
    print(video_name)
    subprocess.run(['ffmpeg', '-i', f"videos/{file}", f"audios/{video_number}_{video_name}.mp3"])