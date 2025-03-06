import moviepy as mp





def split_video_audio():
    file = mp.VideoFileClip("WIN_20250212_11_50_48_Pro.mp4") # Load the file

    audio = file.audio # Extract only the audio from the file
    audio.write_audiofile("Audio File.mp3") # Create new file with extracted audio only
    audio.write_audiofile("Audio File.wav")

    video = file.without_audio() #Removes audio from file
    video.write_videofile('output_video.mp4') # Create new file with video

