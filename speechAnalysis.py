import librosa.effects
import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
import pandas as pd


##Speech2Text#############
def speech_text(AudioFile):
    audio_file = "../../Downloads/Audio File.wav"  # load the audio file

    recognizer = sr.Recognizer()  # iniailize the recognizer

    with sr.AudioFile(audio_file) as source:
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.record(source)

    try:
        print("Converting Speech2TEXT")
        text = recognizer.recognize_google(audio_data)  # use Google Api to convert to text
        with open("../../Downloads/Speech_text.txt", "w") as f:  # save text in text file
            f.write(text)

    except sr.UnknownValueError:
        return "Google Speech Recognition could not understand the audio."
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"




##SPEECH RATE ###############################
def speech_rate_analysis(AudioFile):
    audio_file = AudioSegment.from_file("Audio File.wav")
    duration = len(audio_file)/1000

    with open("../../Downloads/Speech_text.txt", "r") as f:
        text = f.read()

    words = text.split()

    speech_Rate = (len(words)/duration)*60

    return speech_Rate,words



####VOCAL CLARITY#########################
def vocal_clarity(AudioFile):
    audio_file = "Audio File.wav"
    y,sr = librosa.load(audio_file,sr=None)

    #Estimate signal energy
    signal_energy = np.sum(np.square(y))

    #Estimate noise energy
    noise_y = y- librosa.effects.preemphasis((y))
    noise_energy = np.sum(np.square(noise_y))

    snr = 10*np.log10(signal_energy/noise_energy)

    if snr > 30:
        return "EXCELLENT CLARITY"
    elif 20<snr<30:
        return "GOOD CLARITY"
    elif snr <10:
        return  "POOR CLARITY"


###Pitch Analysis#######
import librosa
import matplotlib.pyplot as plt

# Load the audio file
audio_file = "interview_audio.wav"
y, sr = librosa.load(audio_file)

# Extract the pitch (fundamental frequency)
y_harmonic, y_percussive = librosa.effects.hpss(y)
pitch, magnitude = librosa.core.piptrack(y=y_harmonic, sr=sr)

# Extract the pitch at a specific frame (time)
pitch_values = [pitch[f].mean() for f in range(pitch.shape[1])]
plt.plot(pitch_values)
plt.title("Pitch over Time")
plt.xlabel("Time (Frames)")
plt.ylabel("Pitch (Hz)")
plt.show()
