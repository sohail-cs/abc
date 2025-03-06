import spacy
import speech_recognition as sr
import pandas as pd
import re
import textstat

nlp = spacy.load('en_core_web_sm')


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


###Vocabulary Usage#######
def vocabulary_usage(word_file):
    with open(word_file, 'r') as f:
        line = f.read().lower()
    words = re.findall(r'\b\w+\b', line)
    doc = nlp("".join(words))
    words = [token.text for token in doc]
    total_words = len(words)
    unique_words = len(set(words))
    ttr =  (unique_words/total_words)*100 if total_words >0 else 0
    readability = textstat.flesch_kincaid_grade(line)
    if readability > 13.0:
        return "readability level Advanced"

##Filler Classification########
def filler_words_classification(word_file):
    words = []
    with open(word_file,'r') as f:
        for line in f:
            words = re.findall(r'\b\w+\b', line)

    df = pd.read_excel("filler_words.xlsx", engine="openpyxl")
    filler_words = set(df.iloc[:, 0].dropna().values)
    for i in words:
        if i in filler_words:
            return f"Filler Word Detected:{i}"



z = vocabulary_usage("Speech_text.txt")
print(z)
x = filler_words_classification("Speech_text.txt")
print(x)



