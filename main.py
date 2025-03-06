import proctoring
import fileProcessing
import transcription

# Split interview application into video and audio
fileProcessing.split_video_audio()


################################################################### Video ####################################
############Iris Tracker###########
iris_tracking = proctoring.iris_tracking()

############Emotion Detection using face##############
predict_emotion = proctoring.predict()

############Full body tracking##################
full_body_tracking = proctoring.body_tracking()


################################################################### Audio ####################################
######## Text2Speech##########
text2speech = transcription.speech_text()

#######  Filler word classification ###########
filler_words = transcription.filler_words_classification()



