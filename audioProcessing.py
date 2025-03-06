import librosa
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Conv1D,Flatten, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#Extract the MFCC features
def extract_features(file_path):
    y,sr = librosa.load(file_path,sr=22050)
    mfccs = np.mean(librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13),axis=1)
    return mfccs


#Define emotion labels based on RAVDESS files
emotion_mapping = {"01":"neutral","02":"calm","03":"happy","04":"sad",
                   "05":"angry","06":"fearful","07":"disgust","08":"suprised"}

label_to_int = {v:i for i,v in enumerate(emotion_mapping.values())} #convert mapping to integer

dataset_path = "C:/Users/zsoha/PycharmProjects/InterviewSystem/audio_speech_actors_01-24" #Dataset path

data = []

for root,_,files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            file_dir = os.path.join(root,file)
            file_path = os.path.basename(file) #Get Audio file
            emotion_code = file_path.split("-")[2] #Extract emotion code
            emotion = emotion_mapping.get(emotion_code,None) #Get emotion of Audio file

            features = extract_features(file_dir)
            data.append((features,label_to_int[emotion]))

#Convert data into pandas dataframe
df = pd.DataFrame(data,columns=["features","label"])
df["features"] = df["features"].apply(lambda x:np.array(x)) #Convert list values to numpy array

#split into independent and dependent variables
x = np.array(df['features'].tolist()) #convert to 2d array
y = np.x = np.array(df['label'].tolist())

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.25,random_state=1)

#Reshape fpr CNN input
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

#CNN Model
model = Sequential([Conv1D(64,kernel_size=3,activation='relu',input_shape=(x_train.shape[1],1)),
                    Conv1D(128,kernel_size=3,activation='relu'),
                    Flatten(),
                    Dense(64,activation='relu'),
                    Dropout(0.3),
                    Dense(len(label_to_int),activation='softmax')
                    ])
import numpy as np

print(f"x_train type: {type(x_train)}, shape: {x_train.shape}")
print(f"y_train type: {type(y_train)}, shape: {y_train.shape}, dtype: {y_train.dtype}")

# Ensure labels are integer type (important for sparse categorical cross-entropy)
y_train = np.array(y_train, dtype=np.int32)
y_test = np.array(y_test, dtype=np.int32)


#Compile the model
model.compile(optimizer="adam",loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#Train the model
model.fit(x_train,y_train,epochs=30,batch_size=16,validation_data=(x_test,y_test))

_,accuracy = model.evaluate(x_test,y_test)
print(f"Test Accuracy:{accuracy:.2f}")




