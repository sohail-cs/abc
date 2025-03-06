#Import necessary libraries
import csv
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle


# Load video file
video_path = "WIN_20250225_14_08_25_Pro.mp4"
cam = cv2.VideoCapture(video_path)

# Get video properties
img_width = int(cam.get(3))
img_height = int(cam.get(4))
fps = int(cam.get(cv2.CAP_PROP_FPS))

def draw_landmarks(img,results,mp_drawing,mp_face_mesh,mp_holistic):
    # Draw Face landmarks
    if results.face_landmarks:
        face = mp_drawing.draw_landmarks(img, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                         mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                         mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    # Draw Right hand landmarks
    if results.face_landmarks:
        right_hand = mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=1, circle_radius=1))

    # Draw Left hand landmarks
    if results.face_landmarks:
        left_hand = mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=1, circle_radius=1))

    # Draw body landmarks
    if results.face_landmarks:
        body = mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=1))


    return face,right_hand,left_hand,body


def iris_tracking():
    eye_contact_counter = 0

    def get_landmark_coord(landmarks, idx, w, h):
        # Find pixel coordinates of landmarks
        x = int(landmarks.landmark[idx].x * w)
        y = int(landmarks.landmark[idx].y * h)
        return (x, y)

    # Initialize Mediapipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    # Iris and eye landmarks
    Right_iris = [474, 475, 476, 477]
    Left_iris = [469, 470, 471, 472]

    # Corners of both eyes
    Left_eye_left = [33]
    Left_eye_right = [133]

    Right_eye_left = [362]
    Right_eye_right = [263]


    # Define output video
    output_path = "Proctored video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (img_width, img_height))

    while True:
        _, img = cam.read()
        if img is not None:
            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img)

            # convert image back to BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmark in results.multi_face_landmarks:
                    h, w, _ = img.shape

                    # Get iris centres
                    left_iris = np.mean([get_landmark_coord(face_landmark, idx, w, h) for idx in Left_iris],
                                        axis=0).astype(
                        int)
                    right_iris = np.mean([get_landmark_coord(face_landmark, idx, w, h) for idx in Right_iris],
                                         axis=0).astype(int)

                    right_eye_left = get_landmark_coord(face_landmark, Right_eye_left[0], w, h)
                    right_eye_right = get_landmark_coord(face_landmark, Right_eye_right[0], w, h)

                    left_eye_left = get_landmark_coord(face_landmark, Left_eye_left[0], w, h)
                    left_eye_right = get_landmark_coord(face_landmark, Left_eye_right[0], w, h)

                    # Draw iris landmark
                    cv2.circle(img, tuple(left_iris), 3, (0, 255, 0), -1)
                    cv2.circle(img, tuple(right_iris), 3, (0, 255, 0), -1)

                    #Find distance between corners of eyes and iris
                    right_eye_width_left = np.linalg.norm(np.array(right_eye_left) - np.array(tuple(right_iris)))
                    right_eye_width_right = np.linalg.norm(np.array(right_eye_right) - np.array(tuple(right_iris)))

                    # Find distance between corners of eyes and iris
                    left_eye_width_left = np.linalg.norm(np.array(left_eye_left) - np.array(tuple(left_iris)))
                    left_eye_width_right = np.linalg.norm(np.array(left_eye_right) - np.array(tuple(left_iris)))

                    # Find ratio of iris distance from both corners of eye
                    right_ratio = right_eye_width_left / right_eye_width_right

                    if right_ratio > 1.6:
                        print("left")

                        cv2.putText(img, "LEFT", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

                    elif right_ratio < 0.9:
                        print("right")
                        cv2.putText(img, "RIGHT", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

                    elif 1 < right_ratio < 1.5:
                        cv2.putText(img, "EYE CONTACT", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                        eye_contact_counter += 1
                        cv2.putText(img, str(eye_contact_counter), (100, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),
                                    2)
        else:
            break

        out.write(img)

        cv2.imshow("video", img)
        key = cv2.waitKey(100) & 0XFF
        if key == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


def body_tracking():

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    mp_face_mesh = mp.solutions.face_mesh

    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5)as holistic:
        while True:
            _, img = cam.read()
            if img is not None:
                # Convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = holistic.process(img) #Make Detection

                # convert image back to BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                draw_landmarks(img,results,mp_drawing,mp_face_mesh,mp_holistic)# Draw landmarks

            else:
                break

            #number of landmark coordinates
            if results is not None:
                num_coords = len(results.face_landmarks.landmark)+ len(results.pose_landmarks.landmark)

            else:
                break

            cv2.imshow("video", img)
            key = cv2.waitKey(80) & 0XFF
            if key == ord("q"):
                break

    cam.release()
    cv2.destroyAllWindows()

df = pd.read_csv("landmark_coordinates.csv")

#Create independent and dependent variables
x = df.drop(columns=['class'])
y = df['class']


#Split into train test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=1)


#Make pipelines
pipelines = {'lr':make_pipeline(StandardScaler(),LogisticRegression()),
             'rc':make_pipeline(StandardScaler(),RidgeClassifier()),
             'rf':make_pipeline(StandardScaler(),RandomForestClassifier()),
             'gb':make_pipeline(StandardScaler(),GradientBoostingClassifier()),
             }

#Train models

fit_models ={}
for algo,pipeline in pipelines.items():
    model = pipeline.fit(x_train,y_train)
    fit_models[algo] = model


for algo,model in fit_models.items():
    y_pred = model.predict(x_test)
    print(algo,accuracy_score(y_test,y_pred))



with open("emotional_analyser.pkl",'wb')as f:
    pickle.dump(fit_models['gb'],f)




def predict_emotion():
    with open("emotional_analyser.pkl", 'rb') as f:
        model = pickle.load(f)
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    mp_face_mesh = mp.solutions.face_mesh

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            _, img = cam.read()
            if img is not None:
                # Convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = holistic.process(img)  # Make Detection

                # convert image back to BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                draw_landmarks(img,results,mp_drawing,mp_face_mesh,mp_holistic)

            else:
                break

            # number of landmark coordinates
            if results is not None:
                num_coords = len(results.face_landmarks.landmark) + len(results.pose_landmarks.landmark)

            else:
                break

            # Export coordinates
            try:
                # Extract body landmark coordinates
                pose = results.pose_landmarks.landmark
                pose_row = list(
                    np.array(
                        [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                # Extract face landmark coordinates
                face = results.face_landmarks.landmark
                face_row = list(
                    np.array(
                        [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                # Concate rows
                row = pose_row + face_row

                x = pd.DataFrame([row]) #dataframe with landmark from video
                body_langueuge = model.predict(x)[0] #Predict emotion from video
                cv2.putText(img,body_langueuge,(50, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            except:
                break

            cv2.imshow("video", img)
            key = cv2.waitKey(80) & 0XFF
            if key == ord("q"):
                break

    cam.release()
    cv2.destroyAllWindows()


