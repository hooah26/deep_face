
# 수정해야함


import face_recognition
from deepface import DeepFace
import numpy as np
import cv2

from os import path


from datetime import datetime


backends = [
    'opencv',
    'ssd',
    'dlib',
    'mtcnn',
    'retinaface',
    'mediapipe',
    'yolov8',
    'yunet',
]
models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
]
metrics = [
    "cosine",
    "euclidean",
    "euclidean_l2"
]

db_path = "db"
backend_name = backends[0]
model_name = models[0]
metric_name = metrics[0]


representations = f"representations_{model_name}.pkl"
representations = representations.replace("-", "_").lower()

if not path.exists(db_path + "/" + representations):
    DeepFace.find(
        img_path=np.zeros([224, 224, 3]),
        db_path=db_path,
        detector_backend=backend_name,
        model_name=model_name,
        distance_metric=metric_name,
        enforce_detection=False,
        silent=False
        )


# DeepFace.stream(db_path=db_path,enable_face_analysis=False)

cap = cv2.VideoCapture(0)  # webcam
# print(cap.get(cv2.CAP_PROP_FPS))

count = 0

# 0.5s
while cv2.waitKey(125) < 0:
    _, frame = cap.read()

    if frame is None:
        break
    count += 1
    cv2.imshow('frame', frame)

    if count % 4 == 0:
        count=0
        # t = datetime.now()
        face_locations = face_recognition.face_locations(frame)
        # print(datetime.now() - t)
        # print(len(face_locations))

        if len(face_locations) == 1:
            t = datetime.now()

            dfs = DeepFace.find(
                img_path=frame,
                db_path=db_path,
                detector_backend=backend_name,
                model_name=model_name,
                distance_metric=metric_name,
                enforce_detection=False,
                silent=True
            )

            df = dfs[0][dfs[0][f'{model_name}_{metric_name}'] < 0.20]

            if len(df) == 0:
                continue

            df['identity'] = df['identity'].apply(lambda x: x.split('/')[1].split('_')[0])
            df = df.drop_duplicates(['identity'], ignore_index=True)

            # print(f"\n┌{'─' * (len(model_name) + len(metric_name) + 21)}┐")
            # print(f"│ {model_name}-{metric_name} => {datetime.now() - t} │")
            # print(f"└{'─' * (len(model_name) + len(metric_name) + 21)}┘")

            print(f"{list(df['identity'])}")
            # for p in list(df['identity']):
            #     img = cv2.imread(db_path+"/"+p+"_1.jpg")
            #     img = cv2.resize(img, (360, 360))
            #     cv2.imshow(p, img)

            name = input("name: ")

            #초기화 버전
            if name == "no!":
                real_name = input("What's Your name?: ")
                cv2.imwrite(f"db/{real_name}_{str(datetime.now().strftime('%Y%m%d_%H%M%S'))}.jpg", frame)

                DeepFace.find(
                    img_path=np.zeros([224, 224, 3]),
                    db_path=db_path,
                    detector_backend=backend_name,
                    model_name=model_name,
                    distance_metric=metric_name,
                    enforce_detection=False,
                    silent=False
                )
            else:
                print(name)

            # cv2.destroyAllWindows()
