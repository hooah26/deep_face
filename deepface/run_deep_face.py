from deepface import DeepFace
from datetime import datetime
from os import path
import numpy as np
import pandas as pd
import face_recognition
import cv2

def init_info():
    # 백엔드 정보
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
    # 모델 정보
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
    # 메트릭스 정보
    metrics = [
        "cosine",
        "euclidean",
        "euclidean_l2"
    ]

    # DB 경로
    db_path = "/data/deep_face/db"
    # 백엔드 선택
    backend_name = backends[0]
    # 모델 선택
    model_name = models[0]
    # 메트릭스 선택
    metric_name = metrics[0]

    # 기존에 가중치 파일 정보 불러오기
    representations = f"representations_{model_name}.pkl"
    representations = representations.replace("-", "_").lower()

    # 기존 가중치 정보가 없으면 초기값으로 생성
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
    return [db_path, backend_name, model_name, metric_name]


def identification(info_list, limit):
    try:
        # 얼굴 이미지 입력
        img_input = input("img:")
    except Exception as e:
        print(f"{e}, input image error")
        img_input = ''

    # 테스트용
    if img_input == "q!":
        print("===end===")
        return None, []

    # 테스트용
    img = "/data/deep_face/test_" + img_input + ".jpg"
    print(img)

    try:
        # 전달 받은 이미지에서 얼굴이 존재하는지 판단
        face_locations = face_recognition.face_locations(face_recognition.load_image_file(img))
    except Exception as e:
        print(f"{e}, face detection error")
        face_locations = []
        return None, []

    # 한명의 얼굴이 존재 한다면 모델 실행
    # 만약 한명 이상의 얼굴이 포함 되는 환경 이라면 로직 수정
    if len(face_locations) == 1:
        try:
            # DB에서 전달 받은 얼굴이미지와 가장 유사한 값들 inference
            dfs = DeepFace.find(
                img_path=img,
                db_path=info_list[0],
                detector_backend=info_list[1],
                model_name=info_list[2],
                distance_metric=info_list[3],
                enforce_detection=False,
                silent=True
            )
        except Exception as e:
            print(f"{e}, DeepFace.find error that find similarly face from DB")
            return None, []

        # 결과값중 유사도?가 먼것들은 제외
        df = dfs[0][dfs[0][f'{info_list[2]}_{info_list[3]}'] < limit]

        # 본인(대상)과 비슷한 얼굴이 존재하지 않을경우
        if len(df) == 0:
            # # 해결방안 1: 추가 로직 구성
            # return
            # 해결방안 2: 유사도 제약조건 삭제
            df = dfs[0][['identity']]

        df['identity'] = df['identity'].apply(lambda x: x.split('/')[-1].split('_')[0])
        df = df.drop_duplicates(['identity'], ignore_index=True)

        return img, list(df['identity'])

    else:
        print("has no face or not detected")
        return None, []


def select_id(info_list, img, ids):
    print(ids)
    # 선택지중 본인(대상)정보 선택하는 과정
    id = input("Select your id(name): ")

    # 본인(대상)정보가 없을경우
    if id == "no!":
        # 본인(대상)의 정보(이름)입력
        id = input("What's Your name?: ")
    # 본인(대상)정보가 있을경우
    else:
        pass
    print(f"Thanks for {id}.")

    # 정보 DB에 추가하는과정
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    cv2.imwrite(f"{info_list[0]}/{id}_{str(datetime.now().strftime('%Y%m%d%H%M%S'))}.jpg", img)

    # 추가한 이미지에 대하여 다시 가중치 생성하는 로직 수정해야함
    # 삭제후 다시 만들지 추가된 이미지만 append할지 고민중
    DeepFace.find(
        img_path=np.zeros([224, 224, 3]),
        db_path=info_list[0],
        detector_backend=info_list[1],
        model_name=info_list[2],
        distance_metric=info_list[3],
        enforce_detection=False,
        silent=False
    )


if __name__ == '__main__':
    info = init_info()
    img, result = identification(info, 0.2)
    if img is not None and result != []:
        select_id(info, img, result)
