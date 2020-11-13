import cv2
import dlib
import time
import numpy as np
import copy
import os
from FaceAligner import FaceAligner
import torchvision
predictor=dlib.shape_predictor( './shape_predictor_5_face_landmarks.dat' )
weight='./mmod_human_face_detector.dat'
face_detector=dlib.cnn_face_detection_model_v1( weight )
ALL=list( range( 0 , 5 ) )
dir = '/home/daehyeon/hdd/deepfake_1st/fake/'
train_data = torchvision.datasets.ImageFolder(root=dir,) # 혹은 os.list_dir(dir)로도 data list 설정가능
count = 0

for i in range(len(train_data.imgs)):
    path = train_data.imgs[i][0]
    image = cv2.imread(path)
    # Create a HOG face detector using the built-in dlib class
    # Load the image into an array

    start=time.time()
    try: faces_cnn=face_detector( image , 1 )
    except: continue
    count += 1
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for face in faces_cnn :
        fa = FaceAligner(predictor,desiredLeftEye=(0.25, 0.25), desiredFaceWidth=256) # 얼굴 Crop Size에서 눈 사이 간격과 width 비율 = 1: 1-0.25-0.25 = 1:0.5
        faceAligned = fa.align(image,gray,face.rect)
        cv2.imwrite('/home/daehyeon/hdd/processed/fake_256/{}.jpg'.format(count), faceAligned)
        # cv2.imshow("Aligned", faceAligned)
        end=time.time()
        print('{}개 중 {}번째 이미지'.format(len(train_data.imgs),count), '걸린시간:' , format( end - start , '.2f' ) )
        cv2.waitKey()
        cv2.destroyAllWindows()


