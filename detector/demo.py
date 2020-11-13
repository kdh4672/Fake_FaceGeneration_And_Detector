import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
import copy
import time
import numpy as np
from PIL import Image
from FaceAligner import FaceAligner
import cv2
import dlib
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--p_weight', type=str, default='./shape_predictor_5_face_landmarks.dat', help='dlib shape predictor weight')
parser.add_argument('--m_weight', type=str, default='./mmod_human_face_detector.dat', help='dlib face detector weight')
parser.add_argument('--r_weight', type=str, default='./resnet50_phase_no_norm_19_999.pth', help='directory of CNN weight')
parser.add_argument('--dir', type=str, default='./dataset', help='directory of fake/real dataset')

opt = parser.parse_args()

# weight = './mmod_human_face_detector.dat'
predictor = dlib.shape_predictor(opt.p_weight)
detector = dlib.cnn_face_detection_model_v1(opt.m_weight)
# Tensorboard : set path
# 데이터를 한번에 batch_size만큼만 가져오는 dataloader를 만든다.
num_classes = 2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pth = opt.r_weight
model = models.resnet50()
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
model.fc = nn.Linear(2048, 2)
model.to(device)
loaded_state = torch.load(pth,map_location='cuda:0')
model.load_state_dict(loaded_state)

model.eval()
list = ['fake', 'real']
fake = 0
real = 0

f = open('fake_real_report.csv', mode='wt', encoding='utf-8')
f.writelines('path,y')

with torch.no_grad():
    for file in sorted(os.listdir(opt.dir)):
        img = cv2.imread(opt.dir+'/'+file)
        print(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = detector(gray, 1)
        for face in dets:
            fa = FaceAligner(predictor, desiredLeftEye=(0.25, 0.25), desiredFaceWidth=256)
            faceAligned = fa.align(img, gray, face.rect)
            cv2.imwrite('face.jpg', faceAligned)
            img = cv2.imread('face.jpg',0)
            ff = np.fft.fft2(img)
            fshift = np.fft.fftshift(ff)
            phase_spectrum = np.angle(fshift)
#             ## normalize
            x = phase_spectrum
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
            x = (x - 0.5) / 0.5
            x = np.expand_dims(phase_spectrum, axis=0)
#             ## normalize
            predict = model(torch.Tensor(x).unsqueeze(0).to(device))
            predicted_label = torch.max(predict, 1)[1]
            print(list[predicted_label])
        f.writelines('\n')
        if list[predicted_label] == 'fake':
            fake += 1
            f.writelines( file + ',{}'.format(1))
        else:
            real += 1
            f.writelines(file + ',{}'.format(0))
        print("fake:{} real:{}".format(fake, real))

f.close()
