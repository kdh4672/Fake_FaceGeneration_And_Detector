import os
file_path = '/home/capstone_ai1/kong/processed/fake/fake_256/' # face alignment 된 데이터들이 있는 dir
list = os.listdir(file_path)
import cv2
import numpy as np
from matplotlib import pyplot as plt
count = 0

save_path = '/home/capstone_ai1/kong/fft_processed/fake' # phase .npy 파일들을 저장할 dir

for file in list:
	img = cv2.imread(file_path+file,0)
	f = np.fft.fft2(img)
	fshift = np.fft.fftshift(f)
	phase_spectrum = np.angle(fshift) ## 1 chnnel 짜리 phase 
	np.save(save_path+'{}.npy'.format(count),phase_spectrum)
	print(count)
