# Fake_FaceGeneration_And_Detector

원본이미지만을 갖고 binary(Real/False) classification을 쓰기엔 무리가 있다고 판단하여, (실제로 resnet50에 원본이미지를 bce로 훈련시킬경우 정확도가 50%밖에 안나옴)

따라서 이미지의 외형보단 내재적인 특성을 추출하기 위해 Fourier Transformation을 이용하기로 함.

- 푸리에 역변환을 통한 Phase와 Spectrum의 이미지에서의 중요도 비교

![phase_spectrum비교](https://user-images.githubusercontent.com/54311546/99070600-e12c7580-25f3-11eb-800b-67b5dd136f5e.png)

이미지 보존에 더 중요한 정보를 포함하고 있는 phase를 추출하여 학습에 이용하기로 함.


- Fake Face FFT Outputs

![fake_phase](https://user-images.githubusercontent.com/54311546/99070220-2b612700-25f3-11eb-9a85-77f76e89c372.png)

- Real Face FFT Outputs

![real_phase](https://user-images.githubusercontent.com/54311546/99070225-2c925400-25f3-11eb-8176-12213084e69e.png)

이렇게 fft를 이용하여 phase spectrum을 추출한 뒤 학습에 이용했을 때 test_dataset에 대해 accuracy가 약 80% 나와서 그 효과를 확인할 수 있었음.

CNN 모델: resnet50

Optimizer: AdamP

Learnig_rate: 0.001로 시작하여 4500 iteration 마다 x 0.1

Batch_size: 64

GPU: RTX 2080


-fake/real face 데이터 출처 : https://dacon.io/competitions/official/235655/overview/
