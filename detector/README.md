requirment 목록:

1.dlib:

git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build; cd build; cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1; cmake --build .
cd ..
python setup.py install

2. pip install cv2
3. pip install numpy
4. torch version :1.4.0
5. torchvision version :0.5.0
6. pip install matplotlib
7. pip install adamp
