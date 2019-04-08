
#rm main.cu.o
#rm cgs_lab 
#/usr/local/cuda-6.0/bin/nvcc -ccbin g++ -I/usr/local/cuda/samples/common/inc  -m64     -o main.cu.o -c main.cu
#/usr/local/cuda-6.0/bin/nvcc -ccbin g++ -m64 -o cgs_lab main.cu.o  -lcublas -lcusparse

rm cgs.cu.o
rm cgs_lab 
nvcc -ccbin g++ -I/usr/local/cuda/samples/common/inc  -m64     -o cgs.cu.o -c cgs.cu
nvcc -ccbin g++ -m64 -o cgs_lab cgs.cu.o  -lcublas -lcusparse



