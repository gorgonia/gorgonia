@echo off

setlocal
set PWD=%~dp0
mkdir ..\testdata\mnist 2>NUL
cd /D..\testdat\mnista

curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gzip -f -d t*-ubyte.gz

cd /D %PWD%
