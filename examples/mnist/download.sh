#!/bin/bash

DST=${1:-$GOPATH/src/gorgonia.org/gorgonia/examples/testdata}
cd "$DST"

curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gunzip t*-ubyte.gz
