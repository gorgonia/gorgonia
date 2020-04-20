#!/bin/bash

set -e

DST=$(cd $(dirname $0)/../testdata/mnist; pwd)
mkdir -p $DST 2> /dev/null
cd "$DST"

curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gzip -f -d t*-ubyte.gz
