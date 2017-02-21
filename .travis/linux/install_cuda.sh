travis_retry wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_${CUDA}_amd64.deb
travis_retry sudo dpkg -i cuda-repo-ubuntu1404_${CUDA}_amd64.deb
travis_retry sudo apt-get update -qq
travis_retry sudo apt-get upgrade -qq
export CUDA_APT=${CUDA:0:3}
export CUDA_APT=${CUDA_APT/./-}
travis_retry sudo apt-get install -f -y nvidia-375 nvidia-375-dev libcuda1-375 nvidia-settings nvidia-opencl-icd-375
travis_retry sudo apt-get install -f -y cuda-drivers cuda-core-${CUDA_APT} cuda-cudart-dev-${CUDA_APT} cuda-cufft-dev-${CUDA_APT}
travis_retry sudo apt-get install cuda
travis_retry sudo apt-get clean
