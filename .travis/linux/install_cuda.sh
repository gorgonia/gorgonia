travis_retry wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_${CUDA}_amd64.deb
travis_retry sudo dpkg -i cuda-repo-ubuntu1404_${CUDA}_amd64.deb
travis_retry sudo apt-get update -qq
travis_retry sudo apt-get install -f -y cuda
travis_retry sudo apt-get clean
