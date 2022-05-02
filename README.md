**Install VSCode**

```
sudo snap install --classic code
```

**Install and Configure Git**:

```
sudo apt install git
git config --global user.name "Krishna"
git config --global user.email "ishna.official@gmail.com"
```

**setup ssh keys for Git**:

```
ssh-keygen -t ed25519 -C  "ishna.official@gmail.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

**Install Additional Tools**:

```
sudo apt install ubuntu-restricted-extras
sudo apt install virtualbox virtualbox-dkms virtualbox-ext-pack virtualbox-guest-additions-iso
sudo apt install curl wget uget unzip tar rar unrar
sudo apt install gimp vlc kdenlive obs-studio
sudo apt install texlive texstudio
sudo apt install kdiff3
```
**Install Chrome** (https://www.google.com/chrome):

```
sudo dpkg -i google-chrome-stable_current_amd64.deb
```

**Install Development Tools**:

```
sudo apt install build-essential pkg-config cmake cmake-qt-gui ninja-build valgrind
```

**Install MinGW Tools**:

```
sudo apt install mingw-w64 mingw-w64-tools
sudo apt install binutils-mingw-w64

sudo update-alternatives --set i686-w64-mingw32-gcc /usr/bin/i686-w64-mingw32-gcc-posix
sudo update-alternatives --set i686-w64-mingw32-g++ /usr/bin/i686-w64-mingw32-g++-posix
 
sudo update-alternatives --set x86_64-w64-mingw32-gcc /usr/bin/x86_64-w64-mingw32-gcc-posix
sudo update-alternatives --set x86_64-w64-mingw32-g++ /usr/bin/x86_64-w64-mingw32-g++-posix
```

**Install ARM Tools**:

```
sudo apt install crossbuild-essential-armel crossbuild-essential-armhf crossbuild-essential-arm64
sudo apt install binutils-multiarch binutils-arm-none-eabi binutils-arm-linux-gnueabi binutils-arm-linux-gnueabihf binutils-aarch64-linux-gnu
```

**Install Python 3 and virtualenv**:

```
sudo apt install python3 python3-wheel python3-pip python3-venv python3-dev python3-setuptools python3-testresources virtualenv virtualenvwrapper
```

**Add virtualenvwrapper entries to .bashrc**

```
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export WORKON_HOME=$HOME/pyenvs
export PROJECT_HOME=$HOME/projects
source /usr/local/bin/virtualenvwrapper.sh
```

**Install NVIDIA drivers**

Launch “Software & Updates.” Under “Additional Drivers” install metapackage from nvidia-driver-510.

```
wget https://mirrors.wikimedia.org/debian/pool/main/libu/liburcu/liburcu6_0.12.2-1_amd64.deb
sudo dpkg -i liburcu6_0.12.2-1_amd64.deb
```

**Install CUDA 11.6**

Download CUDA from NVIDIA using deb (network). Since 22.04 isn’t out yet, use the 20.04 version.

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda
```

**add the following entries to the end of .bashrc**

```
export PATH=/usr/local/cuda-11.6/bin${PATH:+:${PATH}}:/home/krishna/bin
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

**Install third party libraries**

```
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev
```

**Install cuDNN 8.3.2**

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get install libcudnn8=8.3.2.*-1+cuda11.5
sudo apt-get install libcudnn8-dev=8.3.2.*-1+cuda11.5
```

**Install OpenCV 4.5.5**

**Install necessary system packages**
```
sudo apt-get install build-essential cmake pkg-config unzip yasm git checkinstall
sudo apt-get install libjpeg-dev libpng-dev libtiff-dev libopenjp2-7-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt-get install libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev
sudo apt-get install libfaac-dev libvorbis-dev
sudo apt-get install libopencore-amrnb-dev libopencore-amrwb-dev
sudo apt-get install libgtk-3-dev
sudo apt-get install libtbb-dev
sudo apt-get install libatlas-base-dev gfortran
sudo apt-get install libprotobuf-dev protobuf-compiler
sudo apt-get install libgoogle-glog-dev libgflags-dev
sudo apt-get install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen
sudo apt-get install libgtkglext1 libgtkglext1-dev
sudo apt-get install libopenblas-dev liblapacke-dev libva-dev libopenjp2-tools libopenjpip-dec-server libopenjpip-server libqt5opengl5-dev libtesseract-dev 
reboot
```

**Install Ceres Solver**

```
sudo apt-get install cmake libeigen3-dev libgflags-dev libgoogle-glog-dev libsuitesparse-dev libatlas-base-dev
git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
mkdir build && cd build
cmake ..
make -j4
make test
sudo make install
```

**22.04 defaults to gcc version 11.2.0-1ubuntu1 (jammy). If you try to compile OpenCV, you’ll get “error: parameter packs not expanded with” which seems to be due to changes in gcc 11.2. Rather than try to downgrade gcc to 11.1 or edit the header files, I installed gcc-10 and g++-10 and then used update-alternatives to switch the compilers. OpenCV builds fine with gcc-10.**

```
sudo apt-get install gcc-10
sudo apt-get install g++-10
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```

**Create virtualenv for OpenCV, activate virtualenv (should be activated after making it), and install numpy.**

```
mkvirtualenv opencv_cuda
workon opencv_cuda
pip install -U numpy
```

**Setup OpenCV and OpenCV Contrib**

```
mkdir opencvbuild
cd opencvbuild
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.5.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.5.zip
unzip opencv.zip
unzip opencv_contrib.zip
cd opencv-4.5.5
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D WITH_TBB=ON \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D WITH_CUDA=ON \
	-D WITH_CUDNN=ON \
	-D OPENCV_DNN_CUDA=ON \
	-D ENABLE_FAST_MATH=1 \
	-D CUDA_FAST_MATH=1 \
	-D CUDA_ARCH_BIN=8.6 \
	-D WITH_CUBLAS=1 \
	-D WITH_OPENGL=ON \
	-D WITH_QT=ON \
	-D OpenGL_GL_PREFERENCE=LEGACY \
	-D OPENCV_EXTRA_MODULES_PATH=/home/krishna/opencvbuild/opencv_contrib-4.5.5/modules \
	-D PYTHON_DEFAULT_EXECUTABLE=/home/krishna/pyenvs/opencv_cuda/bin/python3.10 \
	-D BUILD_EXAMPLES=ON ..
make -j$(nproc)
sudo make install
sudo ldconfig
cd ~/pyenvs/opencv_cuda/lib/python3.10/site-packages/
ln -s /usr/local/lib/python3.10/site-packages/cv2/python-3.10/cv2.cpython-310-x86_64-linux-gnu.so cv2.so
```

**OpenCV Verification**:

```
workon opencv_cuda
python
import cv2
cv2.__version__
exit()
deativate
```

**Machine Learning Environment**:

```
mkvirtualenv ml
pip install --upgrade pip setuptools wheel
pip install --upgrade numpy scipy matplotlib ipython jupyter pandas sympy nose
pip install --upgrade scikit-learn scikit-image
deactivate
```

**Deep Learning Environment (Tensorflow-CPU)**:

```
mkvirtualenv tfcpu
pip install --upgrade pip setuptools wheel
pip install --upgrade opencv-python opencv-contrib-python
pip install --upgrade tensorflow-cpu tensorboard keras
deactivate
```

**Deep Learning Environment (Tensorflow-GPU)**:

```
mkvirtualenv tfgpu
cd ~/pyenvs/tfgpu/lib/python3.10/site-packages/
ln -s /usr/local/lib/python3.10/site-packages/cv2/python-3.10/cv2.cpython-310-x86_64-linux-gnu.so cv2.so
pip install --upgrade pip setuptools wheel
pip install --upgrade tensorflow-gpu tensorboard keras
deactivate
```

**Tensorflow-GPU Verification**:

```
workon tfgpu
python
>>> from tensorflow.python.client import device_lib
>>> device_lib.list_local_devices()
>>> exit()
deactivate
```

**Deep Learning Environment (PyTorch-CPU)**:

```
mkvirtualenv torchcpu
pip install --upgrade pip setuptools wheel
pip install --upgrade opencv-python opencv-contrib-python
pip install --upgrade torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
deactivate
```

**Deep Learning Environment (PyTorch-GPU)**:

```
mkvirtualenv torchgpu
cd ~/pyenvs/tfgpu/lib/python3.10/site-packages/
ln -s /usr/local/lib/python3.10/site-packages/cv2/python-3.10/cv2.cpython-310-x86_64-linux-gnu.so cv2.so
pip install --upgrade pip setuptools wheel
pip install --upgrade torch torchvision torchaudio
deactivate
```

**PyTorch-GPU Verification**:

```
workon torchgpu
python
>>> import torch
>>> torch.cuda.is_available()
>>> exit()
deactivate
```

**Additional Useful Python Packages**:

```
pip install mkdocs mkdocs-material
```

**Install PyCharm**:

Ubuntu Software > PyCharm

**Enable GPU support for PyCharm Projects**:

- Edit Configurations...
- Environment variables: (File | Settings | Build, Execution, Deployment | Console | Python Console)
	- `PATH=$PATH:/usr/local/cuda-11.6/bin`
	- `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2/lib64:/usr/local/cuda-11.6/extras/CUPTI/lib64`

**Install Miniconda**:

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/bin/activate 
conda config --set auto_activate_base false
conda deactivate
```

Activate and Deactivate:

```
conda activate
conda deactivate
```

Managing conda:

```
conda info
conda update conda
```

Managing environments:

```
conda info --envs
conda create --name devone python=3.10
conda info --envs
conda activate devone
python --version
conda search beautifulsoup4
conda install beautifulsoup4
conda list
conda update beautifulsoup4
conda uninstall beautifulsoup4
conda list
conda deactivate
conda remove --name devone --all
conda info --envs
```

**Razer Keyboard and Mouse drivers**

```
sudo add-apt-repository ppa:openrazer/stable
sudo apt install openrazer-meta
sudo gpasswd -a $USER plugdev
```

**Install and Configure Docker with Nvidia support**

```
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh ./get-docker.sh DRY_RUN=1 -o-
docker --version
sudo docker
sudo systemctl --now enable docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
curl -s -L https://nvidia.github.io/libnvidia-container/experimental/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**Verify Docker**

```
sudo docker run hello-world
```

**Verify Nvidia-Docker-**

```
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

**Install Android Studio**:

```
sudo apt install build-essential libgl1-mesa-dev
sudo apt install openjdk-8-jre openjdk-8-jdk
sudo apt install libstdc++6 libncurses5
sudo apt install libsdl1.2debian
```
Ubuntu Software > Android Studio

**Install Node, npm**:

```
curl https://raw.githubusercontent.com/creationix/nvm/master/install.sh | bash
nvm install 16.15.0
```