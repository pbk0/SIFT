# SIFT

## Installation

#### Clone the code with sub-module Vigra

```bat
git clone https://github.com/praveenneuron/SIFT.git
cd SIFT
git submodule update --init --recursive
```
OR

```bat
git clone --recursive https://github.com/praveenneuron/SIFT.git
cd SIFT
git submodule init
git submodule update
```


#### Install dependencies

+ Other dependencies

```sh
sudo apt-get install libjpeg-dev
sudo apt-get install libtiff4-dev
sudo apt-get install libtiff5-dev
sudo apt-get install libpng12-dev
sudo apt-get install openexr
sudo apt-get install libfftw3-dev
```

+ Install Eigen

```sh
hg clone https://bitbucket.org/eigen/eigen/
cd eigen
mkdir build
cmake ..
make
sudo make install
```
+ Compile and install Vigra submodules first

Note: When you use gcc 4.8.1, make sure to change the optimization
level to -O2 in the cmake configuration (this is best done in the
cmake GUI that you get by calling ccmake . before invoking make).
The -O3 level in that compiler is buggy and leads to crashes.

```sh
# git submodule add https://github.com/praveenneuron/vigra.git
cd vigra
mkdir build
cd build
ccmake ..
make
make check
make doc
make examples
sudo make install
```

+ OpenCV (optional if you want to plot images)

```sh
sudo apt-get install libopencv-dev
```

#### Compile and run our code

```bash
cd ~/SIFT
mkdir build
cd build
cmake ..
make
./SIFT
```
