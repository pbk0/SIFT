# SIFT
SIFT implementation


## to get the code

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

## Compile and run the code

#### Compile and install submodules first

+ vigra

```bash
sudo apt-get install libjpeg-dev
sudo apt-get install libtiff4-dev
sudo apt-get install libtiff5-dev
sudo apt-get install libpng12-dev
sudo apt-get install openexr
sudo apt-get install libfftw3-dev
```
```bash
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

Note: When you use gcc 4.8.1, make sure to change the optimization
level to -O2 in the cmake configuration (this is best done in the
cmake GUI that you get by calling ccmake . before invoking make).
The -O3 level in that compiler is buggy and leads to crashes.

+ CudaSift

```bash
# git submodule add https://github.com/praveenneuron/CudaSift.git
cd CudaSift
make
./cudasift
```

+ GistSift

  + [link](https://gist.github.com/lxc-xx/7088609#file-sift-cpp)

```bash
cd ~/SIFT
mkdir GistSift
cd GistSift
wget https://gist.githubusercontent.com/lxc-xx/7088609/raw/a638f0d879fd39f7680c478503217a9e61e05c19/sift.cpp

```

+ opensift

```bash
# git submodule add https://github.com/praveenneuron/opensift.git
cd opensift
mkdir build
cd build
cmake ..
make
./cudasift
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
