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

## already added some external code

+ vigra

```bash
# git submodule add https://github.com/ukoethe/vigra.git
cd vigra
mkdir build
cd build
cmake ..
make
sudo make install
git submodule add https://github.com/Celebrandil/CudaSift.git
```


+ CudaSift

```bash
# git submodule add https://github.com/Celebrandil/CudaSift.git
cd CudaSift
mkdir build
cd build
cmake ..
make
#sudo make install
```


