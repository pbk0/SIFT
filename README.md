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
# git submodule add https://github.com/praveenneuron/vigra.git
cd vigra
mkdir build
cd build
cmake ..
make
sudo make install
```


+ CudaSift

```bash
# git submodule add https://github.com/praveenneuron/CudaSift.git
cd CudaSift
mkdir build
cd build
cmake ..
make
#sudo make install
```


