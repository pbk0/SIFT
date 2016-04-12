# SIFT
SIFT implementation



# to get the code

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

# My changes

```bat
git clone https://github.com/praveenneuron/SIFT.git
cd SIFT
git add SET_ENV.bat
git submodule add https://github.com/Celebrandil/CudaSift.git
git add SIFT.sln
cd CudaSift
git add CudaSift.vcxproj
git add CudaSift.vcxproj.filters
cd ../MasterProjectPart1
git add MasterProjectPart1.vcxproj
git add MasterProjectPart1.vcxproj.filters
```

# Some Environment variables

+ Check SET_ENV.bat file to set the paths for OpenCV2 and CUDA 7.5
+ Run it with admin rights
+ and check if all paths are set
+ The above set Paths will be used by this project


