# SIFT implementation

[:arrow_double_down: Project report download link :arrow_double_down:]()

## Guide for compilation and running the code

#### Clone the code with sub-module Vigra

```bash
git clone https://github.com/praveenneuron/SIFT.git
cd SIFT
git submodule update --init --recursive
```
OR

```bash
git clone --recursive https://github.com/praveenneuron/SIFT.git
cd SIFT
git submodule init
git submodule update
```


#### Install dependencies

+ Other dependencies

```bash
sudo apt-get install libjpeg-dev
sudo apt-get install libtiff4-dev
sudo apt-get install libtiff5-dev
sudo apt-get install libpng12-dev
sudo apt-get install openexr
sudo apt-get install libfftw3-dev
```

+ Install Eigen

```bash
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

+ OpenCV (optional if you want to plot images)

```bash
sudo apt-get install libopencv-dev
```

#### Compile and run our code

+ Configure
```bash
cd ~/SIFT
mkdir build
cd build
cmake ..
```

+ Compile code and generate documentation
  + When you run make
    + the html documents will be generated in folder `~/SIFT/html`
    + the latex documents will be generated in folder `~/SIFT/latex`
    + `SIFT` executable file will be generated in folder `~/SIFT/build`
```bash
cd build
make
```


#### Run the code executable

```bash
cd build
./SIFT
```

#### Snapshot of the results

```txt
neuron@neuron-GT70-2PE:~/SIFT/build$ cmake ..
neuron@neuron-GT70-2PE:~/SIFT/build$ make
neuron@neuron-GT70-2PE:~/SIFT/build$ ./SIFT
Setting parameters ...
Loading image ...
Detecting Keypoints ...
Octaves: 5
Gaussian Pyramid built
DOG Pyramid built
Octave: 0 , Interval: 1
Octave: 0 , Interval: 2
Octave: 1 , Interval: 1
Octave: 1 , Interval: 2
Octave: 2 , Interval: 1
Octave: 2 , Interval: 2
Octave: 3 , Interval: 1
Octave: 3 , Interval: 2
Octave: 4 , Interval: 1
Octave: 4 , Interval: 2

Number of keypoints detected: 10972
Number of keypoints rejected 21894
Keypoints computed
Setting Keypoint descriptors parameters...
Allocate memory for descriptor array ...
Print Results ...
--------------------------------------------------------

        Descriptor for some vigra keypoints ...

        Keypoint 0
calculation descriptors ...
Descriptor debug log
                Radius of circle: 21
                Pixels considered inside circle: 598
                Cosine of angle: 0.165292
                Sine of angle: 0.00453269
                Number of rows: 256
                Number of cols: 256
                Descriptor vector with 128 values:
0 0 0 0 0 0 1 0 21 3 17 16 22 17 24 6 34 11 15 13 53 13 29 11 150 5 5 19 136 11 4 25 0 1 0 0 0 0 0 0 32 5 0 5 60 2 1 0 64 17 11 13 114 0 0 0 150 9 4 26 150 0 0 31 0 0 0 0 0 0 0 0 24 10 2 13 74 0 11 9 54 27 36 25 109 9 53 17 150 3 5 41 150 9 6 34 0 6 2 0 0 0 0 0 36 36 17 14 50 2 4 8 65 9 0 16 93 1 5 9 150 1 1 25 150 1 1 22

        Keypoint 2
calculation descriptors ...
Descriptor debug log
                Radius of circle: 21
                Pixels considered inside circle: 628
                Cosine of angle: 0.165292
                Sine of angle: 0.00453269
                Number of rows: 256
                Number of cols: 256
                Descriptor vector with 128 values:
0 0 0 0 0 0 0 0 0 0 1 0 0 0 5 0 0 2 2 0 1 2 7 0 47 1 1 7 43 3 2 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 3 31 10 13 1 1 0 51 12 87 18 46 4 0 13 0 0 1 0 0 0 0 0 0 5 11 0 0 2 5 0 75 24 82 68 125 33 26 41 124 52 125 76 125 87 120 125 0 0 0 0 0 0 0 0 0 2 6 0 0 2 10 0 43 33 17 29 107 104 41 15 107 62 97 125 94 75 109 125

        Keypoint 3
calculation descriptors ...
Descriptor debug log
                Radius of circle: 21
                Pixels considered inside circle: 628
                Cosine of angle: 0.165292
                Sine of angle: 0.00453269
                Number of rows: 256
                Number of cols: 256
                Descriptor vector with 128 values:
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 2 0 0 0 35 0 1 9 33 1 0 9 0 0 0 0 0 0 0 0 0 2 5 0 0 0 0 0 34 8 78 48 121 5 13 21 64 35 121 37 90 45 65 58 0 0 0 0 0 0 1 0 0 4 9 0 0 3 15 0 70 42 22 33 121 107 37 24 121 64 103 121 121 95 114 121 0 0 1 0 0 0 0 0 0 1 10 0 0 1 5 0 5 6 16 28 45 34 18 5 52 58 101 121 83 67 83 85

        Keypoint 4
calculation descriptors ...
Descriptor debug log
                Radius of circle: 19
                Pixels considered inside circle: 513
                Cosine of angle: 0.181944
                Sine of angle: 2.05327e-07
                Number of rows: 256
                Number of cols: 256
                Descriptor vector with 128 values:
0 0 0 0 0 0 0 0 0 0 2 0 0 0 7 0 5 3 9 8 21 15 11 2 34 13 53 46 27 42 92 21 0 0 1 0 0 0 1 0 0 2 11 2 11 1 7 1 37 27 101 119 74 81 43 24 109 52 119 119 41 68 52 108 0 0 0 0 0 0 0 0 0 0 5 0 5 1 5 1 20 31 59 24 74 119 69 12 76 119 119 35 54 119 101 26 0 0 1 0 0 0 1 0 0 0 9 0 0 0 10 0 7 5 25 27 62 23 42 14 14 45 46 52 56 45 119 82

        Keypoint 5
calculation descriptors ...
Descriptor debug log
                Radius of circle: 20
                Pixels considered inside circle: 535
                Cosine of angle: 0.180948
                Sine of angle: 0.00993077
                Number of rows: 256
                Number of cols: 256
                Descriptor vector with 128 values:
0 0 0 0 0 0 0 0 0 1 3 0 0 0 2 0 1 7 33 36 12 5 7 3 30 15 81 74 18 28 50 33 0 0 0 0 0 0 1 0 0 2 7 2 12 3 11 1 50 40 85 104 96 117 58 25 117 90 117 110 55 117 68 89 0 0 1 0 0 0 0 0 0 0 9 0 0 0 5 0 5 31 56 29 71 110 47 10 39 117 111 47 60 117 110 38 0 0 0 0 0 0 1 0 0 0 4 0 0 0 8 0 11 4 19 24 47 13 46 13 21 16 24 34 39 18 117 58

        Keypoint 8
calculation descriptors ...
Descriptor debug log
                Radius of circle: 21
                Pixels considered inside circle: 580
                Cosine of angle: 0.171997
                Sine of angle: 0.00942974
                Number of rows: 256
                Number of cols: 256
                Descriptor vector with 128 values:
0 1 15 0 0 1 10 0 0 16 118 0 0 16 118 4 5 87 118 1 6 90 118 5 16 82 56 1 7 118 85 3 0 1 6 0 0 1 9 1 14 9 73 1 26 26 85 11 22 98 79 2 41 84 97 8 118 91 50 5 90 54 26 0 0 2 0 0 0 0 0 0 29 2 0 3 48 0 0 0 38 16 0 10 75 0 0 4 118 0 0 26 118 2 0 22 0 1 2 0 0 0 0 0 18 7 10 17 29 2 6 4 33 15 20 21 46 5 10 6 118 0 1 27 92 2 0 20

        Keypoint 9
calculation descriptors ...
Descriptor debug log
                Radius of circle: 21
                Pixels considered inside circle: 580
                Cosine of angle: 0.172255
                Sine of angle: 1.58006e-05
                Number of rows: 256
                Number of cols: 256
                Descriptor vector with 128 values:
0 0 20 0 0 0 16 1 0 5 122 1 0 9 122 5 3 84 122 1 4 87 122 5 15 77 59 1 8 113 91 3 0 0 10 0 0 1 11 2 15 3 86 0 26 24 90 13 22 78 81 0 43 75 100 8 122 74 45 0 105 46 23 0 0 0 0 0 0 0 0 0 30 0 0 0 49 0 0 0 40 0 0 4 76 0 0 3 122 0 0 15 122 0 0 24 0 0 1 0 0 0 0 0 16 4 9 13 30 1 5 3 27 4 22 15 43 4 11 5 122 0 1 19 91 3 1 16
neuron@neuron-GT70-2PE:~/SIFT/build$
```

