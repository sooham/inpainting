# Inpainting
My implementation of the algorithm described in ["Region Filling and Object Removal by Exemplar-Based Image Inpainting"](http://research.microsoft.com/pubs/67276/criminisi_tip2004.pdf "Link to paper") by A. Criminisi et al.

## Properties of the algorithm
The inpainting algorithm prioritizes propagating linear structures flowing into the target region before all other low gradient structures; This results in images free of artifacts and unrealistic textures.

## Example
<p style="text-align:center"><img src="./example.gif" width="300"></p>

## Compiling
You will need both CMake and OpenCV 4 installed. Then, run the following commands:
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make
```

The binary will be located on the bin/ folder. If you just compiled the binary, you may test it with the following command:
```bash
./../bin/inpainting ../inpainting/input-color.png ../inpainting/input-alpha.png
```
