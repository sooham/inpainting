# Inpainting
My implementation of the algorithm described in ["Region Filling and Object Removal by Exemplar-Exemplar-Based Image Inpainting"](http://research.microsoft.com/pubs/67276/criminisi_tip2004.pdf "Link to paper") by A. Criminisi et al.

## Properites of the algorithm
The inpainting algorithm prioritizes propagating linear structures flowing into the target region before all other low gradient structures; This results in images free of artefacts and unrealistic textures.

## Example
<p align:"center"><img style="width:50px" src="./example.gif"/></p>
