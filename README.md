Network Dissection: Quantifying Interpretability of Deep Visual Representations
===============================================================================
David Bau\*, Bolei Zhou\*, Aditya Khosla, Aude Oliva, Antonio Torralba  
Massachusetts Institute of Technology  
http://netdissect.csail.mit.edu/  
https://arxiv.org/abs/1704.05796  
CVPR 2017 

This directory includes

(1) Code to create the merged Broden dataset from constituent datasets
    ADE, PASCAL, PASCAL Parts, PASCAL Context, OpenSurfaces, and DTD.
    `dlbroden.sh` just downloads Broden directly; the script `makebroden.sh`
    downloads the pieces and generates the merged broden dataset in three
    resolutions.

(2) Code to run network dissection on an arbitrary deep convolutional
    neural network provided as a Caffe deploy.prototxt and .caffemodel.
    The script `rundissect.sh` runs all the needed phases.

A set of baseline models to test can be downloaded using `dlzoo.sh`.

Depends on pycaffe and scipy.

Example usage:

```
# download some baseline models
./dlzoo.sh

# download Broden1 dataset
./dlbroden.sh

# run network dissection to compare two layers of vgg16_places365
./rundissect.sh --model vgg16_places365 --layers "conv4_3 conv5_3"

# results will be computed in dissection/vgg16_places365/
```
