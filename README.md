Network Dissection: Quantifying Interpretability of Deep Visual Representations
===============================================================================
David Bau\*, Bolei Zhou\*, Aditya Khosla, Aude Oliva, Antonio Torralba
https://arxiv.org/abs/1704.05796
CVPR 2017

This directory includes

(1) Code to create the merged Broden dataset from constituent datasets
    ADE, PASCAL, PASCAL Parts, PASCAL Context, OpenSurfaces, and DTD.
    The script makebroden.sh downloads the pieces and generates the broden
    dataset in three resolutions.
    Alternatively, dlbroden.sh just downloads Broden directly.

(2) Code to run network dissection on an arbitrary deep convolutional
    neural network provided as a Caffe deploy.prototxt and .caffemodel.
    The script rundissect.sh runs all the needed phases.

Depends on pycaffe and scipy.
