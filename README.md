# Network Dissection: Quantifying Interpretability of Deep Visual Representations

## Introduction
This repository contains the demo code for the [CVPR'17 paper](http://netdissect.csail.mit.edu/final-network-dissection.pdf) Network Dissection: Quantifying Interpretability of Deep Visual Representations. You can use this code with naive [Caffe](https://github.com/BVLC/caffe), with matcaffe and pycaffe compiled. There are dissection results for several networks at the [project page](http://netdissect.csail.mit.edu/).

This code includes

* Code to run network dissection on an arbitrary deep convolutional
    neural network provided as a Caffe deploy.prototxt and .caffemodel.
    The script `rundissect.sh` runs all the needed phases.

* Code to create the merged Broden dataset from constituent datasets
    ADE, PASCAL, PASCAL Parts, PASCAL Context, OpenSurfaces, and DTD.


## Download
* Clone the code of Network Dissection from github
```
    https://github.com/CSAILVision/NetDissect.git
    cd NetDissect
```
* Download the Broden dataset (~1GB space) and the example pretrained models. 
```
    ./dlbroden_227.sh
    ./dlzoo_example.sh
```

Note that you could run ```./dlbroden.sh``` to download Broden dataset with images in all three resolution (227x227,224x224,384x384), or run ```./dlzoo.sh``` to download more CNN models. AlexNet models work with 227x227 image input, while VGG, ResNet, GoogLeNet works with 224x224 image input.

## Run
* Run Network Dissection to compare two layers of AlexNet trained on Places365. Results will be saved to ```dissection/caffe_reference_model_places365/```. It takes about 40 mintues to run.

```
    ./rundissect.sh --model caffe_reference_places365 --layers "conv4 conv5" --dataset dataset/broden1_227 --resolution 227
```


* If you need to regenerate the Broden dataset from scratch, you could run ```./makebroden.sh```. The script will download the pieces and merge them.

## Reference 
If you find the codes useful, please cite this paper
```
@inproceedings{networkdissect2017,
  title={Network Dissection: Quantifying Interpretability of Deep Visual Representations},
  author={Bau, David and Zhou, Bolei and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
  booktitle={Computer Vision and Pattern Recognition},
  year={2017}
}
```

