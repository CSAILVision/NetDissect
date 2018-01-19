# Network Dissection

 ## News (Jan.18, 2018)!

 **We release a light and portable version of Network Dissection in pyTorch at [NetDissect-Lite](https://github.com/CSAILVision/NetDissect-Lite). It is much faster      than this first version and the code structure is cleaned up, without any complex shell commands. It takes about 30 min for a resnet18 model and 2 hours for a          densenet161. If you have questions, please open issues at NetDissect-Lite**
 
## Introduction
This repository contains the demo code for the [CVPR'17 paper](http://netdissect.csail.mit.edu/final-network-dissection.pdf) Network Dissection: Quantifying Interpretability of Deep Visual Representations. You can use this code with naive [Caffe](https://github.com/BVLC/caffe), with matcaffe and pycaffe compiled. We also provide a [PyTorch wrapper](script/rundissect_pytorch.sh) to apply NetDissect to probe networks in PyTorch format. There are dissection results for several networks at the [project page](http://netdissect.csail.mit.edu/).

This code includes

* Code to run network dissection on an arbitrary deep convolutional
    neural network provided as a Caffe deploy.prototxt and .caffemodel.
    The script `rundissect.sh` runs all the needed phases.

* Code to create the merged Broden dataset from constituent datasets
    ADE, PASCAL, PASCAL Parts, PASCAL Context, OpenSurfaces, and DTD.
    The script `makebroden.sh` runs all the needed steps.


## Download
* Clone the code of Network Dissection from github
```
    https://github.com/CSAILVision/NetDissect.git
    cd NetDissect
```
* Download the Broden dataset (~1GB space) and the example pretrained models. 
```
    script/dlbroden_227.sh
    script/dlzoo_example.sh
```

Note that you can run ```script/dlbroden.sh``` to download Broden dataset with images in all three resolution (227x227,224x224,384x384), or run ```script/dlzoo.sh``` to download more CNN models. AlexNet models work with 227x227 image input, while VGG, ResNet, GoogLeNet works with 224x224 image input.

## Run in Caffe
* Run Network Dissection in Caffe to probe the conv5 layer of the AlexNet trained on Places365. Results will be saved to ```dissection/caffe_reference_model_places365/```, in which ```html``` contains the visualization of all the units in a html page and ```conv5-result.csv``` contains the raw predicted labels for each unit. The code takes about 40 mintues to run, and it will generate about 1.5GB intermediate results (mmap) for one layer, which you could delete after the code finishes running.

```
    script/rundissect.sh --model caffe_reference_places365 --layers "conv5" --dataset dataset/broden1_227 --resolution 227
```

* Run Network Dissection to compare three layers of AlexNet trained on ImageNet. Results will be saved to ```dissection/caffe_reference_model_imagenet/```. 

```
    script/rundissect.sh --model caffe_reference_imagenet --layers "conv3 conv4 conv5" --dataset dataset/broden1_227 --resolution 227
```

* If you need to regenerate the Broden dataset from scratch, you can run ```script/makebroden.sh```. The script will download the pieces and merge them.

* Network dissection depends on scipy as well as pycaffe. [Details on installing pycaffe can be found here](http://caffe.berkeleyvision.org/tutorial/interfaces.html#python).

## Run in PyTorch

* Run Network Dissection in PyTorch. Please install [PyTorch](http://pytorch.org/) and [Torchvision](https://github.com/pytorch/vision) first. We provide a [feature extract wrapper](src/netprobe_pytorch.py) for PyTorch. So you could run ```script/rundissect_pytorch.sh``` to probe the existing networks trained on ImageNet in [Torchvision](https://github.com/pytorch/vision/tree/master/torchvision/models).     

```
    script/rundissect_pytorch.sh
```

* Or try ```script/rundissect_pytorch_external.sh``` on a resnet18 trained on [Places365](https://github.com/CSAILVision/places365).

```
    script/rundissect_pytorch_external.sh
```

## Report
* At the end of the dissection script, a report will be generated that summarizes the semantics of the networks.  For example, after you have tested the conv5 layer of caffe_reference_places365, you will have:

```
    dissection/caffe_reference_places365/html/conv5.html
    dissection/caffe_reference_places365/html/image/conv5-bargraph.svg
    dissection/caffe_reference_places365/html/image/conv5-0[###].png    
    dissection/caffe_reference_places365/conv5-result.csv
```

These are, respectively, the HTML-formatted report, the semantics of the units of the layer summarized as a bar graph, visualizations of all the units of the layer (using zero-indexed unit numbers), and a CSV file containing raw scores of the top matching semantic concepts in each category for each unit of the layer.

* Dissect results of all the existing networks in mat format. After the csv file containing the raw data of the unit semantics is generated, you could use the sample scripts in ```plot/extract_csv.m``` to plot the figure. ```plot/semantics_cvpr_release.mat``` contains the semantics of all the networks analyzed in the CVPR paper. It will generate a [figure](plot/semantics_allnetwork.pdf) showing the number of unique detectors across different networks.


## Reference 
If you find the codes useful, please cite this paper
```
@inproceedings{netdissect2017,
  title={Network Dissection: Quantifying Interpretability of Deep Visual Representations},
  author={Bau, David and Zhou, Bolei and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
  booktitle={Computer Vision and Pattern Recognition},
  year={2017}
}
```

