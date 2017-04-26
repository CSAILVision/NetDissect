#!/usr/bin/env bash
set -e

# Start from parent directory of script
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

declare -a MODELS=(
  "caffe_reference_imagenet"      # alexnet-imagenet
  "caffe_reference_places205"     # alexnet-places205
  "caffe_reference_places365"     # alexnet-places365
  "vgg16_imagenet"                # vgg-imagenet
  "vgg16_places205"               # vgg-places205
  "vgg16_places365"               # vgg-places365
  "googlenet_imagenet"            # googlenet-imagenet
  "googlenet_places205"           # googlenet-places205
  "googlenet_places365"           # googlenet-places365
  "resnet-152-torch-imagenet"     # resnet-imagenet
  "resnet-152-torch-places365"    # resnet-places365
)

for MODEL in "${MODELS[@]}"
do

if [ ! -f zoo/${MODEL}.prototxt ]  || [ ! -f zoo/${MODEL}.caffemodel ]
then

echo "Downloading $MODEL"
mkdir -p zoo
pushd zoo
wget --progress=bar \
   http://netdissect.csail.mit.edu/dissect/zoo/$MODEL.prototxt
wget --progress=bar \
   http://netdissect.csail.mit.edu/dissect/zoo/$MODEL.caffemodel
popd

fi

done
