#!/usr/bin/env bash
set -e

echo "Download AlexNet trained on Places365"
declare -a MODELS=(
  "caffe_reference_places365"     # alexnet-places365
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
