#!/usr/bin/env bash
set -e

# Start from parent directory of script
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

# PASCAL 2010 Images
if [ ! -f dataset/pascal/VOC2010/ImageSets/Segmentation/train.txt ]
then

echo "Downloading Pascal VOC2010 images"
mkdir -p dataset/pascal
pushd dataset/pascal
wget --progress=bar \
   http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar \
   -O VOCtrainval_03-May-2010.tar
tar xvf VOCtrainval_03-May-2010.tar
rm VOCtrainval_03-May-2010.tar
mv VOCdevkit/* .
rmdir VOCdevkit
popd

fi


# PASCAL Part dataset
if [ ! -f dataset/pascal/part/part2ind.m ]
then

echo "Downloading Pascal Part Dataset"
mkdir -p dataset/pascal/part
pushd dataset/pascal/part
wget --progress=bar \
   http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/trainval.tar.gz \
   -O trainval.tar.gz
tar xvfz trainval.tar.gz
rm trainval.tar.gz
popd

fi


# PASCAL Context dataset
if [ ! -f dataset/pascal/context/labels.txt ]
then

echo "Downloading Pascal Context Dataset"
mkdir -p dataset/pascal/context
pushd dataset/pascal/context
wget --progress=bar \
   http://www.cs.stanford.edu/~roozbeh/pascal-context/trainval.tar.gz \
   -O trainval.tar.gz
tar xvfz trainval.tar.gz
rm trainval.tar.gz
popd

fi


# DTD
if [ ! -f dataset/dtd/dtd-r1.0.1/imdb/imdb.mat ]
then

echo "Downloading Describable Textures Dataset"
mkdir -p dataset/dtd
pushd dataset/dtd
wget --progress=bar \
   https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz \
   -O dtd-r1.0.1.tar.gz
tar xvzf dtd-r1.0.1.tar.gz
mv dtd dtd-r1.0.1
rm dtd-r1.0.1.tar.gz
popd

fi


# OpenSurfaces
if [ ! -f dataset/opensurfaces/photos.csv ]
then

echo "Downloading OpenSurfaces Dataset"
mkdir -p dataset/opensurfaces
pushd dataset/opensurfaces
wget --progress=bar \
   http://labelmaterial.s3.amazonaws.com/release/opensurfaces-release-0.zip \
   -O opensurfaces-release-0.zip
unzip opensurfaces-release-0.zip
rm opensurfaces-release-0.zip
PROCESS=process_opensurfaces_release_0.py
wget --progress=bar \
  http://labelmaterial.s3.amazonaws.com/release/$PROCESS \
  -O $PROCESS
python $PROCESS
popd

fi


# ADE20K
if [ ! -f dataset/ade20k/index_ade20k.mat ]
then

echo "Downloading ADE20K Dataset"
mkdir -p dataset/ade20k
pushd dataset/ade20k
wget --progress=bar \
   http://groups.csail.mit.edu/vision/datasets/ADE20K/ADE20K_2016_07_26.zip \
   -O ADE20K_2016_07_26.zip
unzip ADE20K_2016_07_26.zip
rm ADE20K_2016_07_26.zip
popd

fi


# Now make broden in various sizes
if [ ! -f dataset/broden1_224/index.csv ]
then
echo "Building Broden1 224"
python src/joinseg.py --size=224
fi

if [ ! -f dataset/broden1_227/index.csv ]
then
echo "Building Broden1 227"
python src/joinseg.py --size=227
fi

if [ ! -f dataset/broden1_384/index.csv ]
then
echo "Building Broden1 384"
python src/joinseg.py --size=384
fi
