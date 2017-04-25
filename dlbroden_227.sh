#!/usr/bin/env bash
set -e

# Download broden1_227
if [ ! -f dataset/broden1_227/index.csv ]
then

echo "Downloading broden1_227"
mkdir -p dataset
pushd dataset
wget --progress=bar \
   http://netdissect.csail.mit.edu/data/broden1_227.zip \
   -O broden1_227.zip
unzip broden1_227.zip
rm broden1_227.zip
popd

fi


