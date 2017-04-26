#!/usr/bin/env bash
set -e

# Start from parent directory of script
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

# Download broden1_224
if [ ! -f dataset/broden1_224/index.csv ]
then

echo "Downloading broden1_224"
mkdir -p dataset
pushd dataset
wget --progress=bar \
   http://netdissect.csail.mit.edu/data/broden1_224.zip \
   -O broden1_224.zip
unzip broden1_224.zip
rm broden1_224.zip
popd

fi

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

# Download broden1_384
if [ ! -f dataset/broden1_384/index.csv ]
then

echo "Downloading broden1_384"
mkdir -p dataset
pushd dataset
wget --progress=bar \
   http://netdissect.csail.mit.edu/data/broden1_384.zip \
   -O broden1_384.zip
unzip broden1_384.zip
rm broden1_384.zip
popd

fi

