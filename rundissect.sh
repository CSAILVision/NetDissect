#!/usr/bin/env bash

# To use this, put the caffe model to be tested in the "zoo" directory
# the following naming convention for a model called "vgg16_places365":
#
# zoo/vgg16_places365.caffemodel
# zoo/vgg16_places365.prototxt
#
# and then, with scipy and pycaffe available in your python, run:
#
# ./rundissect.sh --model vgg16_places365 --layers "conv3_3 conv4_3 conv5_3"
#
# the output will be placed in a directory dissection/vgg_places365/
#
# More options are listed below.

# Defaults
DATASET="dataset/broden1_224"
RESOLUTION="224"
THRESHOLD="0.04"
WORKDIR="dissection"
TALLYDEPTH=2048
PARALLEL=4
TALLYBATCH=16
PROBEBATCH=64
QUANTILE="0.005"
COLORDEPTH="3"
CENTERED="c"
MEAN="0 0 0"
FORCE="none"
ENDAFTER="none"
MODELDIR="zoo"

# Parse command-line arguments. http://stackoverflow.com/questions/192249

while [[ $# -gt 1 ]]
do
key="$1"

case $key in
    -d|--model)
    DIR="$2"
    shift
    ;;
    -w|--weights)
    WEIGHTS="$2"
    shift
    ;;
    -p|--proto)
    PROTO="$2"
    shift
    ;;
    -l|--layers)
    LAYERS="$2"
    shift
    ;;
    -s|--dataset)
    DATASET="$2"
    shift
    ;;
    --colordepth)
    COLORDEPTH="$2"
    shift
    ;;
    --resolution)
    RESOLUTION="$2"
    shift
    ;;
    --probebatch)
    PROBEBATCH="$2"
    shift
    ;;
    -t|--threshold)
    THRESHOLD="$2"
    shift
    ;;
    --tallydepth)
    TALLYDEPTH="$2"
    shift
    ;;
    --tallybatch)
    TALLYBATCH="$2"
    shift
    ;;
    --mean)
    MEAN="$2"
    shift
    ;;
    -w|--workdir)
    WORKDIR="$2"
    shift
    ;;
    -f|--force)
    FORCE="$2"
    shift
    ;;
    --endafter)
    ENDAFTER="$2"
    shift
    ;;
    --gpu)
    export CUDA_VISIBLE_DEVICES="$2"
    shift
    ;;
    *)
    echo "Unknown option" $key
    exit 3
    # unknown option
    ;;
esac
shift # past argument or value
done

# Get rid of slashes in layer names for directory purposes
LAYERA=(${LAYERS//\//-})

# For expanding globs http://stackoverflow.com/questions/2937407
function globexists {
  set +f
  test -e "$1" -o -L "$1";set -f
}

if [ -z $DIR ]; then
  echo '--model directory' must be specified
  exit 1
fi

if [ -z $LAYERS ]; then
  echo '--layers layers' must be specified
  exit 1
fi

# Set up directory to work in, and lay down pid file etc.
mkdir -p $WORKDIR/$DIR
if [ -z "${FORCE##*pid*}" ] || [ ! -e $WORKDIR/$DIR/job.pid ]
then
    exec &> >(tee -a "$WORKDIR/$DIR/job.log")
    echo "Beginning pid $$ on host $(hostname) at $(date)"
    trap "rm -rf $WORKDIR/$DIR/job.pid $WORKDIR/$DIR/job.host" EXIT
    echo $$ > $WORKDIR/$DIR/job.pid
    echo $(hostname) > $WORKDIR/$DIR/job.host
else
    echo "Already running $DIR under pid $(cat $WORKDIR/$DIR/job.pid)"
    exit 1
fi

if [ "$COLORDEPTH" -le 0 ]
then
  (( COLORDEPTH = -COLORDEPTH ))
  CENTERED=""
fi

if [ -z "${CENTERED##*c*}" ]
then
  MEAN="109.5388 118.6897 124.6901"
fi

# Convention: dir, weights, and proto all have the same name
if [[ -z "${WEIGHTS}" && -z "${PROTO}" ]]
then
  WEIGHTS="zoo/$DIR.caffemodel"
  PROTO="zoo/$DIR.prototxt"
fi

echo DIR = "${DIR}"
echo LAYERS = "${LAYERS}"
echo DATASET = "${DATASET}"
echo COLORDEPTH = "${COLORDEPTH}"
echo RESOLUTION = "${RESOLUTION}"
echo WORKDIR = "${WORKDIR}"
echo WEIGHTS = "${WEIGHTS}"
echo PROTO = "${PROTO}"
echo THRESHOLD = "${THRESHOLD}"
echo PROBEBATCH = "${PROBEBATCH}"
echo TALLYDEPTH = "${TALLYDEPTH}"
echo TALLYBATCH = "${TALLYBATCH}"
echo MEAN = "${MEAN}"
echo FORCE = "${FORCE}"
echo ENDAFTER = "${ENDAFTER}"

# Step 1: do a forward pass over the network specified by the model files.
# The output is e.g.,: "conv5.mmap", "conv5-info.txt"
if [ -z "${FORCE##*probe*}" ] || \
  ! ls $(printf " $WORKDIR/$DIR/%s.mmap" "${LAYERA[@]}") 2>/dev/null
then

echo 'Testing activations'
python netprobe.py \
    --directory $WORKDIR/$DIR \
    --blobs $LAYERS \
    --weights $WEIGHTS \
    --definition $PROTO \
    --batch_size $PROBEBATCH \
    --mean $MEAN \
    --colordepth $COLORDEPTH \
    --dataset $DATASET

[[ $? -ne 0 ]] && exit $?
echo netprobe > $WORKDIR/$DIR/job.done
fi

if [ -z "${ENDAFTER##*probe*}" ]
then
  exit 0
fi

# Step 2: compute quantiles
# saves results in conv5-quant-1000.mmap; conv5-rank-1001.mmap inverts this too.
#echo 'Computing quantiles'
if [ -z "${FORCE##*sort*}" ] || \
  ! ls $(printf " $WORKDIR/$DIR/%s-quant-*.mmap" "${LAYERA[@]}") 2>/dev/null
then

echo 'Collecting quantiles of activations'
python quantprobe.py \
    --directory $WORKDIR/$DIR \
    --blobs $LAYERS
[[ $? -ne 0 ]] && exit $?

echo quantprobe > $WORKDIR/$DIR/job.done
fi

if [ -z "${ENDAFTER##*quant*}" ]
then
  exit 0
fi

# Step 3: the output here is the ~1G file called "conv5-tally-005.mmap".
# It contains all the I/O/U etc counts for every label and unit (at the 0.5%
# top activtation mask) for EVERY image.  I.e., for image #n, we can read
# out the number of pixels that light up both unit #u and label #l for
# any combination of (n, u, l).  That is a a very large sparse matrix, so
# we encode that matrix specially.  This is a
# (#images, 2048, 3) dimensional file with entries such as this:
# E.g., for image 412, we will have a list of up to 2048 triples.
# Each triple has (count, #unit, #label) in it.
if [ -z "${FORCE##*tally*}" ] || \
  ! ls $(printf " $WORKDIR/$DIR/%s-tally-*.mmap" "${LAYERA[@]}") 2>/dev/null
then

echo 'Tallying counts'
python labelprobe.py \
    --directory $WORKDIR/$DIR \
    --quantile $QUANTILE \
    --tally_depth $TALLYDEPTH \
    --blobs $LAYERS \
    --parallel $PARALLEL \
    --batch_size $TALLYBATCH \
    --ahead 4

[[ $? -ne 0 ]] && exit $?

echo tallyprobe > $WORKDIR/$DIR/job.done
fi

if [ -z "${ENDAFTER##*tally*}" ]
then
  exit 0
fi

# Step 4: compute conv5-imgmax.mmap / conv5-imgmax.mat
# This contains the per-imgae maximum activation for every unit.
if [ -z "${FORCE##*imgmax*}" ] || \
  ! ls $(printf " $WORKDIR/$DIR/%s-imgmax.mmap" "${LAYERA[@]}") 2>/dev/null
then

echo 'Computing imgmax'
python maxprobe.py \
    --directory $WORKDIR/$DIR \
    --blobs $LAYERS

[[ $? -ne 0 ]] && exit $?
echo maxprobe > $WORKDIR/$DIR/job.done
fi

if [ -z "${ENDAFTER##*imgmax*}" ]
then
  exit 0
fi


# Step 5: we just run over the tally file to extract whatever score we
# want to derive.  That's pretty easy, so we also generate HTML at the
# same time.
if [ -z "${FORCE##*view*}" ] || \
  ! ls $(printf " $WORKDIR/$DIR/html/%s.html" "${LAYERA[@]}") || \
  ! ls $(printf " $WORKDIR/$DIR/%s-result.csv" "${LAYERA[@]}")
then

echo 'Generating views'
python viewprobe.py \
    --directory $WORKDIR/$DIR \
    --format csv,html,quantmat \
    --imscale 72 \
    --imsize $RESOLUTION \
    --blobs $LAYERS

[[ $? -ne 0 ]] && exit $?

echo viewprobe > $WORKDIR/$DIR/job.done
fi

if [ -z "${ENDAFTER##*view*}" ]
then
  exit 0
fi

#
# Step 6: graph the results.
if [ -z "${FORCE##*graph*}" ] || ! globexists $WORKDIR/$DIR/html/*-graph.png
then

# Compute text for labeling graphs.
PERCENT=$(printf '%g%%' $(echo "scale=0; $THRESHOLD * 100" | bc))

echo 'Generating graph'
python graphprobe.py \
    --directories $WORKDIR/$DIR \
    --blobs $LAYERS \
    --labels $LAYERS \
    --threshold $THRESHOLD \
    --include_total true \
    --title "Interpretability By Layer ($PERCENT IOU)" \
    --out $WORKDIR/$DIR/html/layer-graph.png

[[ $? -ne 0 ]] && exit $?
echo graphprobe > $WORKDIR/$DIR/job.done
fi

if [ "$WORKDIR" != "$WORKDIR" ]
then
rm -rf $WORKDIR/$DIR/
[[ $? -ne 0 ]] && exit $?
fi

echo finished > $WORKDIR/$DIR/job.done

if [ -z "${ENDAFTER##*graph*}" ]
then
  exit 0
fi

