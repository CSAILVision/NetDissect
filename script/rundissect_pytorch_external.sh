# pre-defined setting
WORKDIR=probes
DIR=pytorch_alexnet_imagenet
ARCH='alexnet' # [alexnet,squeezenet1_1,resnet18,...]. It should work for all the models in https://github.com/pytorch/vision/tree/master/torchvision/models
LAYERS="features"
DATASET=dataset/broden1_224
NUMCLASSES=1000

# default setting
THRESHOLD=0.04
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
RESOLUTION=100

# Download the external model and reset the parameters
DIR="pytorch_resnet18_places365"
ARCH="resnet18"
NUMCLASSES=365
LAYERS="layer4"
WEIGHTS=zoo/resnet18_places365.pth.tar
if [ ! -e $WEIGHTS ]
then
    echo "Download the resnet18 trained on Places365"
    wget http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar -O $WEIGHTS
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

# Get rid of slashes in layer names for directory purposes
LAYERA=(${LAYERS//\//-})

# Step 1: do a forward pass over the network specified by the model files.
# The output is e.g.,: "conv5.mmap", "conv5-info.txt"
if [ -z "${FORCE##*probe*}" ] || \
  ! ls $(printf " $WORKDIR/$DIR/%s.mmap" "${LAYERA[@]}") 2>/dev/null
then

echo 'Testing activations'
python src/netprobe_pytorch.py \
    --directory $WORKDIR/$DIR \
    --blobs $LAYERS \
    --mean $MEAN \
    --definition $ARCH \
    --weights $WEIGHTS \
    --num_classes $NUMCLASSES \
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
python src/quantprobe.py \
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
python src/labelprobe.py \
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
python src/maxprobe.py \
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
# want to derive.  That gets summarized in a [layer]-result.csv file.
if [ -z "${FORCE##*result*}" ] || \
  ! ls $(printf " $WORKDIR/$DIR/%s-result.csv" "${LAYERA[@]}")
then

echo 'Generating result.csv'
python src/makeresult.py \
    --directory $WORKDIR/$DIR \
    --blobs $LAYERS

[[ $? -ne 0 ]] && exit $?

echo makeresult > $WORKDIR/$DIR/job.done
fi

if [ -z "${ENDAFTER##*result*}" ]
then
  exit 0
fi

# Step 6: now generate the HTML visualization and images.
if [ -z "${FORCE##*report*}" ] || \
  ! ls $(printf " $WORKDIR/$DIR/html/%s.html" "${LAYERA[@]}") || \
  ! ls $(printf " $WORKDIR/$DIR/html/image/%s-bargraph.svg" "${LAYERA[@]}")
then

echo 'Generating report'
python src/report.py \
    --directory $WORKDIR/$DIR \
    --blobs $LAYERS \
    --threshold ${THRESHOLD}

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
python src/graphprobe.py \
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

