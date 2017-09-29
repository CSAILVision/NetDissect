#!/usr/bin/env python

# Bolei added
import pdb
import torch
import torchvision
from torch.autograd import Variable as V
from torchvision import transforms as trn

import os
import numpy
import glob
import shutil
import codecs
import time
import sys

os.environ['GLOG_minloglevel'] = '2'
from scipy.misc import imresize, imread
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import zoom
from tempfile import NamedTemporaryFile
from contextlib import contextmanager
from collections import namedtuple
import upsample
import rotate
import expdir


def create_probe(
        directory, dataset, definition, weights, mean, blobs,
        colordepth=3,
        rotation_seed=None, rotation_power=1,
        limit=None, split=None,
        batch_size=16, ahead=4,
        cl_args=None, verbose=True):
    # If we're already done, skip it!
    ed = expdir.ExperimentDirectory(directory)
    if all(ed.has_mmap(blob=b) for b in blobs):
        return

    '''
    directory: where to place the probe_conv5.mmap files.
    data: the AbstractSegmentation data source to draw upon
    definition: the filename for the pytorch
    weights: the filename for the weights
    mean: to use to normalize rgb values for the network
    blobs: ['conv3', 'conv4', 'conv5'] to probe
    '''
    if verbose:
        print 'Opening dataset', dataset
    data = loadseg.SegmentationData(args.dataset)

    # the network to dissect
    if args.weights == 'none':
        # load the imagenet pretrained model
        net = torchvision.models.__dict__[args.definition](pretrained=True)
    else:
        # load your own model
        net = torchvision.models.__dict__[args.definition](num_classes=args.num_classes)
        checkpoint = torch.load(args.weights)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].iteritems()} # the data parallel layer will add 'module' before each layer name
        net.load_state_dict(state_dict)
    net.eval()
    # hook up to get the information for each selected layer
    layers = net._modules.keys()
    size_blobs_output = []
    def hook_size(module, input, output):
        size_blobs_output.append(output.data.size())
    input_sample = V(torch.randn(1,3,args.input_size,args.input_size))
    for blob in blobs:
        net._modules.get(blob).register_forward_hook(hook_size)

    output_sample = net(input_sample)

    input_dim = [args.input_size, args.input_size]
    data_size = data.size(split) # the image size
    if limit is not None:
        data_size = min(data_size, limit)

    # Make sure we have a directory to work in
    ed.ensure_dir()

    # Step 0: write a README file with generated information.
    ed.save_info(dict(
        dataset=dataset,
        split=split,
        definition=definition,
        weights=weights,
        mean=mean,
        blobs=blobs,
        input_dim=input_dim,
        rotation_seed=rotation_seed,
        rotation_power=rotation_power))

    # Clear old probe data
    ed.remove_all('*.mmap*')

    # Create new (empty) mmaps
    if verbose:
         print 'Creating new mmaps.'
    out = {}
    rot = None
    if rotation_seed is not None:
        rot = {}
    for idx, blob in enumerate(blobs):
        #shape = (data_size, ) + net.blobs[blob].data.shape[1:]
        shape = (data_size, int(size_blobs_output[idx][1]), int(size_blobs_output[idx][2]),int(size_blobs_output[idx][3]))
        out[blob] = ed.open_mmap(blob=blob, mode='w+', shape=shape)

        # Rather than use the exact RF, here we use some heuristics to compute the approximate RF
        size_RF = (args.input_size/size_blobs_output[idx][2], args.input_size/size_blobs_output[idx][3])
        fieldmap = ((0, 0), size_RF, size_RF)

        ed.save_info(blob=blob, data=dict(
            name=blob, shape=shape, fieldmap=fieldmap))

    # The main loop
    if verbose:
         print 'Beginning work.'
    pf = loadseg.SegmentationPrefetcher(data, categories=['image'],
            split=split, once=True, batch_size=batch_size, ahead=ahead)
    index = 0
    start_time = time.time()
    last_batch_time = start_time
    batch_size = 0
    net.cuda()

    # hook the feature extractor
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())
    for blob in blobs:
        net._modules.get(blob).register_forward_hook(hook_feature)

    for batch in pf.tensor_batches(bgr_mean=mean):
        del features_blobs[:] # clear up the feature basket
        batch_time = time.time()
        rate = index / (batch_time - start_time + 1e-15)
        batch_rate = batch_size / (batch_time - last_batch_time + 1e-15)
        last_batch_time = batch_time
        if verbose:
            print 'netprobe index', index, 'items per sec', batch_rate, rate
            sys.stdout.flush()
        inp = batch[0]
        batch_size = len(inp)
        if limit is not None and index + batch_size > limit:
            # Truncate last if limited
            batch_size = limit - index
            inp = inp[:batch_size]
        if colordepth == 1:
            inp = numpy.mean(inp, axis=1, keepdims=True)
        # previous feedforward case
        inp = inp[:,::-1,:,:]
        inp_tensor = V(torch.from_numpy(inp.copy()))
        inp_tensor.div_(255.0*0.224) # hack: approximately normalize the input to make the images scaled at around 1.
        inp_tensor = inp_tensor.cuda()
        result = net.forward(inp_tensor)
        # output the hooked feature
        for i, key in enumerate(blobs):
            out[key][index:index+batch_size] = numpy.copy(features_blobs[i][:batch_size])
        # print 'Recording data in mmap done'
        index += batch_size
        if index >= data_size:
            break
    assert index == data_size, (
            "Data source should return evey item once %d %d." %
            (index, data_size))
    if verbose:
        print 'Renaming mmaps.'
    for blob in blobs:
        ed.finish_mmap(out[blob])

    # Final step: write the README file
    write_readme_file([
        ('cl_args', cl_args),
        ('data', data),
        ('definition', definition),
        ('weight', weights),
        ('mean', mean),
        ('blobs', blobs)], ed, verbose=verbose)


def ensure_dir(targetdir):
    if not os.path.isdir(targetdir):
        try:
            os.makedirs(targetdir)
        except:
            print 'Could not create', targetdir
            pass

def write_readme_file(args, ed, verbose):
    '''
    Writes a README.txt that describes the settings used to geenrate the ds.
    '''
    with codecs.open(ed.filename('README.txt'), 'w', 'utf-8') as f:
        def report(txt):
            f.write('%s\n' % txt)
            if verbose:
                print txt
        title = '%s network probe' % ed.basename()
        report('%s\n%s' % (title, '=' * len(title)))
        for key, val in args:
            if key == 'cl_args':
                if val is not None:
                    report('Command-line args:')
                    for ck, cv in vars(val).items():
                        report('    %s: %r' % (ck, cv))
            report('%s: %r' % (key, val))
        report('\ngenerated at: %s' % time.strftime("%Y-%m-%d %H:%M"))
        try:
            label = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
            report('git label: %s' % label)
        except:
            pass

if __name__ == '__main__':
    import sys
    import traceback
    import argparse
    try:
        import loadseg

        parser = argparse.ArgumentParser(description=
            'Probe a pytorch network and save results in a directory.')
        parser.add_argument(
                '--directory',
                default='.',
                help='output directory for the net probe')
        parser.add_argument(
                '--blobs',
                nargs='*',
                help='network blob names to collect')
        parser.add_argument(
                '--definition',
                help='the deploy prototext defining the net')
        parser.add_argument(
                '--weights',
                default=None,
                help='the pretrained weight')
        parser.add_argument(
                '--mean',
                nargs='*', type=float,
                help='mean values to subtract from input')
        parser.add_argument(
                '--dataset',
                help='the directory containing the dataset to use')
        parser.add_argument(
                '--split',
                help='the split of the dataset to use')
        parser.add_argument(
                '--limit',
                type=int, default=None,
                help='limit dataset to this size')
        parser.add_argument(
                '--batch_size',
                type=int, default=64,
                help='the batch size to use')
        parser.add_argument(
                '--input_size',
                type=int, default=224,
                help='the image size input to the network(usually it is 224x224, but alexnet uses 227x227)')
        parser.add_argument(
                '--ahead',
                type=int, default=4,
                help='number of batches to prefetch')
        parser.add_argument(
                '--rotation_seed',
                type=int, default=None,
                help='the seed for the random rotation to apply')
        parser.add_argument(
                '--rotation_power',
                type=float, default=1.0,
                help='the power of hte random rotation')
        parser.add_argument(
                '--colordepth',
                type=int, default=3,
                help='set to 1 for grayscale')
        parser.add_argument(
                '--num_classes',
                type=int, default=365,
                help='the number of classes for the network output(default is 365)')

        args = parser.parse_args()
        create_probe(
            args.directory, args.dataset, args.definition, args.weights,
            numpy.array(args.mean, dtype=numpy.float32), args.blobs,
            batch_size=args.batch_size, ahead=args.ahead, limit=args.limit,
            colordepth=args.colordepth,
            rotation_seed=args.rotation_seed, rotation_power=args.rotation_power,
            split=args.split, cl_args=args, verbose=True)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
