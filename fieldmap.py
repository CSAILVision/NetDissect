#!/usr/bin/env python

import os

os.environ['GLOG_minloglevel'] = '2'
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import upsample
import rotate

def print_fieldmap(definition, blob):
    '''
    definition: the filename for the caffe prototxt
    blobs: 'conv3' to probe
    '''
    np = caffe_pb2.NetParameter()
    with open(definition, 'r') as dfn_file:
        text_format.Merge(dfn_file.read(), np)
    # Compute the input-to-layer fieldmap for each blob
    # ind = max(i for i, lay in enumerate(np.layer) if blob in lay.top)
    # path = upsample.shortest_layer_path(np.input, blob, np.layer)
    net = caffe.Net(definition, caffe.TEST)
    fieldmap, path = upsample.composed_fieldmap(np.layer, blob)
    for b, layer in path:
        f, _ = upsample.composed_fieldmap(np.layer, b)
        a = net.blobs[b].data.shape[-2:]
        s = upsample.upsampled_shape(f, a)
        print b, f, a, '->', s
    print blob, fieldmap

if __name__ == '__main__':
    import argparse
    import loadseg

    parser = argparse.ArgumentParser(
        description='Calculate fieldmap for a blob in a caffe network.')
    parser.add_argument(
            '--blob',
            help='network blob name to probe')
    parser.add_argument(
            '--definition',
            help='the deploy prototext defining the net')
    args = parser.parse_args()

    print_fieldmap(args.definition, args.blob)
