import os
import numpy
import re
import shutil
import time
import expdir
import sys
from scipy.io import savemat

def max_probe(directory, blob, batch_size=None):
    # Make sure we have a directory to work in
    ed = expdir.ExperimentDirectory(directory)

    # If it's already computed, then skip it!!!
    print "Checking", ed.mmap_filename(blob=blob, part='imgmax')
    if ed.has_mmap(blob=blob, part='imgmax'):
        print "Already have %s-imgmax.mmap, skipping." % (blob)
        return

    # Read about the blob shape
    blob_info = ed.load_info(blob=blob)
    shape = blob_info.shape

    print 'Computing imgmax for %s shape %r' % (blob, shape)
    data = ed.open_mmap(blob=blob, shape=shape)
    imgmax = ed.open_mmap(blob=blob, part='imgmax',
            mode='w+', shape=data.shape[:2])

    # Automatic batch size selection: 64mb batches
    if batch_size is None:
        batch_size = max(1, int(128 * 1024 * 1024 / numpy.prod(shape[1:])))

    # Algorithm: one pass over the data
    start_time = time.time()
    last_batch_time = start_time
    for i in range(0, data.shape[0], batch_size):
        batch_time = time.time()
        rate = i / (batch_time - start_time + 1e-15)
        batch_rate = batch_size / (batch_time - last_batch_time + 1e-15)
        last_batch_time = batch_time
        print 'Imgmax %s index %d: %f %f' % (blob, i, rate, batch_rate)
        sys.stdout.flush()
        batch = data[i:i+batch_size]
        imgmax[i:i+batch_size] = batch.max(axis=(2,3))
    print 'Writing imgmax'
    sys.stdout.flush()
    # Save as mat file
    filename = ed.filename('imgmax.mat', blob=blob)
    savemat(filename, { 'imgmax': imgmax })
    # And as mmap
    ed.finish_mmap(imgmax)


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        import loadseg

        parser = argparse.ArgumentParser(
            description='Generate sorted files for probed activation data.')
        parser.add_argument(
                '--directory',
                default='.',
                help='output directory for the net probe')
        parser.add_argument(
                '--blobs',
                nargs='*',
                help='network blob names to sort')
        parser.add_argument(
                '--batch_size',
                type=int, default=None,
                help='the batch size to use')
        args = parser.parse_args()
        for blob in args.blobs:
            max_probe(args.directory, blob, args.batch_size)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
