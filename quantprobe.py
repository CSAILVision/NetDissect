import os
import numpy
import re
import shutil
import time
import expdir
import sys
import vecquantile

def quant_probe(directory, blob, quantiles=None, batch_size=None):
    '''
    Adds a [blob]-sort directory to a probe such as conv5-sort,
    where the directory contains [blob]-[unit].mmap files for
    each unit such as conv5-0143.mmap, where the contents are
    sorted.  Also creates a [blob]-quant-[num].mmap file
    such as conv5-quantile-1000.mmap, where the higest seen
    value for each quantile of each unit is summarized in a
    single (units, quantiles) matrix.
    '''
    # Make sure we have a directory to work in
    ed = expdir.ExperimentDirectory(directory)

    # If it's already computed, then skip it!!!
    print "Checking", ed.mmap_filename(blob=blob, part='quant-%d' % quantiles)
    if ed.has_mmap(blob=blob, part='quant-%d' % quantiles):
        print "Already have %s-quant-%d.mmap, skipping." % (blob, quantiles)
        return

    # Read about the blob shape
    blob_info = ed.load_info(blob=blob)
    shape = blob_info.shape

    print 'Computing quantiles for %s shape %r' % (blob, shape)
    data = ed.open_mmap(blob=blob, shape=shape)
    qmap = ed.open_mmap(blob=blob, part='quant-%d' % quantiles,
            mode='w+', shape=(data.shape[1], quantiles))
    linmap = ed.open_mmap(blob=blob, part='minmax',
            mode='w+', shape=(data.shape[1], 2))

    # Automatic batch size selection: 64mb batches
    if batch_size is None:
        batch_size = max(1, int(16 * 1024 * 1024 / numpy.prod(shape[1:])))
    # Algorithm: one pass over the data with a quantile counter for each unit
    quant = vecquantile.QuantileVector(depth=data.shape[1], seed=1)
    start_time = time.time()
    last_batch_time = start_time
    for i in range(0, data.shape[0], batch_size):
        batch_time = time.time()
        rate = i / (batch_time - start_time + 1e-15)
        batch_rate = batch_size / (batch_time - last_batch_time + 1e-15)
        last_batch_time = batch_time
        print 'Processing %s index %d: %f %f' % (blob, i, rate, batch_rate)
        sys.stdout.flush()
        batch = data[i:i+batch_size]
        batch = numpy.transpose(batch,axes=(0,2,3,1)).reshape(-1, data.shape[1])
        quant.add(batch)
    print 'Writing quantiles'
    sys.stdout.flush()
    # Reverse the quantiles, largest first.
    qmap[...] = quant.readout(quantiles)[:,::-1]
    linmap[...] = quant.minmax()

    if qmap is not None:
       ed.finish_mmap(qmap)
    if linmap is not None:
       ed.finish_mmap(linmap)


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
        parser.add_argument(
                '--quantiles',
                type=int, default=1000,
                help='the number of quantiles to summarize')
        args = parser.parse_args()
        for blob in args.blobs:
            quant_probe(args.directory, blob, args.quantiles, args.batch_size)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
