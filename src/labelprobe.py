'''
labelprobe does the following:
For every sample,
  - Get the ground-truth labels.
  - Upsample all activations, and make a mask at a .01 top cutoff.
  - Accumulate counts of pixels of ground-truth labels (count_g)
  - Accumulate counts of pixels of all activations per-image (count_a)
  - Accumulate counts of intersections between g and a (count_i)
  - Optionally count intesections between each category and a (count_c)
  - Tally top 500 intersection counts per unit per sample (count_t).
'''

import glob
import os
import numpy
import re
import upsample
import time
import loadseg
import signal
import multiprocessing
import multiprocessing.pool
import expdir
import sys
import intersect

# Should we tally intersections between each category (as a whole)
# and each units' activations?  This can be useful if we wish to
# ignore pixels that are unlabeled in a category when computing
# precision (i.e., count cross-label and not label-to-unlabeled
# confusion).  This does not work that well, so we do not tally these.
COUNT_CAT_INTERSECTIONS = False

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NoDaemonPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

def label_probe(directory, blob,
        batch_size=5, ahead=4, tally_depth=2048,
        quantile=0.01, categories=None, verbose=False,
        parallel=0):
    '''
    Evaluates the given blob from the probe stored in the given directory.
    Note that as part of the evaluation, the dataset is loaded again.
    '''
    # Make sure we have a directory to work in
    qcode = ('%f' % quantile).replace('0.', '').rstrip('0')
    ed = expdir.ExperimentDirectory(directory)
    # If the tally file already exists, return.
    if ed.has_mmap(blob=blob, part='tally-%s' % qcode):
        if verbose:
            print 'Have', (ed.mmap_filename(blob=blob, part='tally-%s' % qcode)
                    ), 'already. Skipping.'
        return
    # Load probe metadata
    info = ed.load_info()
    ih, iw = info.input_dim
    # Load blob metadata
    blob_info = ed.load_info(blob=blob)
    shape = blob_info.shape
    unit_size = shape[1]
    fieldmap = blob_info.fieldmap
    # Load the blob quantile data and grab thresholds
    quantdata = ed.open_mmap(blob=blob, part='quant-*',
            shape=(shape[1], -1))
    threshold = quantdata[:, int(round(quantdata.shape[1] * quantile))]
    thresh = threshold[:, numpy.newaxis, numpy.newaxis]
    # Map the blob activation data for reading
    fn_read = ed.mmap_filename(blob=blob)
    # Load the dataset
    ds = loadseg.SegmentationData(info.dataset)
    if categories is None:
        categories = ds.category_names()
    labelcat = onehot(primary_categories_per_index(ds, categories))
    # Be sure to zero out the background label - it belongs to no category.
    labelcat[0,:] = 0
    # Set up output data
    # G = ground truth counts
    # A = activation counts
    # I = intersection counts
    # TODO: consider activation_quantile and intersect_quantile
    # fn_s = os.path.join(directory, '%s-%s-scores.txt' % (blob, qcode))
    # T = tally, a compressed per-sample record of intersection counts.
    fn_t = ed.mmap_filename(blob=blob, part='tally-%s' % qcode,
            inc=True)
    # Create the file so that it can be mapped 'r+'
    ed.open_mmap(blob=blob, part='tally-%s' % qcode, mode='w+',
            dtype='int32', shape=(ds.size(), tally_depth, 3))
    if parallel > 1:
        fast_process(fn_t, fn_read, shape, tally_depth, ds, iw, ih,
                categories, fieldmap,
                thresh, labelcat, batch_size, ahead, verbose, parallel)
    else:
        process_data(fn_t, fn_read, shape, tally_depth, ds, iw, ih,
                categories, fieldmap,
                thresh, labelcat, batch_size, ahead, verbose, False,
                0, ds.size())
    fn_final = re.sub('-inc$', '', fn_t)
    if verbose:
        print 'Renaming', fn_t, 'to', fn_final
    os.rename(fn_t, fn_final)

def fast_process(fn_t, fn_read, shape, tally_depth, ds, iw, ih,
        categories, fieldmap,
        thresh, labelcat, batch_size, ahead, verbose, parallel):
    psize = int(numpy.ceil(float(ds.size()) / parallel))
    ranges = [(s, min(ds.size(), s + psize))
            for s in range(0, ds.size(), psize) if s < ds.size()]
    parallel = len(ranges)
    original_sigint_handler = setup_sigint()
    # pool = multiprocessing.Pool(processes=parallel, initializer=setup_sigint)
    pool = multiprocessing.pool.ThreadPool(processes=parallel)
    restore_sigint(original_sigint_handler)
    # Precache memmaped files
    blobdata = cached_memmap(fn_read, mode='r', dtype='float32',
            shape=shape)
    count_t = cached_memmap(fn_t, mode='r+', dtype='int32',
        shape=(ds.size(), tally_depth, 3))
    data = [
        (fn_t, fn_read, shape, tally_depth, ds, iw, ih, categories, fieldmap,
            thresh, labelcat, batch_size, ahead, verbose, True) + r
        for r in ranges]
    try:
        result = pool.map_async(individual_process, data)
        result.get(31536000)
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
        raise
    else:
        pool.close()
    pool.join()

def individual_process(args):
    process_data(*args)

global_memmap_cache = {}

def cached_memmap(fn, mode, dtype, shape):
    global global_memmap_cache
    if fn not in global_memmap_cache:
        global_memmap_cache[fn] = numpy.memmap(
                fn, mode=mode, dtype=dtype, shape=shape)
    return global_memmap_cache[fn]

def process_data(fn_t, fn_read, shape, tally_depth, ds, iw, ih,
        categories, fieldmap,
        thresh, labelcat, batch_size, ahead, verbose, thread,
        start, end):
    unit_size = len(thresh)
    blobdata = cached_memmap(fn_read, mode='r', dtype='float32',
            shape=shape)
    count_t = cached_memmap(fn_t, mode='r+', dtype='int32',
        shape=(ds.size(), tally_depth, 3))
    count_t[...] = 0
    # The main loop
    if verbose:
         print 'Beginning work for evaluating', blob
    pf = loadseg.SegmentationPrefetcher(ds, categories=categories,
            start=start, end=end, once=True,
            batch_size=batch_size, ahead=ahead, thread=False)
    index = start
    start_time = time.time()
    last_batch_time = start_time
    batch_size = 0
    for batch in pf.batches():
        batch_time = time.time()
        rate = (index - start) / (batch_time - start_time + 1e-15)
        batch_rate = batch_size / (batch_time - last_batch_time + 1e-15)
        last_batch_time = batch_time
        if verbose:
            print 'labelprobe index', index, 'items per sec', batch_rate, rate
            sys.stdout.flush()
        for rec in batch:
            sw, sh = [rec[k] for k in ['sw', 'sh']]
            reduction = int(round(iw / float(sw)))
            up = upsample.upsampleL(fieldmap, blobdata[index],
                    shape=(sh,sw), reduction=reduction)
            mask = up > thresh
            accumulate_counts(
                    mask,
                    [rec[cat] for cat in categories],
                    count_t[index],
                    unit_size,
                    labelcat)
            index += 1
        batch_size = len(batch)
    count_t.flush()

def accumulate_counts(masks, label_list, tally_i, unit_size, labelcat):
    '''
    masks shape (256, 192, 192) - 1/0 for each pixel
    label_list is a list of items whose shape varies:
        () - scalar label for the whole image
        (2) - multiple scalar whole-image labels
        (192, 192) - scalar per-point
        (3, 192, 192) - multiple scalar per-point
    tally_i = (32, 3) - 32x3 list of (count, label, unit) (one per sample)
    '''
    # First convert label_list to scalar-matrix pair
    scalars = []
    pixels = []
    for label_group in label_list:
        shape = numpy.shape(label_group)
        if len(shape) % 2 == 0:
            label_group = [label_group]
        if len(shape) < 2:
            scalars.append(label_group)
        else:
            pixels.append(label_group)
    labels = (
        numpy.concatenate(pixels) if pixels else numpy.array([], dtype=int),
        numpy.concatenate(scalars) if scalars else numpy.array([], dtype=int))
    intersect.tallyMaskLabel(masks, labels, out=tally_i)

def primary_categories_per_index(ds, categories):
    '''
    Returns an array of primary category numbers for each label, where the
    first category listed in ds.category_names is given category number 0.
    '''
    catmap = {}
    for cat in categories:
        imap = ds.category_index_map(cat)
        if len(imap) < ds.label_size(None):
            imap = numpy.concatenate((imap, numpy.zeros(
                ds.label_size(None) - len(imap), dtype=imap.dtype)))
        catmap[cat] = imap
    result = []
    for i in range(ds.label_size(None)):
        maxcov, maxcat = max(
                (ds.coverage(cat, catmap[cat][i]) if catmap[cat][i] else 0, ic)
                for ic, cat in enumerate(categories))
        result.append(maxcat)
    return numpy.array(result)

def onehot(arr, minlength=None):
    '''
    Expands an array of integers in one-hot encoding by adding a new last
    dimension, leaving zeros everywhere except for the nth dimension, where
    the original array contained the integer n.  The minlength parameter is
    used to indcate the minimum size of the new dimension.
    '''
    length = numpy.amax(arr) + 1
    if minlength is not None:
        length = max(minlength, length)
    result = numpy.zeros(arr.shape + (length,), dtype='int64')
    result[list(numpy.indices(arr.shape)) + [arr]] = 1
    return result

def setup_sigint():
    return signal.signal(signal.SIGINT, signal.SIG_IGN)

def restore_sigint(original):
    if original is None:
        original = signal.SIG_DFL
    signal.signal(signal.SIGINT, original)

if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser(
            description='Generate evaluation files for probed activation data.')
        parser.add_argument(
                '--directory',
                default='.',
                help='output directory for the net probe')
        parser.add_argument(
                '--blobs',
                nargs='*',
                help='network blob names to tally')
        parser.add_argument(
                '--batch_size',
                type=int, default=5,
                help='the batch size to use')
        parser.add_argument(
                '--ahead',
                type=int, default=4,
                help='the prefetch lookahead size')
        parser.add_argument(
                '--quantile',
                type=float, default=0.01,
                help='the quantile cutoff to use')
        parser.add_argument(
                '--tally_depth',
                type=int, default=2048,
                help='the number of top label counts to tally for each sample')
        parser.add_argument(
                '--parallel',
                type=int, default=0,
                help='the number of parallel processes to apply')
        args = parser.parse_args()
        for blob in args.blobs:
            label_probe(args.directory, blob, batch_size=args.batch_size,
                    ahead=args.ahead, quantile=args.quantile,
                    tally_depth=args.tally_depth, verbose=True,
                    parallel=args.parallel)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
