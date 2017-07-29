'''
viewprobe creates visualizations for a certain eval.
'''

import os
import numpy
import upsample
import loadseg
from scipy.misc import imread, imresize, imsave
from loadseg import normalize_label
import expdir

def open_dataset(ed):
    return loadseg.SegmentationData(ed.load_info().dataset)

def score_tally_stats(ed, ds, layerprobe, verbose=False):
    # First, score every unit
    if verbose:
        print 'Adding tallys of unit/label alignments.'
        sys.stdout.flush()
    categories = ds.category_names()
    # Provides a category for each label in the dataset.
    primary_categories = primary_categories_per_index(ds)
    # tally activations, groundtruth, and intersection
    ta, tg, ti = summarize_tally(ds, layerprobe)
    labelcat = onehot(primary_categories)
    # also tally category-groundtruth-presence
    tc = count_act_with_labelcat(ds, layerprobe)
    # If we were doing per-category activations p, then:
    # c = numpy.dot(p, labelcat.transpose())
    epsilon = 1e-20 # avoid division-by-zero
    # If we were counting activations on non-category examples then:
    # iou = i / (a[:,numpy.newaxis] + g[numpy.newaxis,:] - i + epsilon)
    iou = ti / (tc + tg[numpy.newaxis,:] - ti + epsilon)
    # Let's tally by primary-category.
    ar = numpy.arange(iou.shape[1])
    # actually - let's get the top iou for every category
    # per-category-iou.
    pciou = numpy.array([iou * (primary_categories[ar] == ci)[numpy.newaxis,:]
            for ci in range(len(categories))])
    label_pciou = pciou.argmax(axis=2)
    # name_iou = [ds.name(None, i) for i in label_iou]
    name_pciou = [
            [ds.name(None, j) for j in label_pciou[ci]]
            for ci in range(len(label_pciou))]
    # score_iou = iou[numpy.arange(iou.shape[0]), label_iou]
    score_pciou = pciou[
            numpy.arange(pciou.shape[0])[:,numpy.newaxis],
            numpy.arange(pciou.shape[1])[numpy.newaxis,:],
            label_pciou]
    bestcat_pciou = score_pciou.argsort(axis=0)[::-1]
    # Assign category for each label
    # cat_per_label = primary_categories_per_index(ds)
    # cat_iou = [categories[cat_per_label[i]] for i in label_iou]
    # Now sort units by score and visulize each one
    return bestcat_pciou, name_pciou, score_pciou, label_pciou, tc, tg, ti

def generate_csv_summary(
        layer, csvfile, tally_stats,
        ds, order=None, verbose=False):
    if verbose:
        print 'Generating csv summary', csvfile
        sys.stdout.flush()
    bestcat_pciou, name_pciou, score_pciou, label_pciou, tc, tg, ti = (
            tally_stats)

    # For each unit in a layer, outputs the following information:
    # - label: best interpretation
    # - object-label: top ranked interpretation for scene/object/color/etc
    # - object-truth: ground truth pixels
    # - object-activation: activating pixels
    # - object-intersect: intersecting pixels
    # - object-iou: iou score
    # - etc, for each category.
    categories = ds.category_names()
    csv_fields = sum([[
        '%s-label' % cat,
        '%s-truth' % cat,
        '%s-activation' % cat,
        '%s-intersect' % cat,
        '%s-iou' % cat] for cat in categories],
        ['unit', 'category', 'label', 'score'])

    if order is not None:
        csv_fields = order

    if verbose:
        print 'Sorting units by score.'
        sys.stdout.flush()
    ordering = score_pciou.max(axis=0).argsort()[::-1]

    import csv
    with open(csvfile, 'w') as f:
        writer = csv.DictWriter(open(csvfile, 'w'), csv_fields)
        writer.writeheader()

        for unit in ordering:
            # Top images are top[unit]
            bestcat = bestcat_pciou[0, unit]
            data = {
               'unit': (unit + 1),
               'category': categories[bestcat],
               'label': name_pciou[bestcat][unit],
               'score': score_pciou[bestcat][unit]
            }
            for ci, cat in enumerate(categories):
                label = label_pciou[ci][unit]
                data.update({
                    '%s-label' % cat: name_pciou[ci][unit],
                    '%s-truth' % cat: tg[label],
                    '%s-activation' % cat: tc[unit, label],
                    '%s-intersect' % cat: ti[unit, label],
                    '%s-iou' % cat: score_pciou[ci][unit]
                })
            writer.writerow(data)

def summarize_tally(ds, layerprobe):
    cat_count = len(ds.category_names())
    tally = layerprobe.tally
    unit_size = layerprobe.shape[1]
    label_size = ds.label_size()
    count = numpy.zeros(
            (unit_size + 1, label_size + 1 + cat_count), dtype='int64')
    for i in range(len(tally)):
        t = tally[i]
        count[t[:,0]+1, t[:,1]+1+cat_count] += t[:,2]
    # count_a.shape = (unit size,)
    count_a = count[1:,cat_count]
    # this would summarize category intersections if we tallied them
    # count_c.shape = (unit_size, cat_size)
    # count_c = count[1:,0:cat_count]
    # count_g.shape = (label_size,)
    count_g = count[0,1+cat_count:]
    # count_i.shape = (unit_size, label_size)
    count_i = count[1:,1+cat_count:]
    # return count_a, count_c, count_g, count_i
    return count_a, count_g, count_i

def count_act_with_labelcat(ds, layerprobe):
    # Because our dataaset is sparse, instead of using count_a to count
    # all activations, we can compute count_act_with_labelcat to count
    # activations only within those images which contain an instance
    # of a given label category.
    labelcat = onehot(primary_categories_per_index(ds))
    # Be sure to zero out the background label - it belongs to no category.
    labelcat[0,:] = 0
    tally = layerprobe.tally
    unit_size = layerprobe.shape[1]
    label_size = ds.label_size()
    count = numpy.zeros((unit_size, labelcat.shape[1]), dtype='int64')
    for i in range(len(tally)):
        c1 = numpy.zeros((unit_size + 1, label_size + 1), dtype='int64')
        t = tally[i]
        c1[t[:,0]+1, t[:,1]+1] = t[:,2]
        count += c1[1:,0][:,numpy.newaxis] * (
                numpy.dot(c1[0,1:], labelcat) > 0)
    # retval: (unit_size, label_size)
    return numpy.dot(count, numpy.transpose(labelcat))

class LayerProbe:
    def __init__(self, ed, ds, blob):
        info = ed.load_info(blob=blob)
        self.shape = info.shape
        self.fieldmap = info.fieldmap
        # Load the raw activation data
        if ed.has_mmap(blob=blob):
            self.blobdata = ed.open_mmap(blob=blob, shape=self.shape, mode='r')
        # Load the blob quantile data and grab thresholds
        if ed.has_mmap(blob=blob, part='quant-*'):
            self.quantdata = ed.open_mmap(blob=blob, part='quant-*',
                                shape=(self.shape[1], -1), mode='r')
        # Load tally too; tally_depth is inferred from file size.
        self.tally = ed.open_mmap(blob=blob, part='tally-*', decimal=True,
                shape=(ds.size(), -1, 3), dtype='int32', mode='r')
        # And load imgmax
        if ed.has_mmap(blob=blob, part='imgmax'):
            self.imgmax = ed.open_mmap(blob=blob, part='imgmax',
                    shape=(ds.size(), self.shape[1]), mode='r')
        # Figure out tally level that was used.
        self.level = ed.glob_number(
                'tally-*.mmap', blob=blob, decimal=True)

def primary_categories_per_index(ds, categories=None):
    '''
    Returns an array of primary category numbers for each label, where the
    first category listed in ds.category_names is given category number 0.
    '''
    if categories is None:
        categories = ds.category_names()
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
    result = numpy.zeros(arr.shape + (length,))
    result[list(numpy.indices(arr.shape)) + [arr]] = 1
    return result

if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser(
            description='Generate blob-result.csv for probed activation data.')
        parser.add_argument(
                '--directory',
                default='.',
                help='output directory for the net probe')
        parser.add_argument(
                '--csvorder',
                help='csv header order')
        parser.add_argument(
                '--blobs',
                nargs='*',
                help='network blob names to visualize')
        args = parser.parse_args()
        ed = expdir.ExperimentDirectory(args.directory)
        ds = open_dataset(ed)
        for blob in args.blobs:
            layerprobe = LayerProbe(ed, ds, blob)
            tally_stats = score_tally_stats(ed, ds, layerprobe, verbose=True)
            filename = ed.csv_filename(blob=blob, part='result')
            generate_csv_summary(blob, filename, tally_stats, ds,
                        order=(args.csvorder.split(',')
                            if args.csvorder else None), verbose=True)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
