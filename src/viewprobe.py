'''
viewprobe creates visualizations for a certain eval.
'''

import glob
import os
import numpy
import re
import upsample
import time
import loadseg
from scipy.misc import imread, imresize, imsave
from loadseg import normalize_label
import expdir

class NetworkProbe:
    def __init__(self, directory, blobs=None):
        self.ed = expdir.ExperimentDirectory(directory)
        # Load probe metadata
        info = self.ed.load_info()
        self.ih, self.iw = info.input_dim
        self.layers = info.blobs if blobs is None else blobs
        self.ds = loadseg.SegmentationData(info.dataset)
        self.layer = {}
        for blob in self.layers:
            self.layer[blob] = LayerProbe(self.ed, blob, self.ds)

    def score_tally_stats(self, layer, verbose=False):
        # First, score every unit
        if verbose:
            print 'Adding tallys of unit/label alignments.'
            sys.stdout.flush()
        ta, tg, ti = self.summarize_tally(layer)
        labelcat = onehot(primary_categories_per_index(self.ds))
        tc = np.count_act_with_labelcat(layer)
        # If we were doing per-category activations p, then:
        # c = numpy.dot(p, labelcat.transpose())
        epsilon = 1e-20 # avoid division-by-zero
        # If we were counting activations on non-category examples then:
        # iou = i / (a[:,numpy.newaxis] + g[numpy.newaxis,:] - i + epsilon)
        iou = ti / (tc + tg[numpy.newaxis,:] - ti + epsilon)
        # Let's tally by primary-category.
        pc = primary_categories_per_index(self.ds)
        categories = self.ds.category_names()
        ar = numpy.arange(iou.shape[1])
        # actually - let's get the top iou for every category
        pciou = numpy.array([iou * (pc[ar] == ci)[numpy.newaxis,:]
                for ci in range(len(categories))])
        # label_iou = iou.argmax(axis=1)
        label_pciou = pciou.argmax(axis=2)
        # name_iou = [self.ds.name(None, i) for i in label_iou]
        name_pciou = [
                [self.ds.name(None, j) for j in label_pciou[ci]]
                for ci in range(len(label_pciou))]
        # score_iou = iou[numpy.arange(iou.shape[0]), label_iou]
        score_pciou = pciou[
                numpy.arange(pciou.shape[0])[:,numpy.newaxis],
                numpy.arange(pciou.shape[1])[numpy.newaxis,:],
                label_pciou]
        bestcat_pciou = score_pciou.argsort(axis=0)[::-1]
        # Assign category for each label
        # cat_per_label = primary_categories_per_index(self.ds)
        # cat_iou = [categories[cat_per_label[i]] for i in label_iou]
        # Now sort units by score and visulize each one
        return bestcat_pciou, name_pciou, score_pciou, label_pciou, tc, tg, ti

    def generate_html_summary(self, layer,
            imsize=64, imcount=16, imscale=None, tally_stats=None,
            gridwidth=None, verbose=False):
        print 'Generating html summary', (
            self.ed.filename(['html', '%s.html' % expdir.fn_safe(layer)]))
        # Grab tally stats
        bestcat_pciou, name_pciou, score_pciou, _, _, _, _ = (tally_stats)
        if verbose:
            print 'Sorting units by score.'
            sys.stdout.flush()
        if imscale is None:
            imscale = imsize
        categories = self.ds.category_names()
        ordering = score_pciou.max(axis=0).argsort()[::-1]
        top = self.max_act_indexes(layer, count=imcount)
        self.ed.ensure_dir('html', 'image')
        css = ('https://maxcdn.bootstrapcdn.com/bootstrap/latest' +
               '/css/bootstrap.min.css')
        html = ['<!doctype html>', '<html>', '<head>',
                '<link rel="stylesheet" href="%s">' % css,
                '</head>', '<body>', '<div class="container-fluid">']
        if gridwidth is None:
            gridname = ''
            gridwidth = imcount
            gridheight = 1
        else:
            gridname = '-%d' % gridwidth
            gridheight = (imcount + gridwidth - 1) // gridwidth
        for unit in ordering:
            if verbose:
                print 'Visualizing %s unit %d' % (layer, unit)
                sys.stdout.flush()
            tiled = numpy.full(
                ((imsize + 1) * gridheight - 1,
                 (imsize + 1) * gridwidth - 1, 3), 255, dtype='uint8')
            for x, index in enumerate(top[unit]):
                row = x // gridwidth
                col = x % gridwidth
                vis = self.activation_visualization(layer, unit, index)
                if vis.shape[:2] != (imsize, imsize):
                    vis = imresize(vis, (imsize, imsize))
                tiled[row*(imsize+1):row*(imsize+1)+imsize,
                      col*(imsize+1):col*(imsize+1)+imsize,:] = vis
            imfn = 'image/%s%s-%04d.jpg' % (
                    expdir.fn_safe(layer), gridname, unit)
            imsave(self.ed.filename(os.path.join('html', imfn)), tiled)
            labels = '; '.join(['%s (%s, %f)' %
                    (name_pciou[c][unit], categories[c], score_pciou[c, unit])
                    for c in bestcat_pciou[:,unit]])
            html.extend([
                '<h6>%s unit %d: %s</h6>' % (layer, unit + 1, labels),
                '<img src="%s" height="%d">' % (os.path.join('html', imfn), imscale)
                ])
        html.extend([
            '</div>', '</body>', '</html>', ''])
        with open(self.ed.filename([
                'html', '%s.html' % expdir.fn_safe(layer)]), 'w') as f:
            f.write('\n'.join(html))

    def generate_csv_summary(
            self, layer, csvfile, tally_stats, order=None, verbose=False):
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
        categories = self.ds.category_names()
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
        # top = self.max_act_indexes(layer, count=imcount)

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

    def generate_quantmat(self, layer, verbose=False):
        if verbose:
            print 'Generating quantmat'
            sys.stdout.flush()
        from scipy.io import savemat
        lp = self.layer[layer]
        filename = self.ed.filename(
                'quant-%d.mat' % lp.quantdata.shape[1], blob=layer)
        savemat(filename, { 'quantile': lp.quantdata })

    def generate_imgmax(self, layer, verbose=False):
        from scipy.io import savemat
        imgmax = self.ed.open_mmap(blob=layer, part='imgmax', mode='w+',
                shape = self.layer[layer].blobdata.shape[:2])
        imgmax[...] = self.layer[layer].blobdata.max(axis=(2, 3))
        self.ed.finish_mmap(imgmax)
        # Also copy out to mat file
        filename = self.ed.filename('imgmax.mat', blob=layer)
        savemat(filename, { 'imgmax': imgmax })
        # And cache
        self.layer[layer].imgmax = imgmax

    def instance_data(self, i, normalize=True):
        record, shape = self.ds.resolve_segmentation(
                self.ds.metadata(i), categories=None)
        if normalize:
            default_shape = (1, ) + shape
            record = dict((cat, normalize_label(dat, default_shape))
                    for cat, dat in record.items())
        return record, shape

    def top_image_indexes(self, layer, unit, count=10):
        t = self.layer[layer].count_a[:,unit].argsort()[::-1]
        return t[:count]

    # Generates a mask at the "lp.level" quantile.
    def activation_mask(self, layer, unit, index, shape=None):
        if shape is None:
             record, shape = self.instance_data(index)
        sw, sh = shape
        # reduction = int(round(self.iw / float(sw)))
        lp = self.layer[layer]
        blobdata = lp.blobdata
        fieldmap = lp.fieldmap
        quantdata = lp.quantdata
        threshold = quantdata[unit, int(round(quantdata.shape[1] * lp.level))]
        up = upsample.upsampleL(
                fieldmap, blobdata[index:index+1, unit],
                shape=(self.ih, self.iw), scaleshape=(sh, sw))[0]
        mask = up > threshold
        return mask

    # Makes an iamge using the mask
    def activation_visualization(self, layer, unit, index, alpha=0.2):
        image = imread(self.ds.filename(index))
        mask = self.activation_mask(layer, unit, index, shape=image.shape[:2])
        return (mask[:, :, numpy.newaxis] * (1 - alpha) + alpha) * image

    def summarize_tally(self, layer):
        cat_count = len(self.ds.category_names())
        tally = self.layer[layer].tally
        unit_size = self.layer[layer].shape[1]
        label_size = self.ds.label_size()
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

    def count_act_with_label(self, layer):
        # Because our dataaset is sparse, instead of using count_a to count
        # all activations, we can compute count_act_with_label to count
        # activations only within those images which contain an instance
        # of a given label.
        tally = self.layer[layer].tally
        unit_size = self.layer[layer].shape[1]
        label_size = self.ds.label_size()
        count = numpy.zeros((unit_size, label_size), dtype='int64')
        for i in range(len(tally)):
            c1 = numpy.zeros((unit_size + 1, label_size + 1), dtype='int64')
            t = tally[i]
            c1[t[:,0]+1, t[:,1]+1] = t[:,2]
            count += c1[1:,0][:,numpy.newaxis] * (c1[0,1:][numpy.newaxis] > 0)
        return count

    def count_act_with_labelcat(self, layer):
        # Because our dataaset is sparse, instead of using count_a to count
        # all activations, we can compute count_act_with_labelcat to count
        # activations only within those images which contain an instance
        # of a given label category.
        labelcat = onehot(primary_categories_per_index(self.ds))
        # Be sure to zero out the background label - it belongs to no category.
        labelcat[0,:] = 0
        tally = self.layer[layer].tally
        unit_size = self.layer[layer].shape[1]
        label_size = self.ds.label_size()
        count = numpy.zeros((unit_size, labelcat.shape[1]), dtype='int64')
        for i in range(len(tally)):
            c1 = numpy.zeros((unit_size + 1, label_size + 1), dtype='int64')
            t = tally[i]
            c1[t[:,0]+1, t[:,1]+1] = t[:,2]
            count += c1[1:,0][:,numpy.newaxis] * (
                    numpy.dot(c1[0,1:], labelcat) > 0)
        # retval: (unit_size, label_size)
        return numpy.dot(count, numpy.transpose(labelcat))

    def max_act_indexes(self, layer, count=10):
        max_per_image = self.layer[layer].imgmax
        return max_per_image.argsort(axis=0)[:-1-count:-1,:].transpose()

    def top_act_indexes(self, layer, count=10):
        tally = self.layer[layer].tally
        unit_size = self.layer[layer].shape[1]
        label_size = self.ds.label_size()
        all_acts = numpy.zeros((len(tally), unit_size), dtype='int64')
        for i in range(len(tally)):
            acts = numpy.zeros((unit_size + 1, 2), dtype='int32')
            t = tally[i]
            acts[t[:,0] + 1, (t[:,1] != -1).astype('int')] = t[:,2]
            all_acts[i] = acts[1:,0]
        return all_acts.argsort(axis=0)[:-1-count:-1,:].transpose()

class LayerProbe:
    def __init__(self, ed, blob, ds):
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
            description='Generate visualization for probed activation data.')
        parser.add_argument(
                '--directory',
                default='.',
                help='output directory for the net probe')
        parser.add_argument(
                '--format',
                default='html',
                help='html or csv or both')
        parser.add_argument(
                '--csvorder',
                help='csv header order')
        parser.add_argument(
                '--blobs',
                nargs='*',
                help='network blob names to visualize')
        parser.add_argument(
                '--gridwidth',
                type=int, default=None,
                help='width of visualization grid')
        parser.add_argument(
                '--imsize',
                type=int, default=72,
                help='thumbnail dimensions')
        parser.add_argument(
                '--imscale',
                type=int, default=None,
                help='thumbnail dimensions')
        parser.add_argument(
                '--imcount',
                type=int, default=16,
                help='number of thumbnails to include')
        args = parser.parse_args()
        np = NetworkProbe(args.directory, blobs=args.blobs)
        for blob in args.blobs:
            formats = args.format.split(',')
            if 'imgmax' in formats:
                np.generate_imgmax(blob)
            if 'html' in formats or 'csv' in formats:
                tally_stats = np.score_tally_stats(blob, verbose=True)
            if 'html' in formats:
                np.generate_html_summary(blob,
                        imsize=args.imsize, imscale=args.imscale,
                        imcount=args.imcount, tally_stats=tally_stats,
                        gridwidth=args.gridwidth,
                        verbose=True)
            if 'csv' in formats:
                filename = os.path.join(args.directory,
                        '%s-result.csv' % expdir.fn_safe(blob))
                np.generate_csv_summary(blob, filename, tally_stats,
                        order=(args.csvorder.split(',')
                            if args.csvorder else None), verbose=True)
            if 'quantmat' in formats:
                np.generate_quantmat(blob, verbose=True)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
