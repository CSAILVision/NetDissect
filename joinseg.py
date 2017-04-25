'''
This package provides a function and a base class:
    unify and AbstractSegmentation.

unify(data_sets, directory, size=None, segmentation_size=None, crop=False,
        min_frequency=None, min_coverage=None, synonyms=None,
        test_limit=None, single_process=False, verbose=False):

unify creates a stadnard format multichannel segementation
out of heterogenous labeled image sources.

Input:

    data_sets is a list of instances of subclasses of
    AbstractSegmentation (or similar objects).  Five methods
    must be provided on each object.

    names(cat, j) - a list of names that describe label j
        as it is used within category cat in this dataset.

    size() - number of images in the dataset.

    filename(i) - the jpg filename for the i'th image in the dataset.

    metadata(i) - a small pickleable record from which the
         segmentation of the ith item can be derived, withou reference
         to a specific corpus instance.

    @classmethod
    resolve_segmentation(m) - resolves to a dict and a shape of a segmentation.
         The dict contains string key names and numpy-array vals describing
         the segmentation of the image drived from metadata m.  The keys
         of this dict are arbitrary short strings, but for example could
         look like:
       {
         'scene': 1293, # 0-d: applies one label to the whole image
         'object': (1024x1280 int16), # 2-d: one label per pixel
         'part': (3x1024x1280 int16), # 3-d: layers of labels, dominant first
         'texture': [14,22] # 1-d: multiple whole-image labels, dominant first
       }

    The reason for the separation between metadata(i) and
    resolve_segmentation(m) so that we can run metadata(i) quickly on the
    main process, and then shard resolve_segmentation(m) out to worker
    processes.

Output is the following directory structure:
'''
README_TEXT= '''
This directory contains the following data and metadata files:

    images/[datasource]/...
         images drawn from a specific datasource are reformatted
         and scaled and saved as jpeg and png files in subdirectories
         of the images directory.

    index.csv
        contains a list of all the images in the dataset, together with
        available labeled data, in the form:

        image,split,ih,iw,sh,sw,[color,object,material,part,scene,texture]

        for examplle:
        dtd/g_0106.jpg,train,346,368,346,368,dtd/g_0106_color.png,,,,,314;194

        The first column is always the original image filename relative to
        the images/ subdirectory; then image height and width and segmentation
        heigh and width dimensions are followed by six columns for label
        data in the six categories.  Label data can be per-pixel or per-image
        (assumed to apply to every pixel in the image), and 0 or more labels
        can be specified per category.  If there are no labels, the field is ''.
        If there are multiple labels, they are separated by semicolons,
        and it is assumed that dominant interpretations are listed first.
        Per-image labels are represented by a decimal number; and per-image
        labels are represented by a filename for an image which encodes
        the per-pixel labels in the (red + 256 * green) channels.

    category.csv
        name,first,last,count,frequency

        for example:
        object,12,1138,529,208688

        In the generic case there may not be six categories; this directory
        may contain any set of categories of segmentations.  This file
        lists all segmentation categories found in the data sources,
        along with statistics on now many class labels apply to
        each category, and in what range; as well as how many
        images mention a label of that category

    label.csv
        number,name,category,frequency,coverage,syns

        for example:
        10,red-c,color(289),289,9.140027,
        21,fabric,material(36);object(3),39,4.225474,cloth

        This lists all label numbers and corresponding names, along
        with some statistics including the number of images for
        which the label appears in each category; the total number
        of images which have the label; and the pixel portions
        of images that have the label.

    c_[category].csv (for example, map_color.csv)
        code,number,name,frequency,coverage

        for example:
        4,31,glass,27,0.910724454131

        Although labels are store under a unified code, this file
        lists a standard dense coding that can be used for a
        specific subcategory of labels.
'''

import codecs
from functools import partial
import itertools
from multiprocessing import Pool, cpu_count
import numpy
import operator
import os
import re
import subprocess
import time
from PIL import Image
from collections import OrderedDict
from scipy.misc import imread, imresize, imsave
from scipy.ndimage.interpolation import zoom
import signal
import shutil
import sys
from unicsv import DictUnicodeWriter


def unify(data_sets, directory, size=None, segmentation_size=None, crop=False,
        splits=None, min_frequency=None, min_coverage=None,
        synonyms=None, test_limit=None, single_process=False, verbose=False):
    # Make sure we have a directory to work in
    directory = os.path.expanduser(directory)
    ensure_dir(directory)

    # Step 0: write a README file with generated information.
    write_readme_file([
        ('data_sets', data_sets), ('size', size),
        ('segmentation_size', segmentation_size), ('crop', crop),
        ('splits', splits),
        ('min_frequency', min_frequency), ('min_coverage', min_coverage),
        ('synonyms', synonyms), ('test_limit', test_limit),
        ('single_process', single_process)],
        directory=directory, verbose=verbose)

    # Clear old images data
    try:
        if verbose:
            print 'Removing old images.'
        shutil.rmtree(os.path.join(directory, 'images'), ignore_errors=True)
    except:
        pass
    ensure_dir(os.path.join(directory, 'images'))
    # Phase 1: Count label statistics
    # frequency = number of images touched by each label
    # coverage  = total portion of images covered by each label
    frequency, coverage = gather_label_statistics(
            data_sets, test_limit, single_process,
            segmentation_size, crop, verbose)
    # Phase 2: Sort, collapse, and filter labels
    labnames, syns = normalize_labels(data_sets, frequency, coverage, synonyms)
    report_aliases(directory, labnames, data_sets, frequency, verbose)
    # Phase 3: Filter by frequncy, and assign numbers
    names, assignments = assign_labels(
            labnames, frequency, coverage, min_frequency, min_coverage, verbose)
    # Phase 4: Collate and output label stats
    cats = write_label_files(
            directory, names, assignments, frequency, coverage, syns, verbose)
    # Phase 5: Create normalized segmentation files
    create_segmentations(
            directory, data_sets, splits, assignments, size, segmentation_size,
            crop, cats, test_limit, single_process, verbose)

def gather_label_statistics(data_sets, test_limit, single_process,
        segmentation_size, crop, verbose):
    '''
    Phase 1 of unification.  Counts label statistics.
    '''
    # Count frequency and coverage for each individual image
    stats = map_in_pool(partial(count_label_statistics,
        segmentation_size=segmentation_size,
        crop=crop,
        verbose=verbose),
            all_dataset_segmentations(data_sets, test_limit),
            single_process=single_process,
            verbose=verbose)
    # Add them up
    frequency, coverage = (sum_histogram(d) for d in zip(*stats))
    # TODO: also, blacklist images that have insufficient labled pixels
    return frequency, coverage

def normalize_labels(data_sets, frequency, coverage, synonyms):
    '''
    Phase 2 of unification.
    Assigns unique label names and resolves duplicates by name.
    '''
    # Sort by frequency, descending, and assign a name to each label
    top_syns = {}
    labname = {}
    freq_items = zip(*sorted((-f, lab) for lab, f in frequency.items()))[1]
    for lab in freq_items:
        dataset, category, label = lab
        names = [n.lower() for n in data_sets[dataset].all_names(
              category, label) if len(n) and n != '-']
        if synonyms:
            names = synonyms(names)
        # Claim the synonyms that have not already been taken
        for name in names:
            if name not in top_syns:
                top_syns[name] = lab
        # best_labname may decide to collapse two labels because they
        # have the same names and seem to mean the same thing
        labname[lab], unique = best_labname(
                lab, names, labname, top_syns, coverage, frequency)
    return labname, top_syns

def report_aliases(directory, labnames, data_sets, frequency, verbose):
    '''
    Phase 2.5
    Report the aliases.  These are printed into 'syns.txt'
    '''
    show_all = True  # Print all details to help debugging
    name_to_lab = invert_dict(labnames)
    with codecs.open(os.path.join(directory, 'syns.txt'), 'w', 'utf-8') as f:
        def report(txt):
            f.write('%s\n' % txt)
            if verbose:
                print txt
        for name in sorted(name_to_lab.keys(),
                key=lambda n: (-len(name_to_lab[n]), n)):
            keys = name_to_lab[name]
            if not show_all and len(keys) <= 1:
                break
            # Don't bother reporting aliases if all use the same index;
            # that is probably due to an underlying shared code.
            if not show_all and len(set(i for d, c, i in keys)) <= 1:
                continue
            report('name: %s' % name)
            for ds, cat, i in keys:
                names = ';'.join(data_sets[ds].all_names(cat, i))
                freq = frequency[(ds, cat ,i)]
                report('%s/%s#%d: %d, (%s)' % (ds, cat, i, freq, names))
            report('')

def assign_labels(
        labnames, frequency, coverage, min_frequency, min_coverage, verbose):
    '''
    Phase 3 of unification.
    Filter names that are too infrequent, then assign numbers.
    '''
    # Collect by-name frequency and coverage
    name_frequency = join_histogram(frequency, labnames)
    name_coverage = join_histogram(coverage, labnames)
    names = name_frequency.keys()
    if min_frequency is not None:
        names = [n for n in names if name_frequency[n] >= min_frequency]
    if min_coverage is not None:
        names = [n for n in names if name_coverage[n] >= min_coverage]
    # Put '-' at zero
    names = [n for n in names if n != '-']
    names = ['-'] + sorted(names,
            key=lambda x: (-name_frequency[x], -name_coverage[x]))
    nums = dict((n, i) for i, n in enumerate(names))
    assignments = dict((k, nums.get(v, 0)) for k, v in labnames.items())
    return names, assignments

def write_label_files(
        directory, names, assignments, frequency, coverage, syns, verbose):
    '''
    Phase 4 of unification.
    Collate some stats and then write then to two metadata files.
    '''
    # Make lists of synonyms claimed by each label
    synmap = invert_dict(dict((w, assignments[lab]) for w, lab in syns.items()))
    # We need an (index, category) count
    ic_freq = join_histogram_fn(frequency, lambda x: (assignments[x], x[1]))
    ic_cov = join_histogram_fn(coverage, lambda x: (assignments[x], x[1]))
    for z in [(j, cat) for j, cat in ic_freq if j == 0]:
        del ic_freq[z]
        del ic_cov[z]
    catstats = [[] for n in names]
    # For each index, get a (category, frequency) list in descending order
    for (ind, cat), f in sorted(ic_freq.items(), key=lambda x: -x[1]):
        catstats[ind].append((cat, f))
    index_coverage = join_histogram(coverage, assignments)
    with open(os.path.join(directory, 'label.csv'), 'w') as csvfile:
        fields = ['number', 'name', 'category', 'frequency', 'coverage', 'syns']
        writer = DictUnicodeWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for ind, name in enumerate(names):
            if ind == 0:
                continue
            writer.writerow(dict(
                number='%d' % ind,
                name=name,
                category=';'.join('%s(%d)' % s for s in catstats[ind]),
                frequency='%d' % sum(f for c, f in catstats[ind]),
                coverage='%f' % index_coverage[ind],
                syns=';'.join([s for s in synmap[ind] if s != name])
            ))
    # For each category, figure the first, last, and other stats
    cat_ind = [(cat, ind) for ind, cat in ic_freq.keys()]
    first_index = build_histogram(cat_ind, min)
    last_index = build_histogram(cat_ind, max)
    count_labels = build_histogram([(cat, 1) for cat, _ in cat_ind])
    cat_freq = join_histogram_fn(ic_freq, lambda x: x[1])
    cats = sorted(first_index.keys(), key=lambda x: first_index[x])
    with open(os.path.join(directory, 'category.csv'), 'w') as csvfile:
        fields = ['name', 'first', 'last', 'count', 'frequency']
        writer = DictUnicodeWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for cat in cats:
            writer.writerow(dict(
                name=cat,
                first=first_index[cat],
                last=last_index[cat],
                count=count_labels[cat],
                frequency=cat_freq[cat]))
    # And for each category, create a dense coding file.
    for cat in cats:
        dense_code = [0] + sorted([i for i, c in ic_freq if c == cat],
                key=lambda i: (-ic_freq[(i, cat)], -ic_cov[(i, cat)]))
        fields = ['code', 'number', 'name', 'frequency', 'coverage']
        with open(os.path.join(directory, 'c_%s.csv' % cat), 'w') as csvfile:
            writer = DictUnicodeWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            for code, i in enumerate(dense_code):
                if code == 0:
                    continue
                writer.writerow(dict(
                    code=code,
                    number=i,
                    name=names[i],
                    frequency=ic_freq[(i, cat)],
                    coverage=ic_cov[(i, cat)]))
    return cats

def create_segmentations(directory, data_sets, splits, assignments, size,
        segmentation_size, crop, cats, test_limit, single_process, verbose):
    '''
    Phase 5 of unification.  Create the normalized segmentation files
    '''
    if size is not None and segmentation_size is None:
        segmentation_size = size
    # Get assignments into a nice form, once, here.
    # (dataset, category): [numpy array with new indexes]
    index_max = build_histogram(
            [((ds, cat), i) for ds, cat, i in assignments.keys()], max)
    index_mapping = dict([k, numpy.zeros(i + 1, dtype=numpy.int16)]
            for k, i in index_max.items())
    for (ds, cat, oldindex), newindex in assignments.items():
        index_mapping[(ds, cat)][oldindex] = newindex
    # Count frequency and coverage for each individual image
    segmented = map_in_pool(
            partial(translate_segmentation,
                directory=directory,
                mapping=index_mapping,
                size=size,
                segmentation_size=segmentation_size,
                categories=cats,
                crop=crop,
                verbose=verbose),
            all_dataset_segmentations(data_sets, test_limit),
            single_process=single_process,
            verbose=verbose)
    # Sort nonempty itesm randomly+reproducibly by md5 hash of the filename.
    ordered = sorted([(hashed_float(r['image']), r) for r in segmented if r])
    # Assign splits, pullout out last 20% for validation.
    cutoffs = cumulative_splits(splits)
    for floathash, record in ordered:
        for name, cutoff in cutoffs:
            if floathash <= cutoff:
                record['split'] = name
                break
        else:
            assert False, 'hash %f exceeds last split %f' % (floathash, c)

    # Now write one row per image and one column per category
    with open(os.path.join(directory, 'index.csv'), 'w') as csvfile:
        fields = ['image', 'split', 'ih', 'iw', 'sh', 'sw'] + cats
        writer = DictUnicodeWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for f, record in ordered:
            writer.writerow(record)

def translate_segmentation(record, directory, mapping, size,
        segmentation_size, categories, crop, verbose):
    '''
    Translates a single segmentation.
    '''
    dataset, index, seg_class, filename, md = record
    basename = os.path.basename(filename)
    if verbose:
        print 'Processing #%d %s %s' % (index, dataset, basename)
    full_seg, shape = seg_class.resolve_segmentation(md)
    # Rows can be omitted by returning no segmentations.
    if not full_seg:
        return None
    jpg = imread(filename)
    if size is not None:
        jpg = scale_image(jpg, size, crop)
        for cat in full_seg:
            full_seg[cat] = scale_segmentation(
                    full_seg[cat], segmentation_size, crop)
    else:
        size = jpg.shape[:2]
        segmentation_size = (1,1)
        for cat in full_seg:
            if len(numpy.shape(full_seg[cat])) >= 2:
                segmentation_size = numpy.shape(full_seg[cat])
                break

    imagedir = os.path.join(directory, 'images')
    ensure_dir(os.path.join(imagedir, dataset))
    fn = save_image(jpg, imagedir, dataset, basename)
    result = {
            'image': os.path.join(dataset, fn),
            'ih': size[0],
            'iw': size[1],
            'sh': segmentation_size[0],
            'sw': segmentation_size[1]
    }
    for cat in full_seg:
        if cat not in categories:
            continue  # skip categories with no data globally
        result[cat] = ';'.join(save_segmentation(full_seg[cat],
                imagedir, dataset, fn, cat, mapping[(dataset, cat)]))
    return result

def best_labname(lab, names, assignments, top_syns, coverage, frequency):
    '''
    Among the given names, chooses the best name to assign, given
    information about previous assignments, synonyms, and stats
    '''
    # Best shot: get my own name, different from other names.
    if 'dog' in names:
        print names
    for name in names:
        if top_syns[name] == lab:
            return name, True
    if len(names) == 0 or len(names) == '1' and names[0] in ['', '-']:
        # If there are no names, we must use '-' and map to 0.
        return '-', False
    elif len(names) == 1:
        # If we have a conflict without ambiguity, then we just merge names.
        other = top_syns[names[0]]
    else:
        # If we need to merge and we have multiple synonyms, let's score them.
        scores = []
        # Get the average size of an object of my type
        size = coverage[lab] / frequency[lab]
        for name in names:
            other = top_syns[name]
            # Compare it to the average size of objects of other types
            other_size = coverage[other] / frequency[other]
            scores.append((abs(size - other_size), -frequency[other], other))
        # Choose the synonyms that has the most similar average size.
        # (breaking ties according to maximum frequency)
        other = min(scores)[2]
    name = assignments[other]
    return name, False

def build_histogram(pairs, reducer=operator.add):
    '''Creates a histogram by combining a list of key-value pairs.'''
    result = {}
    for k, v in pairs:
        if k not in result:
            result[k] = v
        else:
            result[k] = reducer(result[k], v)
    return result

def join_histogram_fn(histogram, makekey):
    '''Rekeys histogram according to makekey fn, summing joined buckets.'''
    result = {}
    for oldkey, val in histogram.iteritems():
        newkey = makekey(oldkey)
        if newkey not in result:
            result[newkey] = val
        else:
            result[newkey] += val
    return result

def join_histogram(histogram, newkeys):
    '''Rekeys histogram according to newkeys map, summing joined buckets.'''
    result = {}
    for oldkey, newkey in newkeys.iteritems():
        if newkey not in result:
            result[newkey] = histogram[oldkey]
        else:
            result[newkey] += histogram[oldkey]
    return result

def sum_histogram(histogram_list):
    '''Adds histogram dictionaries elementwise.'''
    result = {}
    for d in histogram_list:
        for k, v in d.iteritems():
            if k not in result:
                result[k] = v
            else:
                result[k] += v
    return result

def invert_dict(d):
    '''Transforms {k: v} to {v: [k,k..]}'''
    result = {}
    for k, v in d.iteritems():
        if v not in result:
            result[v] = [k]
        else:
            result[v].append(k)
    return result

def count_label_statistics(record, segmentation_size, crop, verbose):
    '''
    Resolves the segmentation record, and then counts all nonzero
    labeled pixels in the resuting segmentation.  Returns two maps:
      freq[(dataset, category, label)] = 1, if the label is present
      coverage[(dataset, category, label)] = p, for portion of pixels covered
    '''
    dataset, index, seg_class, fn, md = record
    if verbose:
        print 'Counting #%d %s %s' % (index, dataset, os.path.basename(fn))
    full_seg, shape = seg_class.resolve_segmentation(md)
    freq = {}
    coverage = {}
    for category, seg in full_seg.items():
        if seg is None:
            continue
        dims = len(numpy.shape(seg))
        if dims <= 1:
            for label in (seg if dims else (seg,)):
                key = (dataset, category, int(label))
                freq[key] = 1
                coverage[key] = 1.0
        elif dims >= 2:
            # We do _not_ scale the segmentation for counting purposes!
            # Different scales should produce the same labels and label order.
            # seg = scale_segmentation(seg, segmentation_size, crop=crop)
            bc = numpy.bincount(seg.ravel())
            pixels = numpy.prod(seg.shape[-2:])
            for label in bc.nonzero()[0]:
                if label > 0:
                    key = (dataset, category, int(label))
                    freq[key] = 1
                    coverage[key] = float(bc[label]) / pixels
    return freq, coverage

def all_dataset_segmentations(data_sets, test_limit=None):
    '''
    Returns an iterator for metadata over all segmentations
    for all images in all the datasets.  The iterator iterates over
    (dataset_name, global_index, dataset_resolver, metadata(i))
    '''
    j = 0
    for name, ds in data_sets.items():
        for i in truncate_range(range(ds.size()), test_limit):
            yield (name, j, ds.__class__, ds.filename(i), ds.metadata(i))
            j += 1

def truncate_range(data, limit):
    '''For testing, if limit is not None, limits data by slicing it.'''
    if limit is None:
        return data
    if isinstance(limit, slice):
        return data[limit]
    return data[:limit]

def map_in_pool(fn, data, single_process=False, verbose=False):
    '''
    Our multiprocessing solution; wrapped to stop on ctrl-C well.
    '''
    if single_process:
        return map(fn, data)
    n_procs = min(cpu_count(), 12)
    original_sigint_handler = setup_sigint()
    pool = Pool(processes=n_procs, initializer=setup_sigint)
    restore_sigint(original_sigint_handler)
    try:
        if verbose:
            print 'Mapping with %d processes' % n_procs
        res = pool.map_async(fn, data)
        return res.get(31536000)
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
        raise
    else:
        pool.close()
        pool.join()

def setup_sigint():
    return signal.signal(signal.SIGINT, signal.SIG_IGN)

def restore_sigint(original):
    signal.signal(signal.SIGINT, original)


def scale_image(im, dims, crop=False):
    '''
    Scales or crops a photographic image using antialiasing.
    '''
    if len(im.shape) == 2:
        # Handle grayscale images by adding an RGB channel
        im = numpy.repeat(im[numpy.newaxis], 3, axis=0)
    if im.shape[0:2] != dims:
        if not crop:
            im = imresize(im, dims)
        else:
            source = im.shape[0:2]
            aspect = float(dims[1]) / dims[0]
            if aspect * source[0] > source[1]:
                width = int(dims[1] / aspect)
                margin = (width - dims[0]) // 2
                im = imresize(im, (width, dims[1]))[
                        margin:margin+dims[0],:,:]
            else:
                height = int(dims[0] * aspect)
                margin = (height - dims[1]) // 2
                im = imresize(im, (dims[0], height))[
                        margin:margin+dims[1],:,:]
    return im

def scale_segmentation(segmentation, dims, crop=False):
    '''
    Zooms a 2d or 3d segmentation to the given dims, using nearest neighbor.
    '''
    shape = numpy.shape(segmentation)
    if len(shape) < 2 or shape[-2:] == dims:
        return segmentation
    peel = (len(shape) == 2)
    if peel:
        segmentation = segmentation[numpy.newaxis]
    levels = segmentation.shape[0]
    result = numpy.zeros((levels, ) + dims,
            dtype=segmentation.dtype)
    ratio = (1,) + tuple(res / float(orig)
            for res, orig in zip(result.shape[1:], segmentation.shape[1:]))
    if not crop:
        safezoom(segmentation, ratio, output=result, order=0)
    else:
        ratio = max(ratio[1:])
        height = int(round(dims[0] / ratio))
        hmargin = (segmentation.shape[0] - height) // 2
        width = int(round(dims[1] / ratio))
        wmargin = (segmentation.shape[1] - height) // 2
        safezoom(segmentation[:, hmargin:hmargin+height,
            wmargin:wmargin+width],
            (1, ratio, ratio), output=result, order=0)
    if peel:
        result = result[0]
    return result

def safezoom(array, ratio, output=None, order=0):
    '''Like numpy.zoom, but does not crash when the first dimension
    of the array is of size 1, as happens often with segmentations'''
    dtype = array.dtype
    if array.dtype == numpy.float16:
        array = array.astype(numpy.float32)
    if array.shape[0] == 1:
        if output is not None:
            output = output[0,...]
        result = zoom(array[0,...], ratio[1:],
                output=output, order=order)
        if output is None:
            output = result[numpy.newaxis]
    else:
        result = zoom(array, ratio, output=output, order=order)
        if output is None:
            output = result
    return output.astype(dtype)

def save_image(im, imagedir, dataset, filename):
    '''
    Try to pick a unique name and save the image with that name.
    This is not race-safe, so the given name should already be unique.
    '''
    trynum = 1
    fn = filename
    while os.path.exists(os.path.join(imagedir, dataset, fn)):
        trynum += 1
        fn = re.sub('(?:\.jpg)?$', '%d.jpg' % trynum, filename)
    imsave(os.path.join(imagedir, dataset, fn), im)
    return fn

def save_segmentation(seg, imagedir, dataset, filename, category, translation):
    '''
    Saves the segmentation in a file or files if it is large, recording
    the filenames.  Or serializes it as decimal numbers to record if it
    is small.  Then returns the array of strings recorded.
    '''
    if seg is None:
        return None
    shape = numpy.shape(seg)
    if len(shape) < 2:
        # Save numbers as decimal strings; and omit zero labels.
        return ['%d' % translation[t]
                for t in (seg if len(shape) else [seg]) if t]

    result = []
    for channel in ([()] if len(shape) == 2 else range(shape[0])):
        # Save bitwise channels as filenames of PNGs; and save the files.
        im = encodeRG(translation[seg[channel]])
        if len(shape) == 2:
            fn = re.sub('(?:\.jpg)?$', '_%s.png' % category, filename)
        else:
            fn = re.sub('(?:\.jpg)?$', '_%s_%d.png' %
                    (category, channel + 1), filename)
        result.append(os.path.join(dataset, fn))
        imsave(os.path.join(imagedir, dataset, fn), im)
    return result

def encodeRG(channel):
    '''Encodes a 16-bit per-pixel code using the red and green channels.'''
    result = numpy.zeros(channel.shape + (3,), dtype=numpy.uint8)
    result[:,:,0] = channel % 256
    result[:,:,1] = (channel // 256)
    return result

def ensure_dir(targetdir):
    if not os.path.isdir(targetdir):
        try:
            os.makedirs(targetdir)
        except:
            pass

def hashed_float(s):
    # Inspired by http://code.activestate.com/recipes/391413/ by Ori Peleg
    '''Hashes a string to a float in the range [0, 1).'''
    import md5, struct
    [number] = struct.unpack(">Q", md5.new(s).digest()[:8])
    return number / (2.0 ** 64)  # python will constant-fold this denominator.

def cumulative_splits(splits):
    '''Converts [0.8, 0.1, 0.1] to [0.8, 0.9, 1.0]'''
    if splits is None:
        return [('train', 1.0)]  # Default to just one split.
    result = []
    c = 0.0
    for name, s in splits.items():
        c += s
        result.append((name, c))
    # Eliminate any fp rounding problem: round last split up to 1.0
    if result[-1][1] < 1.0 - len(result) * sys.float_info.epsilon:
        raise ValueError('splits must add to 1.0, but %s add to %s' % (
            repr(splits), result[-1][1]))
    result[-1] = (result[-1][0], 1.0)
    return result

def write_readme_file(args, directory, verbose):
    '''
    Writes a README.txt that describes the settings used to geenrate the ds.
    '''
    with codecs.open(os.path.join(directory, 'README.txt'), 'w', 'utf-8') as f:
        def report(txt):
            f.write('%s\n' % txt)
            if verbose:
                print txt
        title = '%s joined segmentation data set' % os.path.basename(directory)
        report('%s\n%s' % (title, '=' * len(title)))
        for key, val in args:
            if key == 'data_sets':
                report('Joins the following data sets:')
                for name, kind in val.items():
                    report('    %s: %d images' % (name, kind.size()))
            else:
                report('%s: %r' % (key, val))
        report('\ngenerated at: %s' % time.strftime("%Y-%m-%d %H:%M"))
        try:
            label = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
            report('git label: %s' % label)
        except:
            pass
        f.write('\n')
        # Add boilerplate readme text
        f.write(README_TEXT)

if __name__ == '__main__':
    import argparse
    import adeseg
    import dtdseg
    import osseg
    import pascalseg
    from synonym import synonyms

    parser = argparse.ArgumentParser(
        description='Generate broden dataset.')
    parser.add_argument(
            '--size',
            type=int, default=224,
            help='pixel size for input images, e.g., 224 or 227')
    args = parser.parse_args()

    image_size = (args.size, args.size)
    seg_size = (args.size // 2, args.size // 2)

    print 'CREATING NEW SEGMENTATION OF SIZE %d.\n' % args.size
    print 'Loading source segmentations.'
    ade = adeseg.AdeSegmentation('dataset/ade20k', 'ADE20K_2016_07_26')
    dtd = dtdseg.DtdSegmentation('dataset/dtd/dtd-r1.0.1')
    # OpenSurfaces is not balanced in scene and object types.
    oss = osseg.OpenSurfaceSegmentation('dataset/opensurfaces/',
            supply=set(['material', 'color']))
    # Remove distinction between upper-arm, lower-arm, etc.
    pascal = pascalseg.PascalSegmentation('dataset/pascal/',
            collapse_adjectives=set([
                'left', 'right', 'front', 'back', 'upper', 'lower', 'side']))
    data = OrderedDict(ade20k=ade, dtd=dtd, opensurfaces=oss, pascal=pascal)
    unify(data,
            splits=OrderedDict(train=0.7, val=0.3),
            size=image_size, segmentation_size=seg_size,
            directory=('dataset/broden1_%d' % args.size),
            synonyms=synonyms,
            min_frequency=10, min_coverage=0.5, verbose=True)
