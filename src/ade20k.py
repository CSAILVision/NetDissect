#!/usr/bin/env python

import os, glob
import re
import numpy
from scipy.io import loadmat
from scipy.misc import imread, imsave
from collections import namedtuple
from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom

ADE_ROOT = '/home/davidbau/bulk/ade20k/'
ADE_VER = 'ADE20K_2016_07_26'

def decodeClassMask(im):
    '''Decodes pixel-level object/part class and instance data from
    the given image, previously encoded into RGB channels.'''
    # Classes are a combination of RG channels (dividing R by 10)
    return (im[:,:,0] // 10) * 256 + im[:,:,1]

def decodeInstanceMask(im):
    # Instance number is scattered, so we renumber them
    (orig, instances) = numpy.unique(im[:,:,2], return_inverse=True)
    return instances.reshape(classes.shape)

def encodeClassMask(im, offset=0):
    result = numpy.zeros(im.shape + (3,), dtype=numpy.uint8)
    if offset:
        support = im > offset
        mapped = (im + support) * offset
    else:
        mapped = im
    result[:,:,1] = mapped % 256
    result[:,:,0] = (mapped // 256) * 10
    return result

class Dataset:
    def __init__(self, directory=None, version=None):
        # Default to value of ADE20_ROOT env variable
        if directory is None:
            directory = os.environ['ADE20K_ROOT']
        directory = os.path.expanduser(directory)
        # Default to the latest version present in the directory
        if version is None:
            contents = os.listdir(directory)
            if not list(c for c in contents if re.match('^index.*mat$', c)):
                version = sorted(c for c in contents if os.path.isdir(
                    os.path.join(directory, c)))[-1]
            else:
                version = ''
        self.root = directory
        self.version = version

        mat = loadmat(self.expand(self.version, 'index*.mat'), squeeze_me=True)
        index = mat['index']
        Ade20kIndex = namedtuple('Ade20kIndex', index.dtype.names)
        # for name in index.dtype.names:
        #     setattr(self, name, index[name][()])
        self.index = Ade20kIndex(
               **{name: index[name][()] for name in index.dtype.names})
        self.raw_mat = mat

    def expand(self, *path):
        '''Expands a filename and directories with the ADE dataset'''
        result = os.path.join(self.root, *path)
        if '*' in result or '?' in result:
            globbed = glob.glob(result)
            if len(globbed):
                return globbed[0]
        return result

    def filename(self, n):
        '''Returns the filename for the nth dataset image.'''
        filename = self.index.filename[n]
        folder = self.index.folder[n]
        return self.expand(folder, filename)

    def short_filename(self, n):
        '''Returns the filename for the nth dataset image, without folder.'''
        return self.index.filename[n]

    def size(self):
        '''Returns the number of images in this dataset.'''
        return len(self.index.filename)

    def num_object_types(self):
        return len(self.index.objectnames)

    def seg_filename(self, n):
        '''Returns the segmentation filename for the nth dataset image.'''
        return re.sub(r'\.jpg$', '_seg.png', self.filename(n))

    def part_filenames(self, n):
        '''Returns all the subpart images for the nth dataset image.'''
        filename = self.filename(n)
        level = 1
        result = []
        while True:
            probe = re.sub(r'\.jpg$', '_parts_%d.png' % level, filename)
            if not os.path.isfile(probe):
                break
            result.append(probe)
            level += 1
        return result

    def part_levels(self):
        return max([len(self.part_filenames(n)) for n in range(self.size())])

    def image(self, n):
        '''Returns the nth dataset image as a numpy array.'''
        return imread(self.filename(n))

    def segmentation(self, n, include_instances=False):
        '''Returns the nth dataset segmentation as a numpy array,
        where each entry at a pixel is an object class value.

        If include_instances is set, returns a pair where the second
        array labels each instance with a unique number.'''
        data = imread(self.seg_filename(n))
        if include_instances:
            return (decodeClassMask(data), decodeInstanceMask(data))
        else:
            return decodeClassMask(data)

    def parts(self, n, include_instances=False):
        '''Returns an list of part segmentations for the nth dataset item,
        with one array for each level available.  If included_instances is
        set, the list contains pairs of numpy arrays (c, i) where i
        represents instances.'''
        result = []
        for fn in self.part_filenames(n):
            data = imread(fn)
            if include_instances:
                result.append((decodeClassMask(data), decodeInstanceMask(data)))
            else:
                result.append(decodeClassMask(data))
        return result

    def full_segmentation(self, n, include_instances=False):
        '''Returns a single tensor with all levels of segmentations included
        in the channels, one channel per level.  If include_instances is
        requested, a parallel tensor with instance labels is returned in
        a tuple.'''
        full = [self.segmentation(n, include_instances)
                ] + self.parts(n, include_instances)
        if include_instances:
            return tuple(numpy.concatenate(tuple(m[numpy.newaxis] for m in d)
                    for d in zip(full)))
        return numpy.concatenate(tuple(m[numpy.newaxis] for m in full))

    def object_name(self, c):
        '''Returns a short English name for the object class c.'''
        # Off by one due to use of 1-based indexing in matlab.
        if c == 0:
            return '-'
        result = self.index.objectnames[c - 1]
        return re.split(',\s*', result, 1)[0]

    def object_count(self, c):
        '''Returns a count of the object over the whole dataset.'''
        # Off by one due to use of 1-based indexing in matlab.
        return self.index.objectcounts[c - 1]

    def object_presence(self, c):
        '''Returns a per-dataset-item count of the object.'''
        # Off by one due to use of 1-based indexing in matlab.
        return self.index.objectPresence[c - 1]

    def scale_image(self, im, dims, crop=False):
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

    def scale_segmentation(self, segmentation, dims, crop=False):
        if segmentation.shape[1:] == dims:
            return segmentation
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
        return result

    def save_image(self, im, filename, folder):
        imsave(os.path.join(folder, filename), im)

    def save_segmentation(self, seg, filename, folder, offset=0):
        for channel in range(seg.shape[0]):
            im = encodeClassMask(seg[channel], offset=offset)
            if channel == 0:
                fn = re.sub('\.jpg$', '_seg.png', filename)
            else:
                fn = re.sub('\.jpg$', '_parts_%s.png' % channel, filename)
            imsave(os.path.join(folder, fn), im)

    def save_sample(self, folder, size=None, indexes=None, crop=False,
            offset=0, reduction=1, progress=False):
        if indexes is None:
            indexes = range(self.size())
        count = len(indexes)
        test_dim = None
        if size is not None:
            test_dim = tuple(int(d / reduction) for d in size)
        for i, index in enumerate(indexes):
            filename = self.short_filename(index)
            print 'Proessing %s (%d of %d)' % (filename, i, count)
            im = self.image(index)
            if size is not None:
                im = self.scale_image(im, size, crop=crop)
            self.save_image(im, filename, folder)
            seg = self.full_segmentation(index)
            if test_dim is not None:
                seg = self.scale_segmentation(seg, test_dim, crop=crop)
            self.save_segmentation(seg, filename, folder, offset=offset)
        print 'Processed %d images' % count

    def save_object_names(self, folder, offset=0):
        with file(os.path.join(folder, 'object_names.txt'), 'w') as f:
            for index in range(offset, self.num_object_types()):
                f.write('%s\t%d\n' % (self.object_name(index), index - offset))

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


