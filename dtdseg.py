import os
import numpy
import colorname
from loadseg import AbstractSegmentation
from scipy.misc import imread

class DtdSegmentation(AbstractSegmentation):
    def __init__(self, directory=None):
        directory = os.path.expanduser(directory)
        self.directory = directory
        with open(os.path.join(directory, 'labels',
                'labels_joint_anno.txt')) as f:
            self.dtd_meta = [line.split(None, 1) for line in f.readlines()]
        self.textures = ['-'] + sorted(list(set(sum(
            [c.split() for f, c in self.dtd_meta], []))))
        self.texture_map = dict((t, i) for i, t in enumerate(self.textures))

    def all_names(self, category, j):
        if j == 0:
            return []
        if category == 'color':
            return [colorname.color_names[j - 1] + '-c']
        if category == 'texture':
            return [self.textures[j]]
        return []

    def size(self):
        '''Returns the number of images in this dataset.'''
        return len(self.dtd_meta)

    def filename(self, i):
        '''Returns the filename for the nth dataset image.'''
        return os.path.join(self.directory, 'images', self.dtd_meta[i][0])

    def metadata(self, i):
        '''Returns an object that can be used to create all segmentations.'''
        fn, tnames = self.dtd_meta[i]
        tnumbers = [self.texture_map[n] for n in tnames.split()]
        return os.path.join(self.directory, 'images', fn), tnumbers

    @classmethod
    def resolve_segmentation(cls, m, categories=None):
        filename, tnumbers = m
        result = {}
        if wants('texture', categories):
            result['texture'] = tnumbers
        if wants('color', categories):
            result['color'] = colorname.label_major_colors(imread(filename)) + 1
        arrs = [a for a in result.values() if numpy.shape(a) >= 2]
        shape = arrs[0].shape[-2:] if arrs else (1, 1)
        return result, shape

def wants(what, option):
    if option is None:
        return True
    return what in option
