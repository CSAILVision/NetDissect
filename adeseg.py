import colorname
import glob
import os
import re
import numpy
from loadseg import AbstractSegmentation
from scipy.io import loadmat
from scipy.misc import imread
from collections import namedtuple

class AdeSegmentation(AbstractSegmentation):
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
        self.index = Ade20kIndex(
               **{name: index[name][()] for name in index.dtype.names})
        self.scenes = ['-'] + [
                norm_name(s) for s in sorted(set(self.index.scene))]
        self.scene_map = dict((s, i) for i, s in enumerate(self.scenes))
        # Zero out special ~ scene names, which mean unlabeled.
        for k in self.scene_map:
            if k.startswith('~'):
                self.scene_map[k] = 0
        self.raw_mat = mat

    def all_names(self, category, j):
        if j == 0:
            return []
        if category == 'color':
            return [colorname.color_names[j - 1] + '-c']
        if category == 'scene':
            return [self.scenes[j] + '-s']
        result = self.index.objectnames[j - 1]
        return re.split(',\s*', result)

    def size(self):
        '''Returns the number of images in this dataset.'''
        return len(self.index.filename)

    def filename(self, n):
        '''Returns the filename for the nth dataset image.'''
        filename = self.index.filename[n]
        folder = self.index.folder[n]
        return self.expand(folder, filename)

    def metadata(self, i):
        '''Returns an object that can be used to create all segmentations.'''
        return dict(
                filename=self.filename(i),
                seg_filename=self.seg_filename(i),
                part_filenames=self.part_filenames(i),
                scene=self.scene_map[norm_name(self.index.scene[i])]
                )

    @classmethod
    def resolve_segmentation(cls, m, categories=None):
        result = {}
        if wants('scene', categories):
            result['scene'] = m['scene']
        if wants('part', categories):
            result['part'] = load_parts(m)
        if wants('object', categories):
            result['object'] = load_segmentation(m)
        if wants('color', categories):
            result['color'] = colorname.label_major_colors(load_image(m)) + 1
        arrs = [a for a in result.values() if numpy.shape(a) >= 2]
        shape = arrs[0].shape[-2:] if arrs else (1, 1)
        return result, shape

    ### End of contract for AbstractSegmentation

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

    def expand(self, *path):
        '''Expands a filename and directories with the ADE dataset'''
        result = os.path.join(self.root, *path)
        if '*' in result or '?' in result:
            globbed = glob.glob(result)
            if len(globbed):
                return globbed[0]
        return result

def norm_name(s):
    return s.replace(' - ', '-').replace('/', '-')

def load_image(m):
    '''Returns the nth dataset image as a numpy array.'''
    return imread(m['filename'])

def load_segmentation(m):
    '''Returns the nth dataset segmentation as a numpy array,
    where each entry at a pixel is an object class value.
    '''
    data = imread(m['seg_filename'])
    return decodeClassMask(data)

def load_parts(m):
    '''Returns an list of part segmentations for the nth dataset item,
    with one array for each level available.
    '''
    result = []
    for fn in m['part_filenames']:
        data = imread(fn)
        result.append(decodeClassMask(data))
    if not result:
        return []
    return numpy.concatenate(tuple(m[numpy.newaxis] for m in result))

def decodeClassMask(im):
    '''Decodes pixel-level object/part class and instance data from
    the given image, previously encoded into RGB channels.'''
    # Classes are a combination of RG channels (dividing R by 10)
    return (im[:,:,0] // 10) * 256 + im[:,:,1]

def wants(what, option):
    if option is None:
        return True
    return what in option
