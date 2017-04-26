import os
import numpy
from unicsv import DictUnicodeReader
import colorname
from loadseg import AbstractSegmentation
from scipy.misc import imread

class OpenSurfaceSegmentation(AbstractSegmentation):
    def __init__(self, directory=None, supply=None):
        directory = os.path.expanduser(directory)
        self.directory = directory
        self.supply = supply
        # Process open surfaces labels: open label-name-colors.csv
        object_name_map = {}
        with open(os.path.join(directory, 'label-name-colors.csv')) as f:
            for row in DictUnicodeReader(f):
                object_name_map[row['name_name']] = int(
                        row['green_color'])
        self.object_names = ['-'] * (1 + max(object_name_map.values()))
        for k, v in object_name_map.items():
            self.object_names[v] = k
        # Process material labels: open label-substance-colors.csv
        subst_name_map = {}
        with open(os.path.join(directory, 'label-substance-colors.csv')) as f:
            for row in DictUnicodeReader(f):
                subst_name_map[row['substance_name']] = int(
                        row['red_color'])
        self.substance_names = ['-'] * (1 + max(subst_name_map.values()))
        for k, v in subst_name_map.items():
            self.substance_names[v] = k
        # Now load the metadata about images from photos.csv
        with open(os.path.join(directory, 'photos.csv')) as f:
            self.image_meta = list(DictUnicodeReader(f))
            scenes = set(row['scene_category_name'] for row in self.image_meta)
        self.scenes = ['-'] + sorted(list(scenes))
        self.scene_map = dict((s, i) for i, s in enumerate(self.scenes))
        # Zero out special ~ scene names, which mean unlabeled.
        for k in self.scene_map:
            if k.startswith('~'):
                self.scene_map[k] = 0

    def all_names(self, category, j):
        if j == 0:
            return []
        if category == 'color':
            return [colorname.color_names[j - 1] + '-c']
        if category == 'scene':
            return [self.scenes[j].lower() + '-s']
        if category == 'material':
            return [norm_name(n) for n in self.substance_names[j].split('/')]
        if category == 'object':
            return [norm_name(n) for n in self.object_names[j].split('/')]
        return []

    def size(self):
        '''Returns the number of images in this dataset.'''
        return len(self.image_meta)

    def filename(self, i):
        '''Returns the filename for the nth dataset image.'''
        photo_id = int(self.image_meta[i]['photo_id'])
        return os.path.join(self.directory, 'photos', '%d.jpg' % photo_id)

    def metadata(self, i):
        '''Returns an object that can be used to create all segmentations.'''
        row = self.image_meta[i]
        return (self.directory,
                row,
                self.scene_map[row['scene_category_name']],
                self.supply)

    @classmethod
    def resolve_segmentation(cls, m, categories=None):
        directory, row, scene, supply = m
        photo_id = int(row['photo_id'])
        fnpng = os.path.join(directory, 'photos-labels', '%d.png' % photo_id)
        fnjpg = os.path.join(directory, 'photos', '%d.jpg' % photo_id)

        if wants('material', categories) or wants('object', categories):
            labels = imread(fnpng)
            # Opensurfaces has some completely unlabled images; omit them.
            # TODO: figure out how to do this nicely in the joiner instead.
            # if labels.max() == 0:
            #     return {}
        result = {}
        if wants('scene', categories) and wants('scene', supply):
            result['scene'] = scene
        if wants('material', categories) and wants('material', supply):
            result['material'] = labels[:,:,0]
        if wants('object', categories) and wants('object', supply):
            result['object'] = labels[:,:,1]
        if wants('color', categories) and wants('color', supply):
            result['color'] = colorname.label_major_colors(imread(fnjpg)) + 1
        arrs = [a for a in result.values() if numpy.shape(a) >= 2]
        shape = arrs[0].shape[-2:] if arrs else (1, 1)
        return result, shape

def norm_name(s):
    return s.replace(' - ', '-').replace('/', '-').strip().lower()

def wants(what, option):
    if option is None:
        return True
    return what in option
