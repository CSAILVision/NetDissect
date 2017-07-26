'''
expdir: manages the filenames and formats in a directory for one experiment.

For example, one directory might contain files such as the following.
This package does not compute the contents of these files but just
deals with the naming of them.

conv1-imgmax.mat       conv2-result.csv       conv4-quant-1000.mmap
conv1-info.txt         conv2-tally-005.mmap   conv4-rank-1001.mmap
conv1-minmax.mmap      conv3-imgmax.mat       conv4-result.csv
conv1.mmap             conv3-info.txt         conv4-tally-005.mmap
conv1-quant-1000.mat   conv3-minmax.mmap      conv5-imgmax.mat
conv1-quant-1000.mmap  conv3.mmap             conv5-info.txt
conv1-rank-1001.mmap   conv3-quant-1000.mat   conv5-minmax.mmap
conv1-result.csv       conv3-quant-1000.mmap  conv5.mmap
conv1-tally-005.mmap   conv3-rank-1001.mmap   conv5-quant-1000.mat
conv2-imgmax.mat       conv3-result.csv       conv5-quant-1000.mmap
conv2-info.txt         conv3-tally-005.mmap   conv5-rank-1001.mmap
conv2-minmax.mmap      conv4-imgmax.mat       conv5-result.csv
conv2.mmap             conv4-info.txt         conv5-tally-005.mmap
conv2-quant-1000.mat   conv4-minmax.mmap      html
conv2-quant-1000.mmap  conv4.mmap             info.txt
conv2-rank-1001.mmap   conv4-quant-1000.mat   README.txt

Examples:

ed = expdir.ExperimentDirectory('~/my-experiment')
with open(ed.filename('README.txt')) as f:
    f.write('hello')
info = ed.read_info()
ed.write_info(blob='conv5', data=dict(
    shape=(1,2,3), name="my-experiment"
))
if ed.has_mmap(blob='conv5', part='tally):
    tensor = ed.open_mmap(blob='conv5', part='tally', shape=(43,-1,3))
'''

import csv
import json
import glob
import os
import numpy
import re
import shutil

# Add basestring for python3
try:
    basestring
except NameError:
    basestring = str


# For managing a global cache of mode='r+'/'r' memmaps,
# to avoid re-opening a memmap that is already open.
global_memmap_cache = {}


class ExperimentDirectory:
    '''
    Management of the files and filenames within a dissection.
    '''

    def __init__(self, directory):
        self.directory = directory = os.path.expanduser(directory)

    def basename(self):
        return os.path.basename(self.directory)

    def filename(self, name=None, last=True, decimal=False,
            blob=None, part=None, directory=None, aspair=False):
        '''
        Expands out a filename.  For example:

        f.filename('result.csv', blob='conv5') -> %DIR/conv5-result.csv

        Supports globbing where * will look for the largest matching #, e.g.:

        f.filename('tally-*.mmap') -> %DIR/tally-1024.csv

        Set decimal=True to treat leading zeros as significant (as if the #
        is the fractional part of a decimal number).  Set aspair=True to
        return the globbed number along with the filename.
        '''
        # Handle arrays of names
        if isinstance(name, basestring):
            name = [name]
        elif name is None:
            name = []
        # Expand out dissection directory.
        path = [self.directory]
        if directory is not None:
            path.append(directory)
        # Insert part name into filename if specified.
        if part is not None:
            name.insert(0, part)
        # Insert blob name into filename if specified.
        if blob is not None:
            name.insert(0, fn_safe(blob))
        # Assemble filename
        path.append('-'.join(name))
        fn = os.path.join(*path)
        # Expand out numbered globs.
        if '*' in fn:
            n, fn = numbered_glob(fn, last=last, decimal=decimal)
            if aspair:
                return (n, fn)
        return fn

    def has(self, name=None, last=True, decimal=False,
            blob=None, part=None, directory=None, aspair=False):
        return os.path.exists(self.filename(name=name, last=last,
            decimal=decimal, blob=blob, part=part, directory=directory))

    def glob_number(self, name, last=True, decimal=False, blob=None, part=None):
        '''
        For a globbed filename, returns the matching number rather than
        the matching filename.
        '''
        n, fn = self.filename(name, last=last, decimal=decimal,
                blob=blob, part=part, aspair=True)
        return n

    def ensure_dir(self, *args):
        '''
        Creates a directory if it does not yet exist.
        '''
        dirname = os.path.join(*([self.directory] + list(args)))
        try:
            os.makedirs(dirname)
        except OSError:
            pass

    def remove_all(self, *args):
        '''
        Removes all files that match a given pattern.
        '''
        try:
            for c in glob.glob(os.path.join(*([self.directory] + args))):
                os.remove(f)
        except:
            pass

    ###################################################################
    # [blob-]info.txt
    ###################################################################

    def info_filename(self, filename=None, blob=None):
        if filename is None:
            filename = 'info.txt'
        return self.filename(filename, blob=blob)

    def has_info(self, filename=None, blob=None):
        return os.path.isfile(self.info_filename(filename, blob))

    def load_info(self, filename=None, blob=None, asdict=False):
        filename = self.info_filename(filename, blob=blob)
        info = {}
        with open(filename, 'r') as f:
            for line in f.readlines():
                k, v = line.split(':', 1)
                info[k] = eval(v.strip())
        # Return as dictionary if requested.
        if asdict:
            return info
        # Otherwise, return as an info object with attributes.
        class Info:
            pass
        result = Info()
        for k, v in info.items():
            setattr(result, k, v)
        return result

    def save_info(self, data=None, filename=None, blob=None, asdict=False):
        filename = self.info_filename(filename, blob)
        with open(filename, 'w') as f:
            for k, v in data.items():
                # Convert numpy arrays to lists for saving as text
                if type(v).__module__ == 'numpy':
                    v = v.tolist()
                f.write('%s: %r\n' % (k, v))

    ###################################################################
    # html/[blob][-part].html, defaulting to index.html
    ###################################################################
    def html_filename(self, blob=None, part=None, **kwargs):
        if blob is None and part is None:
            blob = 'index'
        result = self.filename('html/%s.html' % (
            '-'.join(filter(None, [fn_safe(blob), part])),), **kwargs)
        return result

    def save_html(self, html, blob=None, part=None, fieldnames=None):
        filename = self.html_filename(blob=blob, part=part)
        wrappers = ['html', 'body']
        with open(filename, 'w') as f:
            f.write('<!doctype html>')
            suffix = []
            for w in wrappers:
                if ('<' + w) not in html:
                    f.write('<%s>\n' % w)
                    suffix.insert(0, '</%s>\n' % w)
            f.write(html)
            f.write('\n'.join(suffix))

    ###################################################################
    # [blob][-part].csv
    ###################################################################
    def csv_filename(self, blob=None, part=None,
            inc=False, **kwargs):
        result = self.filename('%s.csv' % (
            '-'.join(filter(None, [fn_safe(blob), part])),), **kwargs)
        return result

    def load_csv(self, blob, part=None, readfields=None):
        def convert(value):
            if re.match(r'^-?\d+$', value):
                try:
                    return int(value)
                except:
                    pass
            if re.match(r'^-?[\.\d]+(?:e[+=]\d+)$', value):
                try:
                    return float(value)
                except:
                    pass
            return value
        filename = self.csv_filename(blob=blob, part=part)
        with open(filename) as f:
            reader = csv.DictReader(f)
            result = [{k: convert(v) for k, v in row.items()} for row in reader]
            if readfields is not None:
                readfields.extend(reader.fieldnames)
        return result

    def save_csv(self, data, blob, part=None, fieldnames=None):
        filename = self.csv_filename(blob=blob, part=part)
        with open(filename, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    ###################################################################
    # [blob][-part].json
    ###################################################################
    def json_filename(self, blob=None, part=None,
            inc=False, **kwargs):
        result = self.filename('%s.json' % (
            '-'.join(filter(None, [fn_safe(blob), part])),), **kwargs)
        return result

    def load_json(self, blob, part=None, readfields=None):
        filename = self.json_filename(blob=blob, part=part)
        with open(filename) as f:
            return json.load(f)

    def save_json(self, data, blob, part=None, fieldnames=None):
        filename = self.json_filename(blob=blob, part=part)
        with open(filename, 'w') as f:
            json.dump(data, f)

    ###################################################################
    # [blob][-part].mmap
    ###################################################################

    def mmap_filename(self, blob=None, part=None,
            inc=False, **kwargs):
        result = self.filename('%s.mmap%s' % (
            '-'.join(filter(None, [fn_safe(blob), part])),
            '-inc' if inc else ''), **kwargs)
        return result

    def has_mmap(self, **kwargs):
        return os.path.isfile(
                self.mmap_filename(**kwargs))

    def open_mmap(self, shape=None, mode='r', dtype='float32', **kwargs):
        '''
        Opens a numpy memmap at filename [blob][-part].mmap.
        Defaults to read-only (override with mode='w+').
        Defaults to float32 data (override with dtype='int32').
        Defaults to linear shape (override with shape=(...)).
        If the supplied shape contains a -1, then infers the missing
        dimension from the size of the file.
        '''
        knownshape = None
        if shape is not None and not isinstance(shape, (list, tuple)):
            shape = (shape,)
        if shape is not None and -1 not in shape:
            knownshape = shape
        nameargs = dict(kwargs)
        if 'inc' not in nameargs:
            nameargs['inc'] = (mode and 'w' in mode)
        aspair = ('aspair' in nameargs) and nameargs['aspair']
        if aspair:
            num, fn = self.mmap_filename(**nameargs)
        else:
            fn = self.mmap_filename(**nameargs)
        result = numpy.memmap(
                fn,
                mode=mode,
                dtype=dtype,
                shape=knownshape)
        # Supports inferring shape of one unknown dimension
        if shape is not None and -1 in shape:
            lastdim = len(result) // abs(numpy.prod(shape))
            inferred = tuple(map(lambda x: x if x >= 0 else lastdim, shape))
            result.shape = inferred
        if aspair:
            return num, result
        else:
            return result

    def cached_mmap(self, shape=None, mode='r', dtype='float32', **kwargs):
        '''
        Holds an memmap open in a global cache.  Used for using memmaps
        in a worker process without re-opening the file repeatedly.
        '''
        global global_memmap_cache
        assert 'r' in mode
        key = (mode, self.mmap_filename(**kwargs))
        if key not in global_memmap_cache:
            global_memmap_cache[key] = self.open_mmap(
                    shape=shape, mode=mode, dtype=dtype, **kwargs)
        return global_memmap_cache[key]

    def finish_mmap(self, array, delete=False):
        '''
        When a new mmap is created, it starts with the name .mmap-inc.
        Call this finish function on the array to flush it and
        rename it to .mmap.
        '''
        if isinstance(array, numpy.memmap) and array.mode and 'w' in array.mode:
            if delete:
                os.remove(array.filename)
            else:
                array.flush()
                if array.filename.endswith('-inc'):
                    os.rename(array.filename, array.filename[:-4])
        if delete:
            del array

    ###################################################################
    # [blob][-part]/ directory
    ###################################################################

    def working_dir(self, blob=None, part=None):
        result = '-'.join(filter(None, [fn_safe(blob), part]))
        self.ensure_dir(result)
        return result

    def remove_dir(self, filename):
        shutil.rmtree(self.filename(filename), ignore_errors=True)

def fn_safe(blob, dotfree=False):
    '''Sometimes a blob name will have delimiters in it which are not
    filename safe.  Fix these by turning them into hyphens.'''
    if blob is None:
        return None
    if dotfree:
        return re.sub('[\.-/#?*!\s]+', '-', blob).strip('-')
    else:
        return re.sub('[-/#?*!\s]+', '-', blob).strip('-')

def numbered_glob(pattern, last=True, decimal=False, every=False):
    '''Given a globbed file_*_pattern, returns a pair of a matching
    filename along with a matching number.'''
    repat = r'(\d+(?:\.\d*)?)'.join(
            re.escape(s) for s in pattern.split('*', 1))
    best_fn = None
    best_n = None
    all_results = []
    for c in glob.glob(pattern):
        m = re.match(repat, c)
        if m:
            if decimal:
                if '.' in m.group(1):
                    n = float(m.group(1))
                else:
                    n = float('0.' + m.group(1))
            else:
                n = int(m.group(1).strip('.'))
            all_results.append((c, n))
            if best_n is None or (best_n < n) == last:
                best_n = n
                best_fn = c
    if every:
        return all_results
    if best_fn is None:
        raise IOError(pattern)
    return best_n, best_fn

