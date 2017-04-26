# Utilities for working with the intersection of arrays of segmentation masks.
# The general task is to produce a tally of IOU data:
# For each of a set of images for which we have two sets of masks
# m_i and n_j we wish to know
#
#   (a) how many pixels are present in mask m_i and n_j for each i and j
#   (b) how many pixels are present in mask m_i * n_j for each (i,j) pair

# TALLY FORMAT for results.
# Because masks are chosen to selective, most result numbers are zero,
# so we encode the results as a list of triples in a (result-length, 3)
# array, where the result-length may be fixed (e.g., 2048) to limit the
# maximum number of entries to remember.  Each triple (i, j, count)
# indicates the number of pixels intersecting m_i and n_j; (i, -1, count)
# indicates m_i alone and (j, -1, count) indicates n_i alone.

# MASKSET FORMAT for inputs.
# The natural format for a mask is as a boolean array, with one bit
# per pixel in the image.  This format represents a mask as an array:
# (#masks, #y, #x).

# LABEL/SCALAR FORMAT for inputs
#
# The binary format can be large and consists mostly of zeros.
# A more compressed format is the LABEL format which works as follows:
# for each pixel, there is a list of labels (i1, i2, i3...) indicating
# masks m_i1, m_i2 which are True on that pixel.  The label-length
# may be fixed (e.g., 64) to limit the number of different masks
# counted for each pixel.  Our labeled training sets arrive in this
# format, as (label-length, #y, #x) arrays.
# 
# Some masks may be present for every pixel of an image; those masks
# can be listed as SCALARS. Each scalar is assumed to represent #y x #x
# pixels.
#
# The LABEL/SCALAR format pairs a label matrix with a scalar list.

import numpy

def tallySelfMask(maskset, symmetric=False, out=None, verbose=False):
    labels = labelsOnlyFromMaskset(maskset)
    out = tallySelfLabelsOnly(labels, symmetric=symmetric,
            out=out, verbose=verbose)
    out[:,:2] -= 1
    return out

def tallySelfMaskOld(maskset, out=None):
    ls = labelsFromMaskset(maskset)
    out = tallySelfLabels(ls, out=out)
    out[:,:2] -= 1
    return out

def tallyMasks(maskset1, maskset2, out=None, verbose=False):
    ls1 = labelsFromMaskset(maskset1)
    ls2 = labelsFromMaskset(maskset2)
    out = tallyLabels(ls1, ls2, out=out, verbose=verbose)
    out[:,:2][out[:,:2] > 0] -= 1
    return out

def tallyMaskLabel(maskset1, data2, out=None, verbose=False):
    ls1 = labelsFromMaskset(maskset1)
    out = tallyLabels(ls1, data2, out=out, verbose=verbose)
    out[:,0][out[:,0] > 0] -= 1
    return out

def tallyThresholdLabel(array1, data2,
        threshold=0.0, fieldmap=None, shape=None, scaleshape=None,
        out=None, verbose=False):
    ls1 = labelsFromThreshold(array1, threshold, fieldmap, shape, scaleshape)
    out = tallyLabels(ls1, data2, out=out, verbose=verbose)
    out[:,0][out[:,0] > 0] -= 1
    return out

def tallyLabels(data1, data2, minlength=None, maxlength=None, out=None,
        verbose=False):
    if out is not None:
        maxlength = out.shape[0]
    label1, scalar1 = data1
    label2, scalar2 = data2
    pixelcount = label1.shape[1] * label1.shape[2]
    result = []

    # Use -1 to encode "unconditional sum"; and concatentate this to scalars
    # Create a list [-1,1,1],[3,1,1],[5,1,1],[u1,1,1]...
    oscalar1 = numpy.ones((len(scalar1) + 1, 3), dtype=numpy.int32)
    oscalar1[0,0] = -1
    oscalar1[1:,0] = scalar1
    # Create a list [1,-1,1],[1,7,1],[1,8,1],[1,u2,1]...
    oscalar2 = numpy.ones((len(scalar2) + 1, 3), dtype=numpy.int32)
    oscalar2[0,1] = -1
    oscalar2[1:,1] = scalar2

    # STEP 1: scalar * scalar.  Tally up scalar products
    if len(scalar1) + len(scalar2):
        # Form tallies of all combinations of these scalars, and drop the -1,-1.
        result.append(numpy.insert(
            numpy.einsum('ij,kj->ikj', oscalar1[:,:2],
                oscalar2[:,:2], dtype=numpy.int32).reshape(-1, 2),
            2, values=pixelcount, axis=1)[1:])

    # STEP 2: scalar * data, data * scalar. Tally each label indepdently
    count1 = numpy.bincount(label1.ravel())
    count1[:1] = 0
    count2 = numpy.bincount(label2.ravel())
    count2[:1] = 0
    # Which indexes are present?
    index1 = count1.nonzero()[0]
    index2 = count2.nonzero()[0]
    # [47,4,1],[21,6,1],[73,7,1],[count,u1,1]...
    ctally1 = numpy.ones((len(index1), 3), dtype=numpy.int32)
    ctally1[:,2] = count1[index1]
    ctally1[:,0] = index1
    # [12,1,2],[61,1,5],[33,1,6],[count,1,u2]....
    ctally2 = numpy.ones((len(index2), 3), dtype=numpy.int32)
    ctally2[:,2] = count2[index2]
    ctally2[:,1] = index2
    # Combine scalars with units, copying counts
    result.append(numpy.einsum('ij,kj->ikj',
        oscalar1, ctally2, dtype=numpy.int32).reshape(-1, 3))
    result.append(numpy.einsum('ij,kj->ikj',
        oscalar2, ctally1, dtype=numpy.int32).reshape(-1, 3))

    # Scalars on their own are already too big?
    length = sum(len(t) for t in result)
    if maxlength is not None:
        if length >= maxlength:
            if length > maxlength:
                # Keep the largest ones only
                stally = numpy.concatenate(result)
                keep = numpy.argsort(-stally[:,2])[:maxlength]
                keep.sort()
                stally = stally[keep]
                return concatAndZero([stally], minlength, out)
            return concatAndZero(result, minlength, out)
        # Our maxlength is reduced by whatever we have already taken
        maxlength -= length


    if len(count1) * len(count2):
        # STEP3: data * data. Tally up intersections in the main data array
        # We have transformed a product of masks into this sum.
        label1max = len(count1)
        label2max = len(count2)
        product = (label1 * label2max)[:,None,:,:] + label2[None,:,:,:]
        # This tally contains the result
        tally = numpy.bincount(product.ravel(), minlength=label1max * label2max)
        tally.shape = (label1max, label2max)
        tally[:1,:] = 0
        tally[:,:1] = 0

        # Now we must encode the tally into a smaller array
        indexes = tally.nonzero()
        # Truncate smallest counts if needed
        if maxlength is not None and len(indexes[0]) > maxlength:
            keep = (-tally[indexes]).argsort()[:maxlength]
            if verbose:
                print 'discarding', len(indexes[0]) - maxlength, (
                        'counts up to'), tally[indexes][keep[-1]]
            indexes = (indexes[0][keep], indexes[1][keep])
        # The intersection results
        c = tally[indexes][:,None]
        u1 = (indexes[0])[:,None]
        u2 = (indexes[1])[:,None]
        result.append(numpy.concatenate((u1, u2, c), axis=1))

    return concatAndZero(result, minlength, out)

def tallySelfLabels(data, minlength=None, maxlength=None,
        out=None, verbose=False):
    if out is not None:
        maxlength = out.shape[0]
    label, scalar = data
    count = numpy.bincount(label.ravel())
    if len(count) == 0:
        if out is not None:
            out[...] = 0
        return out
    count[0] = 0
    result = []

    if len(scalar):
        # STEP 1: scalar * scalar.  Tally up scalar products
        pixelcount = label.shape[1] * label.shape[2]
        # Create a list [3,1,1],[5,1,1],[u1,1,1]...
        oscalar1 = numpy.ones((len(scalar), 3), dtype=numpy.int32)
        oscalar1[:,0] = scalar
        oscalar2 = numpy.ones((len(scalar), 3), dtype=numpy.int32)
        oscalar2[:,1] = scalar

        # Form tallies of all combinations of these scalars, and drop the -1,-1.
        result.append(numpy.insert(
            numpy.einsum('ij,kj->ikj', oscalar1[:,:2],
                oscalar2[:,:2], dtype=numpy.int32).reshape(-1, 2),
            2, values=pixelcount, axis=1))

        # STEP 2: scalar * data, data * scalar. Tally each label indepdently
        # Which indexes are present?
        index = count.nonzero()[0]
        # [47,4,1],[21,6,1],[73,7,1],[count,u1,1]...
        ctally1 = numpy.ones((len(index), 3), dtype=numpy.int32)
        ctally1[:,2] = count[index]
        ctally1[:,0] = index
        # [12,1,2],[61,1,5],[33,1,6],[count,1,u2]....
        ctally2 = numpy.ones((len(index), 3), dtype=numpy.int32)
        ctally2[:,2] = count[index]
        ctally2[:,1] = index
        # Combine scalars with units, copying counts
        result.append(numpy.einsum('ij,kj->ikj',
            oscalar1, ctally2, dtype=numpy.int32).reshape(-1, 3))
        result.append(numpy.einsum('ij,kj->ikj',
            oscalar2, ctally1, dtype=numpy.int32).reshape(-1, 3))

        # Scalars on their own are already too big?
        length = sum(len(t) for t in result)
        if maxlength is not None:
            if length >= maxlength:
                if length > maxlength:
                    # Keep the largest ones only
                    stally = numpy.concatenate(result)
                    keep = numpy.argsort(-stally[:,2])[:maxlength]
                    keep.sort()
                    stally = stally[keep]
                    return concatAndZero([stally], minlength, out)
                return concatAndZero(result, minlength, out)
            # Our maxlength is reduced by whatever we have already taken
            maxlength -= length

    if len(label):
        # STEP3: data * data. Tally up intersections in the main data array
        # We have transformed a product of masks into this sum.
        labelmax = len(count)
        product = (label * labelmax)[:,None,:,:] + label[None,:,:,:]
        # This tally contains the result
        tally = numpy.bincount(product.ravel(), minlength=labelmax * labelmax)
        tally.shape = (labelmax, labelmax)
        tally[0,:] = 0
        tally[:,0] = 0

        # Now we must encode the tally into a smaller array
        indexes = tally.nonzero()
        # Truncate smallest counts if needed
        if maxlength is not None and len(indexes[0]) > maxlength:
            keep = (-tally[indexes]).argsort()[:maxlength]
            if verbose:
                print 'discarding', len(indexes[0]) - maxlength, (
                        'counts up to'), tally[indexes][keep[-1]]
            indexes = (indexes[0][keep], indexes[1][keep])
        # The intersection results
        c = tally[indexes][:,None]
        u1 = (indexes[0])[:,None]
        u2 = (indexes[1])[:,None]
        result.append(numpy.concatenate((u1, u2, c), axis=1))

    return concatAndZero(result, minlength, out)

def tallySelfLabelsOnly(label, minlength=None, maxlength=None, symmetric=False,
        out=None, verbose=False):
    count = numpy.bincount(label.ravel())
    if len(count) == 0:
        if out is not None:
            out[...] = 0
        if verbose:
            print 'returning 0'
        return out
    count[0] = 0
    result = []
    if out is not None:
        maxlength = out.shape[0]

    # STEP3: data * data. Tally up intersections in the main data array
    # We have transformed a product of masks into this sum.
    labelmax = len(count)
    product = (label * labelmax)[:,None,:,:] + label[None,:,:,:]
    if not symmetric:
        product = numpy.triu(numpy.transpose(product, (2, 3, 0, 1)))
    # This tally contains the result
    tally = numpy.bincount(product.ravel(), minlength=labelmax * labelmax)
    tally.shape = (labelmax, labelmax)
    tally[0,:] = 0
    tally[:,0] = 0

    # Now we must encode the tally into a smaller array
    indexes = tally.nonzero()
    # Truncate smallest counts if needed
    if maxlength is not None and len(indexes[0]) > maxlength:
        keep = (-tally[indexes]).argsort()[:maxlength]
        if verbose:
            print 'discarding', len(indexes[0]) - maxlength, (
                    'counts up to'), tally[indexes][keep[-1]]
        indexes = (indexes[0][keep], indexes[1][keep])
    # The intersection results
    c = tally[indexes][:,None]
    u1 = (indexes[0])[:,None]
    u2 = (indexes[1])[:,None]
    result.append(numpy.concatenate((u1, u2, c), axis=1))

    return concatAndZero(result, minlength, out)

def slideup(sparse, minlength=None, maxlength=None, out=None):
    '''Creates a matrix in which all the entries of 2d matrix sparse
    are slid up so that all the zeros are below all the
    nonzeros.  Chooses the height of the matrix to the minimum
    possible between minlength and maxlength (inclusive), or
    fills the out array as given.  If more nonzeros exist than
    there is space, the rightmost ones are truncated.'''
    # if an output matrix is given, that defines the maxlength
    if out is not None:
        maxlength = out.shape[0]
    # where are the nonzeros?  Transpose to get in col-major order.
    nonzeros = sparse.transpose().nonzero()
    # colind repeats col# for each nz 0,0,0,1,1,1,3,3,5,5,5,5.
    # we will use this to synthesize a compacted row-index compactind
    colind = nonzeros[0]
    rowind = nonzeros[1]
    if len(colind) == 0:
        depth = 0
    else:
        # bincount is # of each colind:   3,3,0,2,0,4
        bincount = numpy.bincount(colind)
        # cumsum[i] counts # less than i: 0,3,6,6,8,8
        depth = bincount.max()
        cumsum = numpy.empty(bincount.shape, dtype=numpy.int32)
        cumsum[0] = 0
        numpy.cumsum(bincount[:-1], out=cumsum[1:])
        # firstind has index of first in run:  0,0,0,3,3,3,6,6,8,8,8,8
        firstind = cumsum[colind]
        # compactind counts each run densely:  0,1,2,0,1,2,0,1,0,1,2,3
        compactind = numpy.arange(len(colind), dtype=numpy.int32) - firstind
        # truncate depth if needed
        if maxlength is not None and depth > maxlength:
            keep = (compactind < maxlength)
            colind = colind[keep]
            rowind = rowind[keep]
            compactind = compactind[keep]
            depth = maxlength
    # allocate the output matrix
    if out is None:
        # pad depth if needed
        if minlength is not None and depth < minlength:
            depth = minlength
        out = numpy.zeros((depth, sparse.shape[1]), dtype=sparse.dtype)
    # slide left into the output matrix
    if len(colind) > 0:
        out[compactind, colind] = sparse[rowind, colind]
    return out


def slideleft(sparse, minlength=None, maxlength=None, out=None):
    '''Creates a matrix in which all the entries of 2d matrix sparse
    are slid left so that all the zeros are to the right of all the
    nonzeros.  Chooses the width of the matrix to the minimum
    possible between minlength and maxlength (inclusive), or
    fills the out array as given.  If more nonzeros exist than
    there is space, the rightmost ones are truncated.'''
    # if an output matrix is given, that defines the maxlength
    if out is not None:
        maxlength = out.shape[1]
    # where are the nonzeros?
    nonzeros = sparse.nonzero()
    # rowind repeats row# for each nz 0,0,0,1,1,1,3,3,5,5,5,5.
    rowind = nonzeros[0]
    colind = nonzeros[1]
    # bincount is # of each rowind:   3,3,0,2,0,4
    bincount = numpy.bincount(rowind)
    depth = bincount.max()
    # cumsum[i] counts # less than i: 0,3,6,6,8,8
    cumsum = numpy.empty(bincount.shape, dtype=numpy.int32)
    cumsum[0] = 0
    numpy.cumsum(bincount[:-1], cumsum[1:])
    # firstind has index of first in run:  0,0,0,3,3,3,6,6,8,8,8,8
    firstind = cumsum[rowind]
    # compactind counts each run densely:  0,1,2,0,1,2,0,1,0,1,2,3
    compactind = numpy.arange(sparse.shape[0], dtype=numpy.int32) - firstind
    # truncate depth if needed
    if maxlength is not None and depth > maxlength:
        keep = (compactind < maxlength)
        rowind = rowind[keep]
        colind = colind[keep]
        compactind = compactind[keep]
        depth = maxlength
    # slide left into the output matrix
    if out is None:
        # pad depth if needed
        if minlength is not None and depth < minlength:
            depth = minlength
        out = numpy.zeros((sparse.shape[0], depth), dtype=sparse.dtype)
    out[rowind, compactind] = sparse[rowind, colind]
    return out

empty_int_array = numpy.array([], dtype='int')

def labelsFromThreshold(array, minlength=None, maxlength=None,
        threshold=0.0, fieldmap=None, shape=None, scaleshape=None):
    '''Reduces the maskset array to a label array and scalar list.'''
    # First, only conside units with nonzero labels
    candidates = (array.max(axis=(1,2)) > threshold).nonzero()
    # Now upsample ONLY THE CANDIDATES!
    maskset = upsample.upsampleL(array[candidates,...],
            fieldmap=fieldmap, shape=shape, scaleshape=scaleshape) > threshold
    # # Continue on before as a normal maskset
    # full = maskset.shape[1] * maskset.shape[2]
    # sums = maskset.sum(axis=(1,2))
    # zerosums = (sums == 0)
    # fullsums = (sums == full)
    # nzu = sums.nonzero()[0]                       # nonzero units
    # scalar = (sums == full).nonzero()[0] + 1      # (full mask)
    # pixu = (~(zerosums | fullsums)).nonzero()[0]  # pixel coded units
    # # Encode every pixel with its channel number (mapped!)
    # labeled = maskset[pixu, ...] * (candidates[pixu] + 1)[:,None,None]

    # Simpler approach without the "scalar" optimization:
    scalar = empty_int_array
    labeled = maskset * (candidates + 1)[:,None,None]
    # Flatten and slide-up the labeled array, then unflatten
    labeled.shape = (maskset.shape[0], full)
    compact = slideup(labeled, minlength=minlength, maxlength=maxlength)
    compact.shape = (compact.shape[0], maskset.shape[1], maskset.shape[2])
    return (compact, scalar)

def labelsFromMaskset(maskset, minlength=None, maxlength=None):
    '''Reduces the maskset array to a label array and scalar list.'''
    # First, only consider nonzero, nonmax labels
    full = maskset.shape[1] * maskset.shape[2]
    sums = maskset.sum(axis=(1,2))
    zerosums = (sums == 0)
    fullsums = (sums == full)
    scalar = (sums == full).nonzero()[0] + 1      # scalar (full mask) units
    pixu = (~(zerosums | fullsums)).nonzero()[0]  # pixel coded units
    # Now reduce everything to depth.
    labeled = maskset[pixu, ...] * (pixu + 1)[:,None,None]
    # Flatten and slide-up the labeled array, then unflatten
    labeled.shape = (len(pixu), full)
    compact = slideup(labeled, minlength=minlength, maxlength=maxlength)
    compact.shape = (compact.shape[0], maskset.shape[1], maskset.shape[2])
    return (compact, scalar)

def labelsOnlyFromMaskset(maskset, minlength=None, maxlength=None):
    '''Reduces the maskset array to a label array and scalar list.'''
    # First, only consider nonzero, nonmax labels
    full = maskset.shape[1] * maskset.shape[2]
    sums = maskset.sum(axis=(1,2))
    zerosums = (sums == 0)
    pixu = (~(zerosums)).nonzero()[0]             # pixel coded units
    # Now reduce everything to depth.
    labeled = maskset[pixu, ...] * (pixu + 1)[:,None,None]
    # Flatten and slide-up the labeled array, then unflatten
    labeled.shape = (len(pixu), full)
    compact = slideup(labeled, minlength=minlength, maxlength=maxlength)
    compact.shape = (compact.shape[0], maskset.shape[1], maskset.shape[2])
    return compact

def concatAndZero(datalist, minlength=None, out=None):
    length = sum(len(d) for d in datalist)
    first = datalist[0]
    if out is None:
        out = numpy.empty((length,) + first.shape[1:], dtype=first.dtype)
    index = 0
    for d in datalist:
        if index + len(d) > len(out):
            out[index:] = d[:len(out) - index]
            index = len(out)
            break
        out[index:index + len(d)] = d
        index += len(d)
    out[index:] = 0
    return out


if __name__ == '__main__':
    maskset1 = numpy.zeros((8, 10, 10), dtype=numpy.bool)
    for z in range(8):
        for y in range(10):
            for x in range(10):
                maskset1[z,y,x] = x*x + y*y <= 4 * z*z
    maskset2 = numpy.zeros((6, 10, 10), dtype=numpy.bool)
    for z in range(6):
        for y in range(10):
            for x in range(10):
                maskset2[z,y,x] = (x > 2 * z)

    out = numpy.empty((36, 3), dtype=numpy.int32)
    result = tallyMasks(maskset1, maskset2, out)
    expect = set([
        (7, -1, 100), (-1, 0, 90), (-1, 1, 70), (-1, 2, 50), (-1, 3, 30),
        (-1, 4, 10), (7, 0, 90), (7, 1, 70), (7, 2, 50), (7, 3, 30),
        (7, 4, 10), (0, -1, 1), (1, -1, 6), (2, -1, 17), (3, -1, 35),
        (4, -1, 58), (5, -1, 88), (6, -1, 97), (6, 0, 87), (5, 0, 78),
        (6, 1, 67), (5, 1, 58), (4, 0, 49), (6, 2, 47), (5, 2, 38),
        (4, 1, 33), (3, 0, 28), (6, 3, 27), (5, 3, 20), (4, 2, 18),
        (3, 1, 16), (2, 0, 12), (6, 4, 8), (4, 3, 5), (3, 2, 5),
        (5, 4, 5)
    ])
    assert len(result) == 36
    for v in result:
        assert tuple(v) in expect
    # Include only the upper-triangular partts
    expect = set([
		(7, 7, 100),
		(0, 7, 1), (1, 7, 6),
		(2, 7, 17), (3, 7, 35), (4, 7, 58), (5, 7, 88), (6, 7, 97),
		(6, 6, 97), (5, 6, 88), (5, 5, 88),
		(4, 6, 58), (4, 5, 58), (4, 4, 58),
		(3, 6, 35), (3, 5, 35), (3, 4, 35),
		(3, 3, 35), (2, 2, 17), (2, 3, 17), (2, 4, 17), (2, 5, 17),
		(2, 6, 17),
        # because only upper-trianular are included, these now fit:
        (0, 0, 1), (0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1), (0, 5, 1),
        (0, 6, 1), (1, 1, 6), (1, 2, 6), (1, 3, 6), (1, 4, 6), (1, 5, 6),
        (1, 6, 6),
    ])
    result = tallySelfMask(maskset1, out=out)
    assert len(result) == 36
    for v in result:
        assert tuple(v) in expect
    print 'OK'
