from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.interpolation import zoom
import numpy

def upsampleL(fieldmap, activation_data, reduction=1, shape=None,
        scaleshape=None, out=None):
    '''
    Applies a bilinear upsampling.
    '''
    offset, size, step = fieldmap
    input_count = activation_data.shape[0]
    ay, ax = centered_arange(fieldmap, activation_data.shape[1:], reduction)
    if shape is None:
        shape = upsampled_shape(
            fieldmap, activation_data.shape[1:], reduction)
    if scaleshape is not None:
        iy, ix = full_arange(scaleshape)
        # TODO: consider treaing each point as a center of a pixel
        iy *= shape[0] / scaleshape[0]
        ix *= shape[1] / scaleshape[1]
    else:
        iy, ix = full_arange(shape)
    if out is None:
        out = numpy.empty((input_count, len(iy), len(ix)),
                dtype=activation_data.dtype)
    for z in range(input_count):
        f = RectBivariateSpline(ay, ax, activation_data[z], kx=1, ky=1)
        out[z] = f(iy, ix, grid=True)
    return out

def upsampleC(fieldmap, activation_data, shape=None, out=None):
    '''
    Applies a bicubic upsampling.
    '''
    offset, size, step = fieldmap
    input_count = activation_data.shape[0]
    ay, ax = centered_arange(fieldmap, activation_data.shape[1:])
    if shape is None:
        shape = upsampled_shape(fieldmap, activation_data.shape[1:])
    iy, ix = full_arange(shape)
    if out is None:
        out = numpy.empty((input_count,) + shape,
                dtype=activation_data.dtype)
    for z in range(input_count):
        f = RectBivariateSpline(ay, ax, activation_data[z], kx=3, ky=3)
        out[z] = f(iy, ix, grid=True)
    return out

def upsampleG(fieldmap, activation_data, shape=None):
    '''
    Upsampling utility functions
    '''
    offset, size, step = fieldmap
    input_count = activation_data.shape[0]
    if shape is None:
        shape = upsampled_shape(fieldmap, activation_data.shape[1:])
    activations = numpy.zeros((input_count,) + shape)
    activations[(slice(None),) +
            centered_slice(fieldmap, activation_data.shape[1:])] = (
        activation_data * numpy.prod(step))
    blurred = gaussian_filter(
        activations,
        sigma=(0, ) + tuple(t // 1.414 for o, s, t in zip(*fieldmap)),
        mode='constant')
    return blurred

def topo_sort(layers):
    # First, build a links-from and also a links-to graph
    links_from = {}
    links_to = {}
    for layer in layers:
        for bot in layer.bottom:
            if bot not in links_from:
                links_from[bot] = []
            links_from[bot].append(layer)
        for top in layer.top:
            if top not in links_to:
                links_to[top] = []
            links_to[top].append(layer)
    # Now do a DFS to figure out the ordering (using links-from)
    visited = set()
    ordering = []
    stack = []
    for seed in links_from:
        if seed not in visited:
            stack.append((seed, True))
            stack.append((seed, False))
            visited.add(seed)
            while stack:
                (blob, completed) = stack.pop()
                if completed:
                    ordering.append(blob)
                elif blob in links_from:
                    for layer in links_from[blob]:
                        for t in layer.top:
                            if t not in visited:
                                stack.append((t, True))
                                stack.append((t, False))
                                visited.add(t)
    # Return a result in front-to-back order, with incoming links for each
    return list((blob, links_to[blob] if blob in links_to else [])
            for blob in reversed(ordering))

def composed_fieldmap(layers, end):
    ts = topo_sort(layers)
    fm_record = {}
    for blob, layers in ts:
        # Compute fm's on all the edges that go to this blob.
        all_fms = [
            (compose_fieldmap(fm_record[bot][0], layer_fieldmap(layer)),
                fm_record[bot][1] + [(bot, layer)])
            for layer in layers for bot in layer.bottom if bot != blob]
        # And take the max fieldmap.
        fm_record[blob] = max_fieldmap(all_fms)
        if blob == end:
            return fm_record[blob]

def max_fieldmap(maps):
    biggest, bp = None, None
    for fm, path in maps:
        if biggest is None:
            biggest, bp = fm, path
        elif fm[1][0] > biggest[1][0]:
            biggest, bp = fm, path
    # When there is no biggest, for example when maps is the empty array,
    # use the trivial identity fieldmap with no path.
    if biggest is None:
        return ((0, 0), (1, 1), (1, 1)), []
    return biggest, bp

def shortest_layer_path(start, end, layers):
    # First, build a blob-to-outgoing-layer graph
    links_from = {}
    for layer in layers:
        for bot in layer.bottom:
            if bot not in links_from:
                links_from[bot] = []
            links_from[bot].append(layer)
    # Then do a BFS on the graph to find the shortest path to 'end'
    queue = [(s, []) for s in start]
    visited = set(start)
    while queue:
        (blob, path) = queue.pop(0)
        for layer in links_from[blob]:
            for t in layer.top:
                if t == end:
                    return path + [layer]
                if t not in visited:
                    queue.append((t, path + [layer]))
                    visited.add(t)
    return None

def upsampled_shape(fieldmap, shape, reduction=1):
    # Given the shape of a layer's activation and a fieldmap describing
    # the transformation to original image space, returns the shape of
    # the input size
    return tuple(((w - 1) * t + s + 2 * o) // reduction
            for (o, s, t), w in zip(zip(*fieldmap), shape))

def make_mask_set(image_shape, fieldmap, activation_data,
              output=None, sigma=0.1, threshold=0.5, percentile=None):
    """Creates a set of receptive field masks with uniform thresholds
    over a range of inputs.
    """
    offset, shape, step = fieldmap
    input_count = activation_data.shape[0]
    activations = numpy.zeros((input_count,) + image_shape)
    activations[(slice(None),) +
            centered_slice(fieldmap, activation_data.shape[1:])] = (
        activation_data)
    blurred = gaussian_filter(
        activations,
        sigma=(0, ) + tuple(s * sigma for s in shape),
        mode='constant')
    if percentile is not None:
        limit = blurred.ravel().percentile(percentile)
        return blurred > limit
    else:
        maximum = blurred.ravel().max()
        return (blurred > maximum * threshold)

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

def receptive_field(location, fieldmap):
    """Computes the receptive field of a specific location.

    Parameters
    ----------
    location: tuple
        The x-y position of the unit being queried.
    fieldmap:
        The (offset, size, step) tuple fieldmap representing the
        receptive field map for the layer being queried.
    """
    return compose_fieldmap(fieldmap, (location, (1, 1), (1, 1)))[:2]


def proto_getattr(p, a, d):
    hf = True
    # Try using HasField to detect the presence of a field;
    # if there is no HasField, then just use getattr.
    try:
        hf = p.HasField(a)
    except:
        pass
    if hf:
        return getattr(p, a, d)
    return d

def wh_attr(layer, attrname, default=0, minval=0):
    if not hasattr(default, '__len__'):
        default = (default, default)
    val = proto_getattr(layer, attrname, None)
    if val is None or val == []:
        h = max(minval, getattr(layer, attrname + '_h', default[0]))
        w = max(minval, getattr(layer, attrname + '_w', default[1]))
    elif hasattr(val, '__len__'):
        h = val[0]
        w = val[1] if len(val) >= 2 else h
    else:
        h = val
        w = val
    return (h, w)

def layer_fieldmap(layer):
    # Only convolutional and pooling layers affect geometry.
    if layer.type == 'Convolution' or layer.type == 'Pooling':
        if layer.type == 'Pooling':
            config = layer.pooling_param
            if config.global_pooling:
                return ((0, 0), (None, None), (1, 1))
        else:
            config = layer.convolution_param
        size = wh_attr(config, 'kernel_size', wh_attr(config, 'kernel', 1))
        stride = wh_attr(config, 'stride', 1, minval=1)
        padding = wh_attr(config, 'pad', 0)
        neg_padding = tuple((-x) for x in padding)
        return (neg_padding, size, stride)
    # All other layers just pass through geometry unchanged.
    return ((0, 0), (1, 1), (1, 1))

def layerarray_fieldmap(layerarray):
    fieldmap = ((0, 0), (1, 1), (1, 1))
    for layer in layerarray:
        fieldmap = compose_fieldmap(fieldmap, layer_fieldmap(layer))
    return fieldmap

# rf1 is the lower layer, rf2 is the higher layer
def compose_fieldmap(rf1, rf2):
    """Composes two stacked fieldmap maps.

    Field maps are represented as triples of (offset, size, step),
    where each is an (x, y) pair.

    To find the pixel range corresponding to output pixel (x, y), just
    do the following:
       start_x = x * step[0] + offset[1]
       limit_x = start_x + size[0]
       start_y = y * step[1] + offset[1]
       limit_y = start_y + size[1]

    Parameters
    ----------
    rf1: tuple
        The lower-layer receptive fieldmap, a tuple of (offset, size, step).
    rf2: tuple
        The higher-layer receptive fieldmap, a tuple of (offset, size, step).
    """
    if rf1 == None:
        import pdb; pdb.set_trace()
    offset1, size1, step1 = rf1
    offset2, size2, step2 = rf2

    size = tuple((size2c - 1) * step1c + size1c
            for size1c, step1c, size2c in zip(size1, step1, size2))
    offset = tuple(offset2c * step1c + offset1c
            for offset2c, step1c, offset1c in zip(offset2, step1, offset1))
    step = tuple(step2c * step1c
            for step1c, step2c in zip(step1, step2))
    return (offset, size, step)

def _cropped_slices(offset, size, limit):
    corner = 0
    if offset < 0:
        size += offset
        offset = 0
    if limit - offset < size:
        corner = limit - offset
        size -= corner
    return (slice(corner, corner + size), slice(offset, offset + size))

def crop_field(image_data, fieldmap, location):
    """Crops image_data to the specified receptive field.

    Together fieldmap and location specify a receptive field on the image,
    which may overlap the edge. This returns a crop to that shape, including
    any zero padding necessary to fill out the shape beyond the image edge.
    """
    offset, size = receptive_field(fieldmap, location)
    return crop_rectangle(image_data, offset, size)

def crop_rectangle(image_data, offset, size):
    coloraxis = 0 if image_data.size <= 2 else 1
    allcolors = () if not coloraxis else (slice(None),) * coloraxis
    colordepth = () if not coloraxis else (image_data.size[0], )
    result = numpy.zeros(colordepth + size)
    (xto, xfrom), (yto, yfrom) = (_cropped_slices(
        o, s, l) for o, s, l in zip(offset, size, image_data.size[coloraxis:]))
    result[allcolors + (xto, yto)] = image_data[allcolors + (xfrom, yfrom)]
    return result

def center_location(fieldmap, location):
    if isinstance(location, numpy.ndarray):
        offset, size, step = fieldmap
        broadcast = (numpy.newaxis, ) * (len(location.shape) - 1) + (
                        slice(None),)
        step = numpy.array(step)[broadcast]
        offset = numpy.array(offset)[broadcast]
        size = numpy.array(size)[broadcast]
        return location * step + offset + size // 2
    else:
        offset, shape = receptive_field(location, fieldmap)
        return tuple(o + s // 2 for o, s in zip(offset, shape))

def centered_slice(fieldmap, activation_shape, reduction=1):
    offset, size, step = fieldmap
    r = reduction
    return tuple(slice((s // 2 + o) // r, (s // 2 + o + a * t) // r, t // r)
            for o, s, t, a in zip(offset, size, step, activation_shape))

def centered_arange(fieldmap, activation_shape, reduction=1):
    offset, size, step = fieldmap
    r = reduction
    return tuple(numpy.arange(
        (s // 2 + o) // r, (s // 2 + o + a * t) // r, t // r)[:a] # Hack to avoid a+1 points
            for o, s, t, a in zip(offset, size, step, activation_shape))

def full_arange(output_shape):
    return tuple(numpy.arange(o) for o in output_shape)

