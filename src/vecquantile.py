import numpy

class QuantileVector:
    """
    Streaming randomized quantile computation for numpy.

    Add any amount of data repeatedly via add(data).  At any time,
    quantile estimates (or old-style percentiles) can be read out using
    quantiles(q) or percentiles(p).

    Accuracy scales according to resolution: the default is to
    set resolution to be accurate to better than 0.1%,
    while limiting storage to about 50,000 samples.

    Good for computing quantiles of huge data without using much memory.
    Works well on arbitrary data with probability near 1.

    Based on the optimal KLL quantile algorithm by Karnin, Lang, and Liberty
    from FOCS 2016.  http://ieee-focs.org/FOCS-2016-Papers/3933a071.pdf
    """

    def __init__(self, depth=1, resolution=24 * 1024, buffersize=None,
            dtype=None, seed=None):
        self.resolution = resolution
        self.depth = depth
        # Default buffersize: 128 samples (and smaller than resolution).
        if buffersize is None:
            buffersize = min(128, (resolution + 7) // 8)
        self.buffersize = buffersize
        self.samplerate = 1.0
        self.data = [numpy.zeros(shape=(depth, resolution), dtype=dtype)]
        self.firstfree = [0]
        self.random = numpy.random.RandomState(seed)
        self.extremes = numpy.empty(shape=(depth, 2), dtype=dtype)
        self.extremes.fill(numpy.NaN)
        self.size = 0

    def add(self, incoming):
        assert len(incoming.shape) == 2
        assert incoming.shape[1] == self.depth
        self.size += incoming.shape[0]
        # Convert to a flat numpy array.
        if self.samplerate >= 1.0:
            self._add_every(incoming)
            return
        # If we are sampling, then subsample a large chunk at a time.
        self._scan_extremes(incoming)
        chunksize = numpy.ceil[self.buffersize / self.samplerate]
        for index in xrange(0, len(incoming), chunksize):
            batch = incoming[index:index+chunksize]
            sample = batch[self.random.binomial(1, self.samplerate, len(batch))]
            self._add_every(sample)

    def _add_every(self, incoming):
        supplied = len(incoming)
        index = 0
        while index < supplied:
            ff = self.firstfree[0]
            available = self.data[0].shape[1] - ff
            if available == 0:
                if not self._shift():
                    # If we shifted by subsampling, then subsample.
                    incoming = incoming[index:]
                    if self.samplerate >= 0.5:
                        print 'SAMPLING'
                        self._scan_extremes(incoming)
                    incoming = incoming[self.random.binomial(1, 0.5,
                        len(incoming - index))]
                    index = 0
                    supplied = len(incoming)
                ff = self.firstfree[0]
                available = self.data[0].shape[1] - ff
            copycount = min(available, supplied - index)
            self.data[0][:,ff:ff + copycount] = numpy.transpose(
                    incoming[index:index + copycount,:])
            self.firstfree[0] += copycount
            index += copycount

    def _shift(self):
        index = 0
        # If remaining space at the current layer is less than half prev
        # buffer size (rounding up), then we need to shift it up to ensure
        # enough space for future shifting.
        while self.data[index].shape[1] - self.firstfree[index] < (
                -(-self.data[index-1].shape[1] // 2) if index else 1):
            if index + 1 >= len(self.data):
                return self._expand()
            data = self.data[index][:,0:self.firstfree[index]]
            data.sort()
            if index == 0 and self.samplerate >= 1.0:
                self._update_extremes(data[:,0], data[:,-1])
            offset = self.random.binomial(1, 0.5)
            position = self.firstfree[index + 1]
            subset = data[:,offset::2]
            self.data[index + 1][:,position:position + subset.shape[1]] = subset
            self.firstfree[index] = 0
            self.firstfree[index + 1] += subset.shape[1]
            index += 1
        return True

    def _scan_extremes(self, incoming):
        # When sampling, we need to scan every item still to get extremes
        self._update_extremes(
                numpy.nanmin(incoming, axis=0),
                numpy.nanmax(incoming, axis=0))

    def _update_extremes(self, minr, maxr):
        self.extremes[:,0] = numpy.nanmin(
                [self.extremes[:, 0], minr], axis=0)
        self.extremes[:,-1] = numpy.nanmax(
                [self.extremes[:, -1], maxr], axis=0)

    def minmax(self):
        if self.firstfree[0]:
            self._scan_extremes(self.data[0][:,:self.firstfree[0]].transpose())
        return self.extremes.copy()

    def _expand(self):
        cap = self._next_capacity()
        if cap > 0:
            # First, make a new layer of the proper capacity.
            self.data.insert(0, numpy.empty(
                shape=(self.depth, cap), dtype=self.data[-1].dtype))
            self.firstfree.insert(0, 0)
        else:
            # Unless we're so big we are just subsampling.
            assert self.firstfree[0] == 0
            self.samplerate *= 0.5
        for index in range(1, len(self.data)):
            # Scan for existing data that needs to be moved down a level.
            amount = self.firstfree[index]
            if amount == 0:
                continue
            position = self.firstfree[index-1]
            # Move data down if it would leave enough empty space there
            # This is the key invariant: enough empty space to fit half
            # of the previous level's buffer size (rounding up)
            if self.data[index-1].shape[1] - (amount + position) >= (
                    -(-self.data[index-2].shape[1] // 2) if (index-1) else 1):
                self.data[index-1][:,position:position + amount] = (
                        self.data[index][:,:amount])
                self.firstfree[index-1] += amount
                self.firstfree[index] = 0
            else:
                # Scrunch the data if it would not.
                data = self.data[index][:,:amount]
                data.sort()
                if index == 1:
                    self._update_extremes(data[:,0], data[:,-1])
                offset = self.random.binomial(1, 0.5)
                scrunched = data[:,offset::2]
                self.data[index][:,:scrunched.shape[1]] = scrunched
                self.firstfree[index] = scrunched.shape[1]
        return cap > 0

    def _next_capacity(self):
        cap = numpy.ceil(self.resolution * numpy.power(0.67, len(self.data)))
        if cap < 2:
            return 0
        return max(self.buffersize, int(cap))

    def _weighted_summary(self, sort=True):
        if self.firstfree[0]:
            self._scan_extremes(self.data[0][:,:self.firstfree[0]].transpose())
        size = sum(self.firstfree) + 2
        weights = numpy.empty(
            shape=(size), dtype='float32') # floating point
        summary = numpy.empty(
            shape=(self.depth, size), dtype=self.data[-1].dtype)
        weights[0:2] = 0
        summary[:,0:2] = self.extremes
        index = 2
        for level, ff in enumerate(self.firstfree):
            if ff == 0:
                continue
            summary[:,index:index + ff] = self.data[level][:,:ff]
            weights[index:index + ff] = numpy.power(2.0, level)
            index += ff
        assert index == summary.shape[1]
        if sort:
            order = numpy.argsort(summary)
            summary = summary[numpy.arange(self.depth)[:,None], order]
            weights = weights[order]
        return (summary, weights)

    def quantiles(self, quantiles, old_style=False):
        if self.size == 0:
            return numpy.full((self.depth, len(quantiles)), numpy.nan)
        summary, weights = self._weighted_summary()
        cumweights = numpy.cumsum(weights, axis=-1) - weights / 2
        if old_style:
            # To be convenient with numpy.percentile
            cumweights -= cumweights[:,0:1]
            cumweights /= cumweights[:,-1:]
        else:
            cumweights /= numpy.sum(weights, axis=-1, keepdims=True)
        result = numpy.empty(shape=(self.depth, len(quantiles)))
        for d in xrange(self.depth):
            result[d] = numpy.interp(quantiles, cumweights[d], summary[d])
        return result

    def integrate(self, fun):
        result = None
        for level, ff in enumerate(self.firstfree):
            if ff == 0:
                continue
            term = numpy.sum(
                    fun(self.data[level][:,:ff]) * numpy.power(2.0, level),
                    axis=-1)
            if result is None:
                result = term
            else:
                result += term
        if result is not None:
            result /= self.samplerate
        return result

    def percentiles(self, percentiles):
        return self.quantiles(percentiles, old_style=True)

    def readout(self, count, old_style=True):
        return self.quantiles(
                numpy.linspace(0.0, 1.0, count), old_style=old_style)


if __name__ == '__main__':
    import time
    # An adverarial case: we keep finding more numbers in the middle
    # as the stream goes on.
    amount = 10000000
    percentiles = 1000
    data = numpy.arange(float(amount))
    data[1::2] = data[-1::-2] + (len(data) - 1)
    data /= 2
    depth = 50
    alldata = data[:,None] + (numpy.arange(depth) * amount)[None, :]
    actual_sum = numpy.sum(alldata * alldata, axis=0)
    amt = amount // depth
    for r in range(depth):
        numpy.random.shuffle(alldata[r*amt:r*amt+amt,r])
    # data[::2] = data[-2::-2]
    # numpy.random.shuffle(data)
    starttime = time.time()
    qc = QuantileVector(depth=depth, resolution=8 * 1024)
    qc.add(alldata)
    ro = qc.readout(1001)
    endtime = time.time()
    # print 'ro', ro
    # print ro - numpy.linspace(0, amount, percentiles+1)
    gt = numpy.linspace(0, amount, percentiles+1)[None,:] + (
            numpy.arange(qc.depth) * amount)[:,None]
    print "Maximum relative deviation among %d perentiles:" % percentiles, (
            numpy.max(abs(ro - gt) / amount) * percentiles)
    print "Minmax eror %f, %f" % (
        max(abs(qc.minmax()[:,0] - numpy.arange(qc.depth) * amount)),
        max(abs(qc.minmax()[:, -1] - (numpy.arange(qc.depth)+1) * amount + 1)))
    print "Integral error:", numpy.max(numpy.abs(
            qc.integrate(lambda x: x * x)
            - actual_sum) / actual_sum)
    print "Count error: ", (qc.integrate(lambda x: numpy.ones(x.shape[-1])
            ) - qc.size) / (0.0 + qc.size)
    print "Time", (endtime - starttime)

