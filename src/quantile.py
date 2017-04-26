import numpy

class QuantileCounter:
    """
    Streaming randomized quantile computation for numpy.

    Add any amount of data repeatedly via add(data).  At any time,
    quantile estimates (or old-style percentiles) can be read out using
    quantiles(q) or percentiles(p).

    Accuracy scales according to resolution: the default is to
    set resolution to be almost-always accurate to approximately 0.1%,
    limiting storage to about 100,000 samples.

    Good for computing quantiles of huge data without using much memory.
    Works well on arbitrary data with probability near 1.

    Based on the optimal KLL quantile algorithm by Karnin, Lang, and Liberty
    from FOCS 2016.  http://ieee-focs.org/FOCS-2016-Papers/3933a071.pdf
    """

    def __init__(self, resolution=32 * 1024, buffersize=None,
            dtype=None, seed=None):
        self.resolution = resolution
        # Default buffersize: 4096 samples (and smaller than resolution).
        if buffersize is None:
            buffersize = min(512, (resolution + 3) // 4)
        self.buffersize = buffersize
        self.samplerate = 1.0
        self.data = [numpy.zeros(shape=resolution, dtype=dtype)]
        self.firstfree = [0]
        # The 0th buffer is left marked full always
        self.full = numpy.ones(shape=1, dtype='bool')
        self.random = numpy.random.RandomState(seed)

    def add(self, incoming):
        # Convert to a flat numpy array.
        incoming = numpy.ravel(incoming)
        if self.samplerate >= 1.0:
            self._add_every(incoming)
            return
        # If we are sampling, then subsample a large chunk at a time.
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
            available = len(self.data[0]) - ff
            if available == 0:
                if not self._shift():
                    # If we shifted by subsampling, then subsample.
                    incoming = incoming[index:][self.random.binomial(1, 0.5,
                        len(incoming - index))]
                    index = 0
                    supplied = len(incoming)
                ff = self.firstfree[0]
                available = len(self.data[0]) - ff
            copycount = min(available, supplied - index)
            self.data[0][ff:ff + copycount] = incoming[index:index + copycount]
            self.firstfree[0] += copycount
            index += copycount

    def _shift(self):
        index = 0
        while self.firstfree[index] * 2 > len(self.data[index]):
            if index + 1 >= len(self.data):
                return self._expand()
            data = self.data[index][0:self.firstfree[index]]
            data.sort()
            offset = self.random.binomial(1, 0.5)
            position = self.firstfree[index + 1]
            subset = data[offset::2]
            self.data[index + 1][position:position + len(subset)] = subset
            self.firstfree[index] = 0
            self.firstfree[index + 1] += len(subset)
            index += 1
        return True

    def _expand(self):
        cap = self._next_capacity()
        if cap > 0:
            # First, make a new layer of the proper capacity.
            self.data.insert(0,
                    numpy.empty(shape=cap, dtype=self.data[-1].dtype))
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
            position = self.firstfree[index - 1]
            if (amount + position) * 2 <= len(self.data[index - 1]):
                # Move the data down if it would leave things half-empty.
                self.data[index - 1][position:position + amount] = (
                        self.data[index][:amount])
                self.firstfree[index - 1] += amount
                self.firstfree[index] = 0
            else:
                # Scrunch the data if it would not.
                data = self.data[index][:amount]
                data.sort()
                offset = self.random.binomial(1, 0.5)
                scrunched = data[offset::2]
                self.data[index][:len(scrunched)] = scrunched
                self.firstfree[index] = len(scrunched)
        return cap > 0

    def _next_capacity(self):
        cap = numpy.ceil(self.resolution * numpy.power(0.67, len(self.data)))
        if cap < 2:
            return 0
        return max(self.buffersize, int(cap))

    def quantiles(self, quantiles, old_style=False):
        size = sum(self.firstfree)
        weights = numpy.empty(shape=size, dtype='float32') # floating point
        summary = numpy.empty(shape=size, dtype=self.data[-1].dtype)
        index = 0
        for level, ff in enumerate(self.firstfree):
            if ff == 0:
                continue
            summary[index:index + ff] = self.data[level][:ff]
            weights[index:index + ff] = numpy.power(2.0, level)
            index += ff
        assert index == len(summary)
        order = numpy.argsort(summary)
        summary = summary[order]
        weights = weights[order]
        cumweights = numpy.cumsum(weights) - weights / 2
        if old_style:
            # To be convenient with numpy.percentile
            cumweights -= cumweights[0]
            cumweights /= cumweights[-1]
        else:
            cumweights /= numpy.sum(weights)
        return numpy.interp(quantiles, cumweights, summary)

    def percentiles(self, percentiles):
        return self.quantiles(percentiles, old_style=True)

    def readout(self, count, old_style=True):
        return self.quantiles(
                numpy.linspace(0.0, 1.0, count), old_style=old_style)


if __name__ == '__main__':
    # An adverarial case: we keep finding more numbers in the middle
    # as the stream goes on.
    qc = QuantileCounter()
    amount = 50000000
    percentiles = 1000
    data = numpy.arange(amount)
    data[1::2] = data[-1::-2] + (len(data) - 1)
    data /= 2
    # data[::2] = data[-2::-2]
    # numpy.random.shuffle(data)
    qc.add(data)
    ro = qc.readout(1001)
    # print ro - numpy.linspace(0, amount, percentiles+1)
    print "Maximum relative deviation among %d perentiles:" % percentiles, max(
            abs(ro - numpy.linspace(0, amount, percentiles+1))
            / amount) * percentiles

