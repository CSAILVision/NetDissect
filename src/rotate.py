import numpy
import scipy.linalg
from tempfile import NamedTemporaryFile

# Generates a random orthogonal matrix correctly.
# See http://www.ams.org/notices/200511/what-is.pdf
# Also http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf
# Equation 35

def randomRotation(n, seed=None):
    if seed is None:
        NR = numpy.random.normal(size=(n, n))
    else:
        NR = numpy.random.RandomState(seed).normal(size=(n, n))
    Q, R = numpy.linalg.qr(NR)
    # Eliminate negative diagonals
    D = numpy.diagonal(R)
    L = numpy.diagflat(D / abs(D))
    # The replacement R would be LR and have a positive diagonal.
    # This allows det(QL) to be random sign whereas Householder Q is fixed.
    result = numpy.dot(Q, L)
    # Now negate the first column if this result is a reflection
    if numpy.linalg.det(result) < 0:
        result[0] = -result[0]
    return result

def randomRotationPowers(n, powers, seed=None):
    RR = randomRotation(n, seed)
    # Reduce the matrix to canonical (block-diag) form using schur decomp
    # (TODO: Consider implementing the stabilized variant of the algorithm
    # by Gragg called UHQR which is tailored to solve this decomposition for
    # unitary RR - or 'OHQR' by Ammar for the real orthogonal case.)
    T, W = scipy.linalg.schur(RR, output='real')
    # Now T is not exactly block diagonal due to numerical approximations.
    # However, its eigenvalues should have the right distribution: let's
    # read the angles off the diagonal and use them to construct a perfectly
    # block diagonal form.
    RA = numpy.arccos(numpy.clip(numpy.diag(T), -1, 1))[0:n:2]
    # Now form the requested powers of W * (RA^p) * W'
    result = []
    for p in powers:
        A = [a * p for a in RA]
        B = [numpy.cos([[a, a + numpy.pi/2], [a - numpy.pi/2, a]]) for a in A]
        BD = scipy.linalg.block_diag(*B)
        result.append(numpy.dot(numpy.dot(W, BD), W.transpose()))
    return result

def randomNearIdentity(n, scale, seed=None):
    if numpy.isinf(scale):
        return randomOrthogonal(n, seed)
    sigma = scale / numpy.sqrt(2 * n)
    if seed is None:
        RR = numpy.random.normal(scale=sigma, size=(n, n))
    else:
        RR = numpy.random.RandomState(seed).normal(scale=sigma, size=(n, n))
    U, _, V = numpy.linalg.svd(numpy.eye(n) + RR)
    return numpy.dot(U, V)

def deviation(rotation):
    return numpy.linalg.norm(rotation - numpy.eye(len(rotation)), 2)

def temparray_like(arr):
    fs = NamedTemporaryFile()
    result = numpy.memmap(fs.name, dtype=arr.dtype, mode='w+',
        shape=arr.shape)
    fs.close()
    return result

if __name__ == '__main__':
    from scipy.io import loadmat, savemat
    import os
    directory = 'randrot'
    useed = 1000
    try:
        os.makedirs(directory)
    except:
        pass
    dims = 256
    seed = 1
    alpha = numpy.arange(0.1, 1.0 + 1e-15, 0.1)
    rots = randomRotationPowers(dims, alpha, seed=seed)
    for rot, a in zip(rots, alpha):
        fn = 'rotpow-%d-%.1f-%.2f-%d.mat' % (dims, a, deviation(rot), seed)
        print fn
        savemat(os.path.join(directory, fn), {'rotation': rot})
