# cython: language_level=3
import numpy as np

from cython.view cimport array
from cython cimport boundscheck, wraparound, cdivision
from scipy.linalg.cython_lapack cimport dgesv
from scipy.linalg.cython_blas cimport dgemv
from libc.math cimport sqrt, exp, log, fmin, fmax


cdef double _linear(double r):
    """Linear / 1st order polyharmonic spline"""
    return -r


cdef double _tps(double r):
    """Thin plate spline / 2nd order polyharmonic spline"""
    if r == 0.0:
        return 0.0
    else:
        return r*r*log(r)


cdef double _cubic(double r):
    """Cubic / 3rd order polyharmonic spline"""
    return r*r*r


cdef double _quintic(double r):
    """Quintic / 5th order polyharmonic spline"""
    return -r*r*r*r*r


cdef double _mq(double r):
    """Multiquadratic"""
    return -sqrt(r*r + 1)


@cdivision(True)
cdef double _imq(double r):
    """Inverse multiquadratic"""
    return 1/sqrt(r*r + 1)


@cdivision(True)
cdef double _iq(double r):
    """Inverse quadratic"""
    return 1/(r*r + 1)


cdef double _ga(double r):
    """Gaussian"""
    return exp(-r*r)


# define a type for the RBF kernel functions, which take and return doubles
ctypedef double (*kernel_func_type)(double)


cdef kernel_func_type _kernel_name_to_func(unicode kernel) except *:
    '''returns the kernel function corresponding to the string'''
    if kernel == 'linear':
        return _linear
    elif kernel == 'tps':
        return _tps
    elif kernel == 'cubic':
        return _cubic
    elif kernel == 'quintic':
        return _quintic
    elif kernel == 'mq':
        return _mq
    elif kernel == 'imq':
        return _imq
    elif kernel == 'iq':
        return _iq
    elif kernel == 'ga':
        return _ga
    else:
        raise ValueError("Invalid kernel name: '%s'" % kernel)


@boundscheck(False)
@wraparound(False)
cdef void _kernel_matrix(double[:, ::1] x,
                         kernel_func_type kernel_func,
                         double[:, :] out):
    cdef:
        int i, j, k
        int m = x.shape[0]
        int n = x.shape[1]
        double value

    for i in range(m):
        for j in range(i+1):
            value = 0.0
            for k in range(n):
                value += (x[i, k] - x[j, k])**2

            value = sqrt(value)
            value = kernel_func(value)
            out[i, j] = value
            out[j, i] = value


@boundscheck(False)
@wraparound(False)
cdef void _polynomial_matrix(double[:, ::1] x,
                             long[:, ::1] powers,
                             double[:, :] out):
    cdef:
        int i, j, k
        int m = x.shape[0]
        int n = x.shape[1]
        int p = powers.shape[0]
        double value

    for i in range(m):
        for j in range(p):
            value = 1.0
            for k in range(n):
                value *= x[i, k]**powers[j, k]

            out[i, j] = value


@boundscheck(False)
@wraparound(False)
cdef void _kernel_vector(double[::1] x,
                         double[:, ::1] y,
                         kernel_func_type kernel_func,
                         double[::1] out):
    cdef:
        int i, j
        int m = y.shape[0]
        int n = y.shape[1]
        double value

    for i in range(m):
        value = 0.0
        for j in range(n):
            value += (x[j] - y[i, j])**2

        value = sqrt(value)
        value = kernel_func(value)
        out[i] = value


@boundscheck(False)
@wraparound(False)
cdef void _polynomial_vector(double[::1] x,
                             long[:, ::1] powers,
                             double[::1] out):
    cdef:
        int i, j
        int m = powers.shape[0]
        int n = powers.shape[1]
        double value

    for i in range(m):
        value = 1.0
        for j in range(n):
            value *= x[j]**powers[i, j]

        out[i] = value


@cdivision(True)
@boundscheck(False)
@wraparound(False)
cdef double[::1] _shift(double[:, ::1] x):
    '''
    Returns a vector to shift `x` by to improve numerical stability. This is
    equivalent to `np.mean(x, axis=0)`
    '''
    cdef:
        int i, j
        int m = x.shape[0]
        int n = x.shape[1]
        double value
        double[::1] out = array(
            shape=(n,),
            itemsize=sizeof(double),
            format='d'
            )

    for j in range(n):
        value = 0.0
        for i in range(m):
            value += x[i, j]

        value /= m
        out[j] = value

    return out


@boundscheck(False)
@wraparound(False)
cdef double _scale(double[:, ::1] x):
    '''
    Returns a value to scale `x` by to improve numerical stability. In most
    cases, this is equivalent to `np.ptp(x, axis=0).max()`
    '''
    cdef:
        int i, j
        int m = x.shape[0]
        int n = x.shape[1]
        double min_value, max_value
        double out = 0.0

    for j in range(n):
        # assuming x has been sanitized and x.shape[0] != 0
        min_value = x[0, j]
        max_value = x[0, j]
        for i in range(1, m):
            min_value = fmin(x[i, j], min_value)
            max_value = fmax(x[i, j], max_value)

        out = fmax(max_value - min_value, out)

    # If there is a single point in `x` or all the points are equal, then `out`
    # will be 0. In that case, return 1 to avoid division by 0
    if out == 0.0:
        return 1.0
    else:
        return out


@boundscheck(False)
@wraparound(False)
cdef void _dgesv(double[::1, :] A, double[::1, :] B) except *:
    '''dgesv for fortran contiguous memoryviews'''
    cdef:
        int n = A.shape[0]
        int nrhs = B.shape[1]
        int[::1] ipiv = array(
            shape=(n,),
            itemsize=sizeof(int),
            format='i',
            )
        int info

    dgesv(&n, &nrhs, &A[0, 0], &n, &ipiv[0], &B[0, 0], &n, &info)
    if info != 0:
        if info < 0:
            raise ValueError(
                'The %d-th argument had an illegal value.' % abs(info)
                )
        else:
            raise ValueError(
                'U(%d,%d) is exactly zero. The factorization has been '
                'completed, but the factor U is exactly singular, so the '
                'solution could not be computed.' % (info, info)
                )


@boundscheck(False)
@wraparound(False)
cdef void _dgemv(char trans,
                 double[::1, :] A,
                 double[::1] x,
                 double[::1] y):
    '''dgemv for fortran contiguous memoryviews'''
    cdef:
        int m = A.shape[0]
        int n = A.shape[1]
        double alpha = 1.0
        double beta = 0.0
        int inc = 1

    dgemv(
        &trans, &m, &n, &alpha, &A[0, 0], &m, &x[0], &inc, &beta, &y[0], &inc
        )


cdef class _RBFInterpolator:
    cdef:
        double[:, ::1] yeps
        double epsilon
        double[::1] shift
        double scale
        long[:, ::1] powers
        double[::1, :] coeffs
        int ny, ndim, ddim, nmonos
        unicode kernel

    @cdivision(True)
    @boundscheck(False)
    @wraparound(False)
    def __init__(self,
                 double[:, ::1] y,
                 double[:, ::1] d,
                 double[::1] smoothing,
                 unicode kernel,
                 double epsilon,
                 long[:, ::1] powers):
        cdef:
            kernel_func_type kernel_func = _kernel_name_to_func(kernel)
            int i, j
            int ny = y.shape[0]
            int ndim = y.shape[1]
            int ddim = d.shape[1]
            int nmonos = powers.shape[0]
            double[::1] shift = _shift(y)
            double scale = _scale(y)
            double[:, ::1] yeps = array(
                shape=(ny, ndim),
                itemsize=sizeof(double),
                format='d',
                )
            double[:, ::1] yhat = array(
                shape=(ny, ndim),
                itemsize=sizeof(double),
                format='d',
                )
            double[::1, :] lhs = array(
                shape=(ny + nmonos, ny + nmonos),
                itemsize=sizeof(double),
                format='d',
                mode='fortran'
                )
            double[::1, :] rhs = array(
                shape=(ny + nmonos, ddim),
                itemsize=sizeof(double),
                format='d',
                mode='fortran'
                )

        for i in range(ny):
            for j in range(ndim):
                yeps[i, j] = y[i, j]*epsilon
                yhat[i, j] = (y[i, j] - shift[j])/scale

        _kernel_matrix(yeps, kernel_func, lhs[:ny, :ny])
        _polynomial_matrix(yhat, powers, lhs[:ny, ny:])
        lhs[ny:, :ny] = lhs[:ny, ny:].T
        lhs[ny:, ny:] = 0.0
        for i in range(ny):
            lhs[i, i] += smoothing[i]

        rhs[:ny] = d
        rhs[ny:] = 0.0

        # solve as a generic system, the solution is written to rhs
        _dgesv(lhs, rhs)

        self.yeps = yeps
        self.epsilon = epsilon
        self.shift = shift
        self.scale = scale
        self.powers = powers
        self.coeffs = rhs
        self.ny = ny
        self.ndim = ndim
        self.ddim = ddim
        self.nmonos = nmonos
        self.kernel = kernel

    @cdivision(True)
    @boundscheck(False)
    @wraparound(False)
    def __call__(self, double[:, ::1] x):
        cdef:
            kernel_func_type kernel_func = _kernel_name_to_func(self.kernel)
            int i, j
            int nx = x.shape[0]
            double[:, ::1] out = array(
                shape=(nx, self.ddim),
                itemsize=sizeof(double),
                format='d',
                )
            double[::1] xeps = array(
                shape=(self.ndim,),
                itemsize=sizeof(double),
                format='d'
                )
            double[::1] xhat = array(
                shape=(self.ndim,),
                itemsize=sizeof(double),
                format='d'
                )
            double[::1] vec = array(
                shape=(self.ny + self.nmonos,),
                itemsize=sizeof(double),
                format='d',
                )

        for i in range(nx):
            for j in range(self.ndim):
                xeps[j] = x[i, j]*self.epsilon
                xhat[j] = (x[i, j] - self.shift[j])/self.scale

            _kernel_vector(xeps, self.yeps, kernel_func, vec[:self.ny])
            _polynomial_vector(xhat, self.powers, vec[self.ny:])
            _dgemv(b'T', self.coeffs, vec, out[i])

        return np.array(out)
