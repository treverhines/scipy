# cython: language_level=3
import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg.lapack import dgesv as py_dgesv

cimport numpy as cnp
from cython.view cimport array, contiguous
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
        return r**2*log(r)


cdef double _cubic(double r):
    """Cubic / 3rd order polyharmonic spline"""
    return r*r*r # faster than r**3


cdef double _quintic(double r):
    """Quintic / 5th order polyharmonic spline"""
    return -r*r*r*r*r # faster than r**5


cdef double _mq(double r):
    """Multiquadratic"""
    return -sqrt(r**2 + 1)


@cdivision(True)
cdef double _imq(double r):
    """Inverse multiquadratic"""
    return 1/sqrt(r**2 + 1)


@cdivision(True)
cdef double _iq(double r):
    """Inverse quadratic"""
    return 1/(r**2 + 1)


cdef double _ga(double r):
    """Gaussian"""
    return exp(-r**2)


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
                         double epsilon,
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
            value = kernel_func(value*epsilon)
            out[i, j] = value
            out[j, i] = value


@cdivision(True)
@boundscheck(False)
@wraparound(False)
cdef void _polynomial_matrix(double[:, ::1] x,
                             long[:, ::1] powers,
                             double[::1] shift,
                             double[::1] scale,
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
                value *= ((x[i, k] - shift[k])/scale[k])**powers[j, k]

            out[i, j] = value


@boundscheck(False)
@wraparound(False)
cdef void _kernel_vector(double[::1] x,
                         double[:, ::1] y,
                         kernel_func_type kernel_func,
                         double epsilon,
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
        value = kernel_func(value*epsilon)
        out[i] = value


@cdivision(True)
@boundscheck(False)
@wraparound(False)
cdef void _polynomial_vector(double[::1] x,
                             long[:, ::1] powers,
                             double[::1] shift,
                             double[::1] scale,
                             double[::1] out):
    cdef:
        int i, j
        int m = powers.shape[0]
        int n = powers.shape[1]
        double value

    for i in range(m):
        value = 1.0
        for j in range(n):
            value *= ((x[j] - shift[j])/scale[j])**powers[i, j]

        out[i] = value


@boundscheck(False)
@wraparound(False)
cdef double[:, ::1] _shift_and_scale(double[:, ::1] x):
    '''
    Returns values to shift and scale x by so that it spans a unit hypercube
    '''
    cdef:
        int i, j
        int m = x.shape[0]
        int n = x.shape[1]
        double min_value, max_value
        # store the shift in the first row of the output array and the scale in
        # the second row
        double[:, ::1] out = array(
            shape=(2, n),
            itemsize=sizeof(double),
            format='d'
            )

    for j in range(n):
        # assuming x has been sanitized and x.shape[0] != 0
        min_value = x[0, j]
        max_value = x[0, j]
        for i in range(1, m):
            min_value = fmin(x[i, j], min_value)
            max_value = fmax(x[i, j], max_value)

        out[0, j] = (max_value + min_value)/2
        if min_value == max_value:
            # if there is a single point in x or all the points in x have the
            # same value for dimension j, then scale dimension j by one rather
            # than dividing by zero.
            out[1, j] = 1.0
        else:
            out[1, j] = (max_value - min_value)/2

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
            raise ValueError('The %d-th argument had an illegal value' % -info)
        else:
            raise LinAlgError('Singular matrix')


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


@boundscheck(False)
@wraparound(False)
def _build_and_solve(double[:, ::1] y,
                     double[:, ::1] d,
                     double[::1] smoothing,
                     unicode kernel,
                     double epsilon,
                     long[:, ::1] powers):
    cdef:
        kernel_func_type kernel_func = _kernel_name_to_func(kernel)
        int i
        int ny = y.shape[0]
        int nmonos = powers.shape[0]
        double[:, ::1] shift_and_scale = _shift_and_scale(y)
        double[::1] shift = shift_and_scale[0]
        double[::1] scale = shift_and_scale[1]
        double[::1, :] lhs = array(
            shape=(ny + nmonos, ny + nmonos),
            itemsize=sizeof(double),
            format='d',
            mode='fortran'
            )
        double[::1, :] rhs = array(
            shape=(ny + nmonos, d.shape[1]),
            itemsize=sizeof(double),
            format='d',
            mode='fortran'
            )

    _kernel_matrix(y, kernel_func, epsilon, lhs[:ny, :ny])
    _polynomial_matrix(y, powers, shift, scale, lhs[:ny, ny:])
    lhs[ny:, :ny] = lhs[:ny, ny:].T
    lhs[ny:, ny:] = 0.0
    for i in range(ny):
        lhs[i, i] += smoothing[i]

    rhs[:ny] = d
    rhs[ny:] = 0.0

    # solve as a generic system, the solution is written to rhs
    _dgesv(lhs, rhs)

    return np.asarray(rhs), np.asarray(shift), np.asarray(scale)


@boundscheck(False)
@wraparound(False)
def _evaluate(double[:, ::1] x,
              double[:, ::1] y,
              unicode kernel,
              double epsilon,
              long[:, ::1] powers,
              double[::1, :] coeffs,
              double[::1] shift,
              double[::1] scale):
    cdef:
        kernel_func_type kernel_func = _kernel_name_to_func(kernel)
        int i
        int nx = x.shape[0]
        int ny = y.shape[0]
        int nmonos = powers.shape[0]
        double[:, ::1] out = array(
            shape=(nx, coeffs.shape[1]),
            itemsize=sizeof(double),
            format='d',
            )
        double[::1] vec = array(
            shape=(ny + nmonos,),
            itemsize=sizeof(double),
            format='d',
            )

    for i in range(nx):
        _kernel_vector(x[i], y, kernel_func, epsilon, vec[:ny])
        _polynomial_vector(x[i], powers, shift, scale, vec[ny:])
        _dgemv(b'T', coeffs, vec, out[i])

    return np.asarray(out)
