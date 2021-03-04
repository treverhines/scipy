"""Module for RBF interpolation"""
import warnings
from functools import lru_cache
from itertools import combinations_with_replacement

import numpy as np
import scipy.linalg
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.special import xlogy, binom


__all__ = ['RBFInterpolator', 'KNearestRBFInterpolator']


def _distance(x, y):
    """
    Returns a distance matrix between `x` and `y`

    Parameters
    ----------
    x : (..., n, d) ndarray
    y : (..., m, d) ndarray

    Returns
    -------
    (..., n, m) ndarray

    """
    if (x.ndim == 2) & (y.ndim == 2):
        # if possible, use the faster function
        return cdist(x, y)
    else:
        return np.linalg.norm(x[..., None, :] - y[..., None, :, :], axis=-1)


def _linear(r):
    """linear / 1st order polyharmonic spline"""
    return -r


def _tps(r):
    """thin plate spline / 2nd order polyharmonic spline"""
    return xlogy(r**2, r)


def _cubic(r):
    """cubic / 3rd order polyharmonic spline"""
    return r*r*r  # faster than r**3


def _quintic(r):
    """quintic / 5th order polyharmonic spline"""
    return -r*r*r*r*r  # faster than r**5


def _mq(r):
    """multiquadratic"""
    return -np.sqrt(r**2 + 1)


def _imq(r):
    """inverse multiquadratic"""
    return 1/np.sqrt(r**2 + 1)


def _iq(r):
    """inverse quadratic"""
    return 1/(r**2 + 1)


def _ga(r):
    """gaussian"""
    return np.exp(-r**2)


@lru_cache()
def _monomial_powers(ndim, degree):
    """
    Returns the powers for each monomial in a polynomial with the specified
    number of dimensions and degree
    """
    out = []
    for deg in range(degree + 1):
        for itm in combinations_with_replacement(np.eye(ndim, dtype=int), deg):
            out.append(sum(itm, np.zeros(ndim, dtype=int)))

    if not out:
        out = np.zeros((0, ndim), dtype=int)
    else:
        out = np.array(out)

    return out


def _vandermonde(x, degree):
    """
    Returns monomials evaluated at `x`. The monomials span the space of
    polynomials with the specified degree

    Parameters
    ----------
    x : (..., d) float array
    degree : int

    Returns
    -------
    (..., p) float array

    """
    pwr = _monomial_powers(x.shape[-1], degree)
    out = np.product(x[..., None, :]**pwr, axis=-1)
    return out


# For RBFs that are conditionally positive definite of order m, the interpolant
# should include polynomial terms with degree >= m - 1. Define the minimum
# degrees here. These values are from Chapter 8 of Fasshauer's "Meshfree
# Approximation Methods with MATLAB". The RBFs that are not in this dictionary
# are positive definite and do not need polynomial terms
_NAME_TO_MIN_DEGREE = {
    'mq': 0,
    'linear': 0,
    'tps': 1,
    'cubic': 1,
    'quintic': 2
    }


_NAME_TO_FUNC = {
    'linear': _linear,
    'tps': _tps,
    'cubic': _cubic,
    'quintic': _quintic,
    'mq': _mq,
    'imq': _imq,
    'iq': _iq,
    'ga': _ga
    }


# The shape parameter does not need to be specified when using these kernels
_SCALE_INVARIANT = {'linear', 'tps', 'cubic', 'quintic'}


def _gcv(d, s, K, P):
    """
    Generalized Cross Validation (GCV) score for RBF interpolation. The
    implementation follows eq. 1.3.23 and 4.3.1 of [1].

    Parameters
    ----------
    d : (N, D) ndarray
        Data values. The data are vector-valued for generality. The returned
        GCV score is the average of GCV scores for each component.
    s : (N,) ndarray
        Smoothing parameter for each data point
    K : (N, N) ndarray
        RBF matrix with smoothing parameters added to diagonals
    P : (N, M) ndarray
        Polynomial matrix

    Returns
    -------
    float

    References
    ----------
    .. [1] Wahba, G., 1990. Spline Models for Observational Data. SIAM.

    """
    n, m = P.shape
    if n == m:
        # if n == m then the interpolant will perfectly fit the data regardless
        # of the smoothing parameter, and the GCV score should evaluate to NaN.
        # Return NaN here because otherwise cho_solve and solve_triangular will
        # receive 0 sized input, which results in an error
        return np.nan

    Q, _ = np.linalg.qr(P, mode='complete')
    Q2 = Q[:, m:]
    K_proj = Q2.T.dot(K).dot(Q2)
    d_proj = Q2.T.dot(d)
    try:
        L, _ = scipy.linalg.cho_factor(K_proj, lower=True)
    except np.linalg.LinAlgError:
        # This error occurs when `K` is not numerically conditionally positive
        # definite, which is indicative of an ill-posed problem. We could try
        # to compute the GCV without using a Cholesky decomposition, but it is
        # not clear if that would produce results that are any more useful.
        return np.nan

    res = s[:, None]*(Q2.dot(scipy.linalg.cho_solve((L, True), d_proj)))
    # `mse` is the mean squared error between `d` and the interpolant. This is
    # in the numerator for the GCV expression in eq. 4.3.1
    mse = np.mean(np.linalg.norm(res, axis=0)**2/n)

    H = scipy.linalg.solve_triangular(L, Q2.T, lower=True)
    # `trc` is 1/n times the trace of the matrix mapping `d` to the residual
    # vector. This is in the denominator for the GCV expression in eq. 4.3.1.
    # This will be zero whenever the smoothing parameter is 0 (or n == m),
    # which will result in a NaN
    trc = np.mean(s*np.sum(H**2, axis=0))
    with np.errstate(divide='ignore', invalid='ignore'):
        out = mse / trc**2

    return out


def _gml(d, K, P):
    """
    Generalized Maximum Likelihood (GML) score for RBF interpolation. The
    implementation follows section 4.8 of [1]

    Parameters
    ----------
    d : (N, D) ndarray
        Data values. The data are vector-valued for generality. The returned
        GML score is the sum of GML scores for each component.
    K : (N, N) ndarray
        RBF matrix with smoothing added to diagonals
    P : (N, M) ndarray
        Polynomial matrix

    Returns
    -------
    float

    References
    ----------
    .. [1] Wahba, G., 1990. Spline Models for Observational Data. SIAM.

    """
    n, m = P.shape
    if n == m:
        # if n == m, the GML score will always be 0. Return 0 here because
        # otherwise solve_triangular will receive a 0 sized input and return an
        # error
        return 0.0

    Q, _ = np.linalg.qr(P, mode='complete')
    Q2 = Q[:, m:]
    K_proj = Q2.T.dot(K).dot(Q2)
    d_proj = Q2.T.dot(d)
    try:
        L, _ = scipy.linalg.cho_factor(K_proj, lower=True)
    except np.linalg.LinAlgError:
        return np.nan

    # compute and sum the Mahalanobis distance for each component of `d_proj`
    # using `K_proj` as the covariance matrix. This is in the numerator of the
    # GML expression.
    weighted_d_proj = scipy.linalg.solve_triangular(L, d_proj, lower=True)
    dist = np.linalg.norm(weighted_d_proj)**2
    # compute the determinant of `K_proj` using the diagonals of its Cholesky
    # decomposition. This is in the denominator of the GML expression.
    logdet = 2*np.sum(np.log(np.diag(L)))
    out = dist*np.exp(logdet/(n - m))
    return out


def _sanitize_init_args(y, d, smoothing, kernel, epsilon, degree, k):
    """
    Sanitize __init__ arguments for RBFInterpolator and KNearestRBFInterpolator
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 2:
        raise ValueError('Expected `y` to be a 2-dimensional array')

    ny, ndim = y.shape

    d = np.asarray(d)
    if d.shape[0] != ny:
        raise ValueError(
            'Expected the first axis of `d` to have length %d' % ny
            )

    if np.isscalar(smoothing):
        smoothing = np.full(ny, smoothing, dtype=float)
    else:
        smoothing = np.asarray(smoothing, dtype=float)
        if smoothing.shape != (ny,):
            raise ValueError(
                'Expected `smoothing` to be a scalar or have shape (%d,)' % ny
                )

    if callable(kernel):
        kernel_func = kernel
    elif kernel in _NAME_TO_FUNC:
        kernel_func = _NAME_TO_FUNC[kernel]
    else:
        raise ValueError(
            'Expected `kernel` to be callable or one of {%s}' %
            ', '.join('"%s"' % kn for kn in _NAME_TO_FUNC.keys())
            )

    if epsilon is None:
        if callable(kernel) | (kernel in _SCALE_INVARIANT):
            epsilon = 1.0
        else:
            raise ValueError(
                '`epsilon` must be specified if `kernel` is not callable or '
                'one of {%s}.' %
                ', '.join('"%s"' % kn for kn in _SCALE_INVARIANT)
                )

    elif not np.isscalar(epsilon):
        raise ValueError('Expected `epsilon` to be a scalar')

    min_degree = _NAME_TO_MIN_DEGREE.get(kernel, -1)
    if degree is None:
        degree = max(min_degree, 0)
    elif max(degree, -1) < min_degree:
        warnings.warn(
            'The polynomial degree should not be below %d for "%s". The '
            'interpolant may not be uniquely solvable, and the smoothing '
            'parameter may have an unintuitive effect.' %
            (min_degree, kernel),
            UserWarning
            )

    degree = int(degree)

    if k is None:
        nobs = ny
    else:
        # make sure the number of nearest neighbors used for interpolation does
        # not exceed the number of observations
        k = int(min(k, ny))
        nobs = k

    # The polynomial matrix must have full column rank in order for the
    # interpolant to be well-posed, which is not possible if there are fewer
    # observations than monomials
    nmonos = int(binom(degree + ndim, ndim))
    if nmonos > nobs:
        raise ValueError(
            'At least %d data points are required when the polynomial degree '
            'is %d and the number of dimensions is %d' % (nmonos, degree, ndim)
            )

    return y, d, smoothing, kernel_func, epsilon, degree, k


class RBFInterpolator:
    """
    Radial basis function (RBF) interpolation in N dimensions

    Parameters
    ----------
    y : (P, N) array_like
        Data point coordinates
    d : (P, ...) array_like
        Data values at `y`
    smoothing : float or (P,) array_like, optional
        Smoothing parameter. The interpolant perfectly fits the data when this
        is set to 0. For large values, the interpolant approaches a least
        squares fit of a polynomial with the specified degree.
    kernel : str or callable, optional
        Type of RBF. This should be one of:

            - 'linear'                       : ``-r``
            - 'tps' (thin plate spline)      : ``r**2 * log(r)``
            - 'cubic'                        : ``r**3``
            - 'quintic'                      : ``-r**5``
            - 'mq' (multiquadratic)          : ``-sqrt(1 + r**2)``
            - 'imq' (inverse multiquadratic) : ``1/sqrt(1 + r**2)``
            - 'iq' (inverse quadratic)       : ``1/(1 + r**2)``
            - 'ga' (Gaussian)                : ``exp(-r**2)``

        Alternatively, this can be a callable that takes an array of distances
        as input and returns an array with the same shape. The callable should
        be a positive definite or conditionally positive definite RBF.
    epsilon : float, optional
        Shape parameter that scales the input to the RBF. This can be ignored
        if `kernel` is 'linear', 'tps', 'cubic', or 'quintic' because it has
        the same effect as scaling the smoothing parameter. This must be
        specified if `kernel` is 'mq', 'imq', 'iq', or 'ga'.
    degree : int, optional
        Degree of the added polynomial. Some RBFs have a minimum polynomial
        degree that is needed for the interpolant to be well-posed. Those RBFs
        and their corresponding minimum degrees are:

            - 'mq'      : 0
            - 'linear'  : 0
            - 'tps'     : 1
            - 'cubic'   : 1
            - 'quintic' : 2

        The default value is the minimum degree required for `kernel` or 0 if
        there is no minimum required degree. Set this to -1 for no added
        polynomial.

    Notes
    -----
    An RBF is a scalar valued function in N-dimensional space whose value at
    :math:`x` can be expressed in terms of :math:`r=||x - c||`, where :math:`c`
    is the center of the RBF.

    An RBF interpolant for the vector of observations :math:`d`, which are made
    at the locations :math:`y`, is a linear combination of RBFs centered at
    :math:`y` plus a polynomial with a specified degree. The RBF interpolant is
    written as

    .. math::
        f(x) = K(x, y) a + P(x) b

    where :math:`K(x, y)` is a matrix of RBFs with centers at :math:`y`
    evaluated at the interpolation points :math:`x`, and :math:`P(x)` is a
    matrix of monomials, which span polynomials with the specified degree,
    evaluated at :math:`x`. The coefficients :math:`a` and :math:`b` are the
    solution to the linear equations

    .. math::
        (K(y, y) + \\lambda I) a + P(y) b = d

    and

    .. math::
        P(y)^T a = 0,

    where :math:`\\lambda` is a positive smoothing parameter that controls how
    well we want to fit the observations. The observations are fit exactly when
    the smoothing parameter is zero.

    For the RBFs 'ga', 'imq', and 'iq', the solution for :math:`a` and
    :math:`b` is analytically unique if :math:`P(y)` has full column rank. As
    an example, :math:`P(y)` would not have full column rank if the
    observations are collinear in two-dimensional space and the degree of the
    added polynomial is 1. For the RBFs 'mq', 'linear', 'tps', 'cubic', and
    'quintic', the solution for  :math:`a` and :math:`b` is analytically unique
    if :math:`P(y)` has full column rank and the degree of the added polynomial
    is not lower than the minimum value listed above (see Chapter 7 of [1]_ or
    [2]_).

    When using an RBF that is not scale invariant ('mq', 'imq', 'iq', and
    'ga'), an appropriate shape parameter must be chosen (e.g., through cross
    validation). Smaller values for the shape parameter correspond to wider
    RBFs. The problem can become ill-conditioned or singular when the shape
    parameter is too small.

    Examples
    --------
    Demonstrate interpolating scattered data to a grid in 2-D

    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import RBFInterpolator
    >>> np.random.seed(0)

    >>> xobs = np.random.uniform(-1, 1, (100, 2))
    >>> yobs = np.sum(xobs, axis=1)*np.exp(-6*np.sum(xobs**2, axis=1))

    >>> xgrid = np.mgrid[-1:1:50j, -1:1:50j]
    >>> xflat = xgrid.reshape(2, -1).T
    >>> yflat = RBFInterpolator(xobs, yobs)(xflat)
    >>> ygrid = yflat.reshape(50, 50)

    >>> plt.pcolormesh(*xgrid, ygrid, vmin=-0.25, vmax=0.25, shading='gouraud')
    >>> plt.scatter(*xobs.T, c=yobs, s=50, ec='k', vmin=-0.25, vmax=0.25)
    >>> plt.colorbar()
    >>> plt.show()

    See Also
    --------
    KNearestRBFInterpolator

    References
    ----------
    .. [1] Fasshauer, G., 2007. Meshfree Approximation Methods with Matlab.
        World Scientific Publishing Co.

    .. [2] http://amadeus.math.iit.edu/~fass/603_ch3.pdf

    .. [3] Wahba, G., 1990. Spline Models for Observational Data. SIAM.

    .. [4] http://pages.stat.wisc.edu/~wahba/stat860public/lect/lect8/lect8.pdf

    """
    @staticmethod
    def gcv(y, d,
            smoothing=0.0,
            kernel='tps',
            epsilon=None,
            degree=None):
        """
        Returns the Generalized Cross Validation (GCV) score for an
        `RBFInterpolator` instance created with the same arguments. The
        smoothing parameter, shape parameter, and/or kernel can be selected to
        minimize the GCV score, as suggested in [1].

        See `__init__` for a description of the arguments.

        The smoothing parameter must be non-zero in order for the GCV score to
        not be NaN.

        See Chapter 4 of [1] for more information on GCV.

        Returns
        -------
        float

        References
        ----------
        .. [1] Wahba, G., 1990. Spline Models for Observational Data. SIAM.

        """
        y, d, smoothing, kernel, epsilon, degree, _ = _sanitize_init_args(
            y, d, smoothing, kernel, epsilon, degree, None
            )

        ny = y.shape[0]
        d = d.reshape((ny, -1))

        yeps = y*epsilon
        Kyy = kernel(_distance(yeps, yeps))
        Kyy[range(ny), range(ny)] += smoothing

        center = y.mean(axis=0)
        scale = y.ptp(axis=0).max() if (ny > 1) else 1.0
        yhat = (y - center)/scale
        Py = _vandermonde(yhat, degree)

        return _gcv(d, smoothing, Kyy, Py)

    @staticmethod
    def gml(y, d,
            smoothing=0.0,
            kernel='tps',
            epsilon=None,
            degree=None):
        """
        Returns the Generalized Maximum Likelihood (GML) score for an
        `RBFInterpolator` instance created with the same arguments.

        See `__init__` for a description of the arguments.

        See Chapter 4 of [1] for more information on GML.

        Returns
        -------
        float

        References
        ----------
        .. [1] Wahba, G., 1990. Spline Models for Observational Data. SIAM.

        """
        y, d, smoothing, kernel, epsilon, degree, _ = _sanitize_init_args(
            y, d, smoothing, kernel, epsilon, degree, None
            )

        ny = y.shape[0]
        d = d.reshape((ny, -1))

        yeps = y*epsilon
        Kyy = kernel(_distance(yeps, yeps))
        Kyy[range(ny), range(ny)] += smoothing

        center = y.mean(axis=0)
        scale = y.ptp(axis=0).max() if (ny > 1) else 1.0
        yhat = (y - center)/scale
        Py = _vandermonde(yhat, degree)

        return _gml(d, Kyy, Py)

    def __init__(self, y, d,
                 smoothing=0.0,
                 kernel='tps',
                 epsilon=None,
                 degree=None):
        y, d, smoothing, kernel, epsilon, degree, _ = _sanitize_init_args(
            y, d, smoothing, kernel, epsilon, degree, None
            )

        ny = y.shape[0]
        data_shape = d.shape[1:]
        d = d.reshape((ny, -1))

        # Create the matrix of RBFs centered and evaluated at y, plus smoothing
        # on the diagonals
        yeps = y*epsilon
        Kyy = kernel(_distance(yeps, yeps))
        Kyy[range(ny), range(ny)] += smoothing

        # Create the matrix of monomials evaluated at y. Normalize the domain
        # to be within [-1, 1]
        center = y.mean(axis=0)
        scale = y.ptp(axis=0).max() if (ny > 1) else 1.0
        yhat = (y - center)/scale
        Py = _vandermonde(yhat, degree)

        nmonos = Py.shape[1]
        Z = np.zeros((nmonos, nmonos), dtype=float)

        LHS = np.block([[Kyy, Py], [Py.T, Z]])

        z = np.zeros((nmonos, d.shape[1]), dtype=float)
        rhs = np.concatenate((d, z), axis=0)

        coeff = np.linalg.solve(LHS, rhs)
        kernel_coeff, poly_coeff = coeff[:ny], coeff[ny:]

        self.y = y
        self.kernel = kernel
        self.epsilon = epsilon
        self.degree = degree
        self.center = center
        self.scale = scale
        self.kernel_coeff = kernel_coeff
        self.poly_coeff = poly_coeff
        self.data_shape = data_shape

    def __call__(self, x, chunk_size=1000):
        """
        Evaluates the interpolant at `x`

        Parameters
        ----------
        x : (Q, N) array_like
            Interpolation point coordinates
        chunk_size : int, optional
            Break `x` into chunks with this size and evaluate the interpolant
            for each chunk

        Returns
        -------
        (Q, ...) ndarray
            Values of the interpolant at `x`

        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError('Expected `x` to be a 2-dimensional array')

        if x.shape[1] != self.y.shape[1]:
            raise ValueError(
                'Expected the second axis of `x` to have length %d' %
                self.y.shape[1]
                )

        nx = x.shape[0]
        if chunk_size is not None:
            # kernel_coeff is complex if d was complex, otherwise it should be
            # a float
            dtype = self.kernel_coeff.dtype
            out = np.zeros((nx,) + self.data_shape, dtype=dtype)
            for start in range(0, nx, chunk_size):
                stop = start + chunk_size
                out[start:stop] = self(x[start:stop], chunk_size=None)

            return out

        xeps, yeps = x*self.epsilon, self.y*self.epsilon
        Kxy = self.kernel(_distance(xeps, yeps))

        xhat = (x - self.center)/self.scale
        Px = _vandermonde(xhat, self.degree)

        out = Kxy.dot(self.kernel_coeff) + Px.dot(self.poly_coeff)
        out = out.reshape((nx,) + self.data_shape)
        return out


class KNearestRBFInterpolator:
    """
    RBF interpolation using the k nearest neighbors

    Parameters
    ----------
    y : (P, N) array_like
        Data point coordinates
    d : (P, ...) array_like
        Data values at `y`
    k : int, optional
        Number of nearest neighbors to use for each interpolation point
    smoothing : float or (P,) array_like, optional
        Smoothing parameter. The interpolant perfectly fits the data when this
        is set to 0.
    kernel : str or callable, optional
        Type of RBF. This should be one of:

            - 'linear'                       : ``-r``
            - 'tps' (thin plate spline)      : ``r**2 * log(r)``
            - 'cubic'                        : ``r**3``
            - 'quintic'                      : ``-r**5``
            - 'mq' (multiquadratic)          : ``-sqrt(1 + r**2)``
            - 'imq' (inverse multiquadratic) : ``1/sqrt(1 + r**2)``
            - 'iq' (inverse quadratic)       : ``1/(1 + r**2)``
            - 'ga' (Gaussian)                : ``exp(-r**2)``

        Alternatively, this can be a callable that takes an array of distances
        as input and returns an array with the same shape. The callable should
        be a positive definite or conditionally positive definite RBF.
    epsilon : float, optional
        Shape parameter that scales the input to the RBF. This can be ignored
        if `kernel` is 'linear', 'tps', 'cubic', or 'quintic' because it has
        the same effect as scaling the smoothing parameter. This must be
        specified if `kernel` is 'mq', 'imq', 'iq', or 'ga'.
    degree : int, optional
        Degree of the added polynomial. Some RBFs have a minimum polynomial
        degree that is needed for the interpolant to be well-posed. Those RBFs
        and their corresponding minimum degrees are:

            - 'mq'      : 0
            - 'linear'  : 0
            - 'tps'     : 1
            - 'cubic'   : 1
            - 'quintic' : 2

        The default value is the minimum degree required for `kernel` or 0 if
        there is no minimum required degree. Set this to -1 for no added
        polynomial.

    See Also
    --------
    RBFInterpolator

    """
    def __init__(self, y, d,
                 smoothing=0.0,
                 k=50,
                 kernel='tps',
                 epsilon=None,
                 degree=None):
        y, d, smoothing, kernel, epsilon, degree, k = _sanitize_init_args(
            y, d, smoothing, kernel, epsilon, degree, k
            )

        data_shape = d.shape[1:]
        d = d.reshape((y.shape[0], -1))
        tree = cKDTree(y)

        self.y = y
        self.d = d
        self.smoothing = smoothing
        self.k = k
        self.kernel = kernel
        self.epsilon = epsilon
        self.degree = degree
        self.tree = tree
        self.data_shape = data_shape

    def __call__(self, x, chunk_size=1000):
        """
        Evaluates the interpolant at `x`

        Parameters
        ----------
        x : (Q, N) array_like
            Interpolation point coordinates
        chunk_size : int, optional
            Break `x` into chunks with this size and evaluate the interpolant
            for each chunk

        Returns
        -------
        (Q, ...) ndarray
            Values of the interpolant at `x`

        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError('Expected `x` to be a 2-dimensional array')

        if x.shape[1] != self.y.shape[1]:
            raise ValueError(
                'Expected the second axis of `x` to have length %d' %
                self.y.shape[1]
                )

        nx = x.shape[0]
        if chunk_size is not None:
            dtype = complex if np.iscomplexobj(self.d) else float
            out = np.zeros((nx,) + self.data_shape, dtype=dtype)
            for start in range(0, nx, chunk_size):
                stop = start + chunk_size
                out[start:stop] = self(x[start:stop], chunk_size=None)

            return out

        # get the indices of the k nearest observations for each interpolation
        # point
        _, nbr = self.tree.query(x, self.k)
        if self.k == 1:
            # cKDTree squeezes the output when k=1
            nbr = nbr[:, None]

        # multiple interpolation points may have the same neighborhood. Make
        # the neighborhoods unique so that we only compute the interpolation
        # coefficients once for each neighborhood
        nbr, inv = np.unique(np.sort(nbr, axis=1), return_inverse=True, axis=0)
        nnbr = nbr.shape[0]

        # Get the observation data for each neighborhood
        y, d, smoothing = self.y[nbr], self.d[nbr], self.smoothing[nbr]

        # build the left-hand-side interpolation matrix consisting of the RBF
        # and monomials evaluated at each neighborhood
        yeps = y*self.epsilon
        Kyy = self.kernel(_distance(yeps, yeps))
        Kyy[:, range(self.k), range(self.k)] += smoothing
        # Normalize each neighborhood to be within [-1, 1] for the monomials
        centers = y.mean(axis=1)
        if self.k > 1:
            scales = y.ptp(axis=1).max(axis=1)
        else:
            scales = np.ones((nnbr,), dtype=float)

        yhat = (y - centers[:, None])/scales[:, None, None]
        Py = _vandermonde(yhat, self.degree)
        PyT = np.transpose(Py, (0, 2, 1))
        nmonos = Py.shape[2]
        Z = np.zeros((nnbr, nmonos, nmonos), dtype=float)
        LHS = np.block([[Kyy, Py], [PyT, Z]])

        # build the right-hand-side data vector consisting of the observations
        # for each neighborhood and extra zeros
        z = np.zeros((nnbr, nmonos, d.shape[2]), dtype=float)
        rhs = np.concatenate((d, z), axis=1)

        # solve for the RBF and polynomial coefficients for each neighborhood
        coeff = np.linalg.solve(LHS, rhs)

        # expand the arrays from having one entry per neighborhood to one entry
        # per interpolation point
        coeff = coeff[inv]
        yeps = yeps[inv]
        centers = centers[inv]
        scales = scales[inv]

        # evaluate at the interpolation points
        xeps = x*self.epsilon
        Kxy = self.kernel(_distance(xeps[:, None], yeps))[:, 0]

        xhat = (x - centers)/scales[:, None]
        Px = _vandermonde(xhat, self.degree)

        kernel_coeff = coeff[:, :self.k]
        poly_coeff = coeff[:, self.k:]

        Kxy = Kxy[:, :, None]
        Px = Px[:, :, None]
        out = (Kxy*kernel_coeff).sum(axis=1) + (Px*poly_coeff).sum(axis=1)
        out = out.reshape((nx,) + self.data_shape)
        return out
