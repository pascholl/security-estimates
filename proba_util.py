from math import factorial as fac
from math import log, ceil, erf, sqrt
from collections import defaultdict


def standard_deviation(D):
    return sqrt(sum(D[i]*i**2 for i in D))

def gaussian_center_weight(sigma, t):
    """ Weight of the gaussian of std deviation s, on the interval [-t, t]
    :param x: (float)
    :param y: (float)
    :returns: erf( t / (sigma*\sqrt 2) )
    """
    return erf(t / (sigma * sqrt(2.)))

def binomial(x, y):
    """ Binomial coefficient
    :param x: (integer)
    :param y: (integer)
    :returns: y choose x
    """
    try:
        binom = fac(x) // fac(y) // fac(x - y)
    except ValueError:
        binom = 0
    return binom

def centered_binomial_pdf(k, x):
    """ Probability density function of the centered binomial law of param k at x
    :param k: (integer)
    :param x: (integer)
    :returns: p_k(x)
    """
    return binomial(2*k, x+k) / 2.**(2*k)

def build_rounded_gaussian_law(sigma, tailcut=None, tailcut_shift=None):
    """ Rounded gaussian centered around 0 with s.d. sigma and tails cut at +/- tailcut*sigma """
    #D = defaultdict(lambda : 2**(-300))
    D = {}
    if tailcut is None:
        tailcut = int(round(6*sigma))
    else:
        tailcut = int(round(tailcut))
    if tailcut_shift is None:
        tailcut_left = -tailcut
        tailcut_right = tailcut
    else:
        tailcut_left = -tailcut + int(round(tailcut_shift))
        tailcut_right = tailcut + int(round(tailcut_shift))
    #print('cutting tails, left: %s, right: %s' % (tailcut_left, tailcut_right))
    # scale to ensure sums to 1 after cutting tails
    scaling_factor = (erf((tailcut_right + 0.5)/(sigma*sqrt(2.))) - erf((tailcut_left - 0.5)/(sigma*sqrt(2.))))/2

    for i in range(tailcut_left, tailcut_right+1):
        # prob. continuous Gaussian is in [i-0.5, i+0.5]
        D[i] = (erf((i + 0.5)/(sigma*sqrt(2.))) - erf((i - 0.5)/(sigma*sqrt(2.))))/2 / scaling_factor
    return D

def build_centered_binomial_law(k):
    """ Construct the binomial law as a dictionary
    :param k: (integer)
    :param x: (integer)
    :returns: A dictionary {x:p_k(x) for x in {-k..k}}
    """
    D = {}
    for i in range(-k, k+1):
        D[i] = centered_binomial_pdf(k, i)
    return D

def binomial_from_sigma(sigma):
    k = int(round(2*sigma**2))
    return build_centered_binomial_law(k)

def build_centered_uniform_law(k):
    """ Uniform in range {-k, ..., k} """
    D = {}
    for i in range(-k, k+1):
        D[i] = 1/(2*k+1)
    return D

def build_uniform_law(k):
    """ Uniform in range {0, ..., k} """
    D = {}
    for i in range(k+1):
        D[i] = 1/(k+1)
    return D

def sum_of_uniforms(k, b):
    """ Generalized centered binomial: sum of k terms (x_i - y_i) where x_i, y_i are uniform in [0, b]

    Variance of Unif(0,b) is ((b+1)**2 - 1)/12
    => Variance of sum_of_uniforms(k, b) is (k * b * (b + 2))/6
    """
    x = build_uniform_law(b)
    y = law_add_constant(x, -b)
    xsum = iter_law_convolution(x, k)
    ysum = iter_law_convolution(y, k)
    return law_convolution(xsum, ysum)

def sum_of_uniforms_from_sigma(sigma, b=255):
    var = sigma**2
    k = int(round(6*var / (b*(b+2))))
    print('k=%s' % k)
    return sum_of_uniforms(k, b)

def mod_switch(x, q, rq):
    """ Modulus switching (rounding to a different discretization of the Torus)
    :param x: value to round (integer)
    :param q: input modulus (integer)
    :param rq: output modulus (integer)
    """
    return int(round(1.* rq * x / q)) % rq


def mod_centered(x, q):
    """ reduction mod q, centered (ie represented in -q/2 .. q/2)
    :param x: value to round (integer)
    :param q: input modulus (integer)
    """
    a = x % q
    if a < q/2:
        return a
    return a - q


def build_mod_switching_error_law(q, rq):
    """ Construct Error law: law of the difference introduced by switching from and back a uniform value mod q
    :param q: original modulus (integer)
    :param rq: intermediate modulus (integer)
    """
    D = {}
    V = {}
    for x in range(q):
        y = mod_switch(x, q, rq)
        z = mod_switch(y, rq, q)
        d = mod_centered(x - z, q)
        D[d] = D.get(d, 0) + 1./q
        V[y] = V.get(y, 0) + 1

    return D

def law_add_constant(A, c):
    """ Distribution of A + c for constant c """
    if type(A) is defaultdict:
        B = defaultdict(A.default_factory)
    else:
        B = {}
    for a in A:
        B[a+c] = A[a]
    return B

def law_convolution(A, B):
    """ Construct the convolution of two laws (sum of independent variables from two input laws)
    :param A: first input law (dictionary)
    :param B: second input law (dictionary)
    """

    C = {}
    for a in A:
        for b in B:
            c = a+b
            C[c] = C.get(c, 0) + A[a] * B[b]
    return C


def law_product(A, B):
    """ Construct the law of the product of independent variables from two input laws
    :param A: first input law (dictionary)
    :param B: second input law (dictionary)
    """
    C = {}
    for a in A:
        for b in B:
            c = a*b
            C[c] = C.get(c, 0) + A[a] * B[b]
    return C


def clean_dist(A):
    """ Clean a distribution to accelerate further computation (drop element of the support with proba less than 2^-300)
    :param A: input law (dictionary)
    """
    B = {}
    for (x, y) in A.items():
        if y>2**(-300):
            B[x] = y
    return B


def iter_law_convolution(A, i):
    """ compute the -ith fold convolution of a distribution (using double-and-add)
    :param A: first input law (dictionary)
    :param i: (integer)
    """
    D = {0: 1.0}
    i_bin = bin(i)[2:]  # binary representation of n
    for ch in i_bin:
        D = law_convolution(D, D)
        D = clean_dist(D)
        if ch == '1':
            D = law_convolution(D, A)
            D = clean_dist(D)
    return D


def tail_probability(D, t):
    '''
    Probability that an element drawn from D is strictly greater than t in absolute value
    :param D: Law (Dictionary)
    :param t: tail parameter (integer)
    '''
    s = 0
    ma = max(D.keys())
    if t >= ma:
        return 0
    for i in reversed(range(int(ceil(t)), ma)):  # Summing in reverse for better numerical precision (assuming tails are decreasing)
        s += D.get(i, 0) + D.get(-i, 0)
    return s
