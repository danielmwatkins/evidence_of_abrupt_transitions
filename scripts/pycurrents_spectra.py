"""
This file is a slight modification of PyCurrents spectra.py which can be
found at https://currents.soest.hawaii.edu/hgstage/pycurrents/
PyCurrents is distributed under the BSD 2-clause license, which is reproduced below.

I (Daniel Watkins) have made modifications to the file to make it possible
to calculate rotary spectra without installing the full PyCurrents package.
Any errors added by these modifications are mine alone.

PyCurrents license:

    With rare exceptions in which individual files contain alternative
    copyrights and/or licenses, this software and documentation is
    covered by the following BSD 2-clause license:
    (https://opensource.org/licenses/BSD-2-Clause)

    Copyright (c) 2016, Currents Group
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
    COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
    OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
    AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
    THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
    DAMAGE.
"""
import numpy as np
from numpy.fft import (fft, fftshift, fftfreq)

# the Bunch class is from PyCurrents/system/misc
# https://currents.soest.hawaii.edu/hgstage/pycurrents/file/tip/pycurrents/system/misc.py
class Bunch(dict):
    """
    A dictionary that also provides access via attributes.
    Additional methods update_values and update_None provide
    control over whether new keys are added to the dictionary
    when updating, and whether an attempt to add a new key is
    ignored or raises a KeyError.
    The Bunch also prints differently than a normal
    dictionary, using str() instead of repr() for its
    keys and values, and in key-sorted order.  The printing
    format can be customized by subclassing with a different
    str_ftm class attribute.  Do not assign directly to this
    class attribute, because that would substitute an instance
    attribute which would then become part of the Bunch, and
    would be reported as such by the keys() method.
    To output a string representation with
    a particular format, without subclassing, use the
    formatted() method.
    """

    str_fmt = "{0!s:<{klen}} : {1!s:>{vlen}}\n"

    def __init__(self, *args, **kwargs):
        """
        *args* can be dictionaries, bunches, or sequences of
        key,value tuples.  *kwargs* can be used to initialize
        or add key, value pairs.
        """
        dict.__init__(self)
        for arg in args:
            self.update(arg)
        self.update(kwargs)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError("'Bunch' object has no attribute '%s'" % name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        return self.formatted()

    def formatted(self, fmt=None, types=False):
        """
        Return a string with keys and/or values or types.
        *fmt* is a format string as used in the str.format() method.
        The str.format() method is called with key, value as positional
        arguments, and klen, vlen as kwargs.  The latter are the maxima
        of the string lengths for the keys and values, respectively,
        up to respective maxima of 20 and 40.
        """
        if fmt is None:
            fmt = self.str_fmt

        items = list(self.items())
        items.sort()

        klens = []
        vlens = []
        for i, (k, v) in enumerate(items):
            lenk = len(str(k))
            if types:
                v = type(v).__name__
            lenv = len(str(v))
            items[i] = (k, v)
            klens.append(lenk)
            vlens.append(lenv)

        klen = min(20, max(klens))
        vlen = min(40, max(vlens))
        slist = [fmt.format(k, v, klen=klen, vlen=vlen) for k, v in items]
        return ''.join(slist)

    def from_pyfile(self, filename):
        """
        Read in variables from a python code file.
        """
        # We can't simply exec the code directly, because in
        # Python 3 the scoping for list comprehensions would
        # lead to a NameError.  Wrapping the code in a function
        # fixes this.
        d = dict()
        lines = ["def _temp_func():\n"]
        with open(filename) as f:
            lines.extend(["    " + line for line in f])
        lines.extend(["\n    return(locals())\n",
                      "_temp_out = _temp_func()\n",
                      "del(_temp_func)\n"])
        codetext = "".join(lines)
        code = compile(codetext, filename, 'exec')
        exec(code, globals(), d)
        self.update(d["_temp_out"])
        return self

    def update_values(self, *args, **kw):
        """
        arguments are dictionary-like; if present, they act as
        additional sources of kwargs, with the actual kwargs
        taking precedence.
        One reserved optional kwarg is "strict".  If present and
        True, then any attempt to update with keys that are not
        already in the Bunch instance will raise a KeyError.
        """
        strict = kw.pop("strict", False)
        newkw = dict()
        for d in args:
            newkw.update(d)
        newkw.update(kw)
        self._check_strict(strict, newkw)
        dsub = dict([(k, v) for (k, v) in newkw.items() if k in self])
        self.update(dsub)

    def update_None(self, *args, **kw):
        """
        Similar to update_values, except that an existing value
        will be updated only if it is None.
        """
        strict = kw.pop("strict", False)
        newkw = dict()
        for d in args:
            newkw.update(d)
        newkw.update(kw)
        self._check_strict(strict, newkw)
        dsub = dict([(k, v) for (k, v) in newkw.items()
                                if k in self and self[k] is None])
        self.update(dsub)

    def _check_strict(self, strict, kw):
        if strict:
            bad = set(kw.keys()) - set(self.keys())
            if bad:
                bk = list(bad)
                bk.sort()
                ek = list(self.keys())
                ek.sort()
                raise KeyError(
                    "Update keys %s don't match existing keys %s" % (bk, ek))

def _can_factor(n, primes):
    for p in primes:
        while n % p == 0:
            n //= p
            if n == 1:
                return True
    return False


def good_nfft(n, maxfac=5):
    """
    Find a minimally-reduced number of points for efficient fft.

    The reduced number is also required to be even.

    Parameters
    ----------
    n : int
        Original number of points.
    maxfac : int, optional
        Maximum prime factor, <= 19; default is 5.

    Returns
    -------
    int
        Reduced number with maximum prime factor equal to maxfac.

    """
    primes20 = [2, 3, 5, 7, 11, 13, 17, 19]
    primes = [p for p in primes20 if p <= maxfac]
    n -= n%2
    while not _can_factor(n, primes):
        n -= 2
    return n


# Windows for the Welch method.  They are implemented directly
# here so we can see exactly what the implementation is, and
# ensure it has the desired property of being even so that
# the FT is real.
# This requires w(0) = w(N) = 0. (The end value is w(N-1).)

def _Tukey(n, r):
    x = np.linspace(0, 1, n, endpoint=False)
    ileft = x < r/2
    iright = x >= 1 - r/2
    weights = np.ones((n,), dtype=float)
    weights[ileft] = 0.5 * (1 + np.cos((2 * np.pi / r) * (x[ileft] - r/2)))
    weights[iright] = 0.5 * (1 + np.cos((2 * np.pi / r) *
                              (x[iright] - 1 + r/2)))
    return weights

def window_vals(n, name):
    _name = name
    name = name.lower()
    x = np.arange(n, dtype=float)
    if name == 'boxcar' or name == 'none':
        weights = np.ones((n,), dtype=float)
    elif name == 'triangle':
        weights = 1 - np.abs(x - 0.5 * n) / (0.5 * n)
    elif name == 'welch' or name == 'quadratic':
        weights = 1 - ((x - 0.5 * n) / (0.5 * n)) ** 2
    elif name == 'blackman':
        phi = 2 * np.pi * (x - n / 2) / n
        weights = 0.42 + 0.5 * np.cos(phi) + 0.08 * np.cos(2 * phi)
    elif name == 'hanning':
        phi = 2 * np.pi * x / n
        weights = 0.5 * (1 - np.cos(phi))
    elif name == 'cosine10':
        weights = _Tukey(n, 0.1)

    else:
        raise ValueError("name %s is not recognized" % _name)

    # Correct for floating point error for all except the boxcar.
    if weights[0] < 1:
        weights[0] = 0

    return weights

def detrend(x, method='linear', axis=-1):
    if method == 'none':
        return x
    if method == 'mean' or method == 'linear':
        try:
            xm = x.mean(axis=axis, keepdims=True)
        except TypeError:
            # older version of numpy
            sh = list(x.shape)
            sh[axis] = 1
            xm = x.mean(axis=axis).reshape(sh)
        y = x - xm
        if method == 'mean':
            return y
        t = np.linspace(-1, 1, x.shape[axis])
        t /= np.sqrt((t ** 2).sum())
        yy = y.swapaxes(axis, -1)
        a = np.dot(yy, t)
        if a.ndim > 0:
            a = a[..., np.newaxis]
        yydetrend = yy - a * t
        return np.ascontiguousarray(yydetrend.swapaxes(-1, axis))
    else:
        raise ValueError("method %s is not recognized" % method)

_detrend = detrend # alias to avoid being clobbered by kwarg name


def _welch_params(*args, **kw):
    kw = Bunch(kw)

    args = [np.array(x, copy=False) for x in np.broadcast_arrays(*args)]
    axis = kw.get('axis', -1)
    npts = args[0].shape[axis]

    noverlap = int(kw.overlap * kw.nfft)

    weights = window_vals(kw.nfft, kw.window)

    step = kw.nfft - noverlap
    ind = np.arange(0, npts - kw.nfft + 1, step)

    args += [weights, ind]
    return args

def _boundaries(freqs):
    freq_boundaries = np.zeros((len(freqs) + 1,), dtype=float)
    df = freqs[1] - freqs[0]
    freq_boundaries[1:] = freqs + 0.5 * df
    freq_boundaries[0] = freqs[0] - 0.5 * df
    return freq_boundaries

# _freqs is now used only by bispectrum, which has not been
# modified to cut off the end effects.
def _freqs(nfft, dt, is_complex):
    freqs = fftshift(fftfreq(nfft, dt))  # cycles per unit time used for dt
    if not is_complex:
        # start with first positive frequency
        freqs = freqs[nfft//2 + 1:]

    freq_boundaries = _boundaries(freqs)

    return freqs, freq_boundaries


def _convolve(a, b, axis=-1):
    return np.apply_along_axis(np.convolve, axis, a, b, mode='same')


def _slice_tuple(sl, axis, ndim):
    freqsel = [slice(None)] * ndim
    tup = freqsel[:]
    tup[axis] = sl
    return tuple(tup)


def spectrum(x, y=None, nfft=256, dt=1, detrend='linear',
               window='hanning', overlap=0.5, axis=-1,
               smooth=None):
    """
    Spectrum and optional cross-spectrum for N-D arrays.

    Rotary spectra are calculated if the inputs are complex.

    detrend can be 'linear' (default), 'mean', 'none', or a function.

    window can be a window function taking nfft as its sole argument,
    or the string name of a numpy window function (e.g. hanning)

    overlap is the fractional overlap, e.g. 0.5 for 50% (default)

    smooth is None or an odd integer. It can be used instead of,
    or in addition to, segment averaging.  To use it exclusively,
    set nfft=None.

    Returns a Bunch with spectrum, frequencies, etc.  The variables
    in the output depend on whether the input is real or complex,
    and on whether an autospectrum or a cross spectrum is being
    calculated.

    """

    if smooth is not None and smooth % 2 != 1:
        raise ValueError("smooth parameter must be None or an odd integer")

    if nfft is None:
        nfft = x.shape[axis]
        nfft -= nfft % 2  # If it was odd, chop off the last point.

    if nfft % 2:
        raise ValueError("nfft must be an even integer")

    # Raw frequencies, negative and positive:
    freqs = fftshift(fftfreq(nfft, dt))  # cycles per unit time used for dt

    if smooth is None:
        n_end = 0
    else:
        n_end = (smooth - 1) // 2  # to be chopped from each end

    kw = dict(nfft=nfft,
              detrend=detrend,
              window=window,
              overlap=overlap,
              axis=axis)
    if y is None:
        x, weights, seg_starts = _welch_params(x, **kw)
    else:
        x, y, weights, seg_starts = _welch_params(x, y, **kw)

    is_complex = (x.dtype.kind == 'c' or y is not None and y.dtype.kind == 'c')

    nsegs = len(seg_starts)

    segshape = list(x.shape)
    segshape[axis] = nfft
    ashape = tuple([nsegs] + segshape)
    fx_k = np.zeros(ashape, np.complex_)
    if y is not None:
        fy_k = fx_k.copy()

    # Make an indexing tuple for the weights.
    bcast = [np.newaxis] * x.ndim
    bcast[axis] = slice(None)
    bcast = tuple(bcast)

    segsel = [slice(None)] * x.ndim

    # Iterate over the segments.  (There might be only one.)
    for i, istart in enumerate(seg_starts):
        indslice = slice(istart, istart + nfft)
        segsel[axis] = indslice
        xseg = x[tuple(segsel)]
        xseg = weights[bcast] * _detrend(xseg, method=detrend, axis=axis)
        fx = fft(xseg, n=nfft, axis=axis)
        fx_k[i] = fx
        if y is not None:
            yseg = y[tuple(segsel)]
            yseg = weights[bcast] * _detrend(yseg, method=detrend, axis=axis)
            fy = fft(yseg, n=nfft, axis=axis)
            fy_k[i] = fy

    fxx = fftshift((np.abs(fx_k) ** 2).mean(axis=0), axes=[axis])
    if y is not None:
        fyy = fftshift((np.abs(fy_k) ** 2).mean(axis=0), axes=[axis])
        fxy = fftshift((np.conj(fx_k) * fy_k).mean(axis=0), axes=[axis])

    # Negative frequencies, excluding Nyquist:
    sl_cw = slice(1, nfft // 2)
    cwtup = _slice_tuple(sl_cw, axis=axis, ndim=x.ndim)

    # Positive frequencies, excluding 0:
    sl_ccw = slice(1 + nfft // 2, None)
    ccwtup = _slice_tuple(sl_ccw, axis=axis, ndim=x.ndim)

    if smooth is not None:
        # start with a boxcar; we can make it more flexible later
        smweights = np.ones((smooth,), dtype=float)
        smweights /= smweights.sum()
        fxx[cwtup] = _convolve(fxx[cwtup], smweights, axis=axis)
        fxx[ccwtup] = _convolve(fxx[ccwtup], smweights, axis=axis)
        if y is not None:
            fyy[cwtup] = _convolve(fyy[cwtup], smweights, axis=axis)
            fyy[ccwtup] = _convolve(fyy[ccwtup], smweights, axis=axis)
            fxy[cwtup] = _convolve(fxy[cwtup], smweights, axis=axis)
            fxy[ccwtup] = _convolve(fxy[ccwtup], smweights, axis=axis)
        # Note: end effects with this algorithm consist of a bias
        # towards zero.  We will adjust the
        # slices to select only the unbiased portions.

    psdnorm = (dt / (weights ** 2).sum())
    psd = fxx * psdnorm
    ps = fxx *  (1.0 / (weights.sum() ** 2))
    if y is not None:
        psd_x = psd
        psd_y = fyy * psdnorm
        psd_xy = fxy * psdnorm
        cohsq = np.abs(fxy) ** 2 / (fxx * fyy)
        phase = np.angle(fxy)

    if smooth is not None:
        ## Adjust the slices to avoid smoothing end effects:
        # Negative frequencies, excluding Nyquist:
        sl_cw = slice(1 + n_end, nfft // 2 - n_end)
        cwtup = _slice_tuple(sl_cw, axis=axis, ndim=x.ndim)

        # Positive frequencies, excluding 0:
        sl_ccw = slice(1 + nfft // 2 + n_end, -n_end)
        ccwtup = _slice_tuple(sl_ccw, axis=axis, ndim=x.ndim)

    out = Bunch(freqs=freqs[sl_ccw],
                freq_boundaries=_boundaries(freqs[sl_ccw]),
                seg_starts=seg_starts,
                smooth=smooth,
                nfft=nfft,
                detrend=detrend,
                window=window,
                overlap=overlap,
                axis=axis
                )

    if not is_complex:
        if y is None:
            out.psd = psd[ccwtup] * 2
            out.ps = ps[ccwtup] * 2
        else:
            out.psd_x = psd_x[ccwtup]
            out.psd_y = psd_y[ccwtup]
            out.psd_xy = psd_xy[ccwtup]
            out.cohsq = cohsq[ccwtup]
            out.phase = phase[ccwtup]

    else:
        out.cwfreqs = -freqs[sl_cw]
        out.cwfreq_boundaries = _boundaries(out.cwfreqs)
        out.ccwfreqs = freqs[sl_ccw]
        out.ccwfreq_boundaries = _boundaries(out.ccwfreqs)

        if y is None:
            out.cwpsd = psd[cwtup]
            out.cwps = ps[cwtup]
            out.ccwpsd = psd[ccwtup]
            out.ccwps = ps[ccwtup]
        else:
            out.cwpsd_x = psd_x[cwtup]
            out.cwpsd_y = psd_y[cwtup]
            out.cwpsd_xy = psd_xy[cwtup]
            out.cwcohsq = cohsq[cwtup]
            out.cwphase = phase[cwtup]

            out.ccwpsd_x = psd_x[ccwtup]
            out.ccwpsd_y = psd_y[ccwtup]
            out.ccwpsd_xy = psd_xy[ccwtup]
            out.ccwcohsq = cohsq[ccwtup]
            out.ccwphase = phase[ccwtup]

    return out


def bispectrum(x, nfft=256, dt=1, detrend='linear',
               window='hanning', overlap=0.5,
               norm='KimPowers',
               frange=None,
               masked=True,
               debug=False,):
    """
    Calculate the auto rotary bispectrum and bicoherence of a sequence.

    detrend can be 'linear' (default), 'mean', 'none', or a function.

    window can be a window function taking nfft as its sole argument,
    or the string name of a numpy window function (e.g. hanning)

    overlap is the fractional overlap, e.g. 0.5 for 50% (default)

    norm can be 'KimPowers' or 'standard'

    frange can be None, a scalar, or a pair of numbers

    Returns a Bunch with bispectra, frequencies, etc.

    """

    if debug:
        inspect = Bunch()

    x, weights, ind = _welch_params(x,
                                    nfft=nfft,
                                    detrend=detrend,
                                    window=window,
                                    overlap=overlap)
    nsegs = len(ind)

    freqs, freq_boundaries = _freqs(nfft, dt)
    if frange is None:
        fslice = slice(None)
        nfreq = nfft
    else:
        try:
            if len(frange) != 2:
                raise ValueError("frange must be a scalar or a pair of numbers")
        except TypeError:
            frange = (-frange, frange)
        ifmax = np.searchsorted(freqs, frange[1], side='right')
        ifmin = np.searchsorted(freqs, frange[0])
        fslice = slice(ifmin, ifmax)
        freqs = freqs[fslice]
        freq_boundaries = freq_boundaries[ifmin:ifmax + 1]
        nfreq = len(freqs)
        assert len(freq_boundaries) - len(freqs) == 1


    # Accumulators for the two parts of the bicoherence estimate denominator:
    fx_kl = np.zeros((nsegs, nfreq, nfreq), np.complex64)     # freqs k and l
    fx_klsum = np.zeros((nsegs, nfreq, nfreq), np.complex64)  # freq k + l

    # Accumulator for the ordinary spectrum:
    fx_k = np.zeros((nsegs, nfreq), np.complex64)

    # Index arrays:
    ii = np.arange(nfft, dtype=int)
    i_klsum = (ii[:, np.newaxis] + ii) % nfft
    # Modify i_klsum to match the shifted fft:
    i_klsum = (i_klsum + nfft // 2) % nfft

    i_klsum = i_klsum[fslice, fslice]


    for i in range(nsegs):
        xseg = x[ind[i]:ind[i]+nfft]
        xseg = weights * _detrend(xseg)
        fx = fftshift(fft(xseg, n=nfft))

        fx_k[i] = fx[fslice]
        fx_klsum[i] = np.conjugate(fx[i_klsum])
        fx_kl[i] = fx[fslice, np.newaxis] * fx[fslice]

    fx_sq = (np.abs(fx_k) ** 2).mean(axis=0)

    # Bispectrum:
    bispec_segs = fx_klsum * fx_kl
    bispec = bispec_segs.mean(axis=0)
    if debug:
        inspect.fx_klsum = fx_klsum
        inspect.fx_kl = fx_kl
        inspect.bispec_segs = bispec_segs
        inspect.fx_sq = fx_sq
    del bispec_segs

    biphase = - np.angle(bispec)    # Note sign reversal!

    d1 = np.mean(np.abs(fx_klsum) ** 2, axis=0)
    del fx_klsum

    if norm == 'KimPowers':
        d2 = np.mean(np.abs(fx_kl) ** 2, axis=0)
        del fx_kl
    elif norm == 'standard':
        d2 = fx_sq[:, np.newaxis] * fx_sq
    else:
        raise ValueError("norm must be 'standard' or 'KimPowers'")
    if debug:
        inspect.d1 = d1
        inspect.d2 = d2

    bicoh = np.abs(bispec) / np.sqrt(d1 * d2)

    # After fftshift, frequencies in cycles per record length, signed:
    cprl = ii.copy()
    cprl -= nfft // 2
    cprl = cprl[fslice]
    cprl_sum = cprl[:, np.newaxis] + cprl

    mask_upper = (ii[fslice, np.newaxis] - ii[fslice]) > 0

    # Mask region where sum of omegas >= Nyquist.
    mask_alias = (cprl_sum >= nfft // 2) | (cprl_sum <= -nfft // 2)

    # Mask region where either frequency is at the Nyquist.
    try:
        iny = np.nonzero(cprl == -nfft // 2)[0][0]
        mask_alias[iny,:] = True
        mask_alias[:,iny] = True
    except IndexError:
        pass

    mask = np.logical_or(mask_alias, mask_upper)

    # Mask regions where any frequency is zero:
    mask_zero = (cprl_sum == 0)
    try:
        izero = np.nonzero(cprl == 0)[0][0]
        mask_zero[:, izero] = True
        mask_zero[izero, :] = True
    except IndexError:
        pass

    mask = np.logical_or(mask, mask_zero)

    out = Bunch(freqs=freqs,
                 freq_boundaries=freq_boundaries,
                 biphase=biphase,
                 bicoh=bicoh,
                 bispec=bispec / (weights.sum() ** 3),
                 mask=mask,
                 isegstart=ind,
                 nfft=nfft,
                 detrend=detrend,
                 window=window,
                 overlap=overlap,
                 frange=frange,
                 fslice=fslice)
    if debug:
        out.inspect = inspect

    if masked:
        for key in ["biphase", "bicoh", "bispec"]:
            out[key] = np.ma.array(out[key], mask=mask, copy=False)

    return out


def example_data(npts, periods=(12, 12.42), amps=None, nl=0, dt=1):
    """
    periods are in time unit used for dt; negative periods give
    clockwise rotation, positive counterclockwise.

    amps must be None (default; unity) or a sequence the same
    length as pberiods

    nl is the nonlinearity coefficient

    Returns t, x where t is the ndarray of times, x the corresponding
    test data values (complex).

    """

    if amps is None:
        amps = np.ones((len(periods),))

    t = np.arange(npts, dtype=float) * dt
    x = np.zeros((npts,), dtype=np.complex128)
    for p, a in zip(periods, amps):
        x += a * np.exp((1j * 2 * np.pi / p) * t)

    x += nl * ((x / x.std()) ** 2)

    return t, x


def example_tide4(npts, amps=None, nl=0, dt=1):
    """
    Example data with 4 tidal constituents: m2, s2, k1, o1.

    """

    from pytide import hour_period_d
    t, y = example_data(npts, periods=[hour_period_d["m2"],
                                       hour_period_d["s2"],
                                       hour_period_d["k1"],
                                       hour_period_d["o1"]],
                                       amps=amps,
                                       nl=nl, dt=dt)
    return t, y

def add_random(x, amp=1, color='white', axis=-1):

    xr = np.random.randn(*x.shape)
    if x.dtype.kind == 'c':
        xr = xr + 1j * np.random.randn(*x.shape)
    if color == 'red':
        xr = np.cumsum(xr, axis=axis)
        xr = detrend(xr, method='mean', axis=axis)
    elif color != 'white':
        raise ValueError("color %s is not supported" % color)
    return x + amp * xr


