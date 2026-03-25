"""Quadratic form fitting and elliptical TS surface analysis.

This module provides utilities to:
- Fit quadratic forms to 2D data (QuadForm class).
- Convert quadratic coefficients to ellipse parameters (Ellipse class).
- Iterate source localization using ellipse fitting on TS surfaces (Localize class).

The workflow is useful for finding transient or weak source positions by fitting
a quadratic surface through TS values sampled on an octagon ring around a candidate.
"""


import numpy as np
from . skydir import SkyDir


def quadfun(r, p):
    """Evaluate a 2D quadratic form at point r.

    Parameters
    ----------
    r : tuple
        (x, y) point coordinates.
    p : array-like
        Quadratic coefficients as [a, b, c, d, e, f] where the quadratic is
        f(x,y) = a*x^2 + b*x + c*y^2 + d*y + e*x*y + f

    Returns
    -------
    float
        f(x, y) evaluated at the given point.
    """
    x, y = r
    return p[0]*x*x + p[1]*x + p[2]*y*y + p[3]*y + p[4]*x*y + p[5]

# Legacy flags (not currently used)
flip = True  # set to measure from N
#major = True # set to make semi-major axis first

class QuadForm:
    """Least-squares quadratic form fit to 2D data on a 9-point cross pattern.

    The class fits a 2D quadratic f(x,y) = a*x^2 + b*x + c*y^2 + d*y + e*x*y + f
    to data values at 9 canonical points: the origin plus 8 points on a unit circle.

    Notes
    -----
    Uses linear least-squares via matrix inversion. Class initialization precomputes
    the fit matrix (F) and its pseudoinverse (Q) for efficiency.
    """
    d = 1 / np.sqrt(2)
    # Points at the center, then ccw from x-axis on a ring of unit radius.
    points = [(0, 0), (1, 0), (d, d), (0, 1), (-d, d), (-1, 0), (-d, -d), (0, -1), (d, -d)]
    F = np.matrix([[x*x, x, y*y, y, x*y, 1] for x, y in points])
    Q = (F.T * F) ** -1
    Q[abs(Q) < 1e-9] = 0

    def __init__(self, u):
        """Initialize and fit the quadratic form.

        Parameters
        ----------
        u : array-like
            Values evaluated at the 9 canonical points. If length == 8,
            assumes values are only at the ring (not the center); a zero is
            prepended for the center.
        """
        if len(u) == 8:
            # Assume only the ring; differences from center, so add zero for center.
            self.u = np.asarray([0] + list(u))
        else:
            self.u = np.asarray(u)
        # Least-squares solve: p = (F.T F)^-1 F.T u
        self.p = np.array((self.u * QuadForm.F) * QuadForm.Q)[0]
        # Evaluate the fit function at the 9 points.
        self.v = np.array([self(r) for r in QuadForm.points])
        # Chi-squared goodness-of-fit.
        self.chisq = ((self.u - self.v) ** 2).sum()

    def __call__(self, r):
        """Evaluate the fitted quadratic form at point r.

        Parameters
        ----------
        r : tuple
            (x, y) coordinates.

        Returns
        -------
        float
            Fitted quadratic value at r.
        """
        return quadfun(r, self.p)

    def fun(self, f):
        """Apply a function to the 9 canonical points using the fitted coefficients.

        Parameters
        ----------
        f : callable
            Function with signature f((x, y), p) -> value.

        Returns
        -------
        array
            Values of f at each of the 9 points using self.p.
        """
        return np.array([f((x, y), self.p) for x, y in QuadForm.points])


class Ellipse:
    """Represent a 2D quadratic as an ellipse with standard parameters.

    Converts between quadratic coefficients and ellipse parameters:
    - Semi-major and semi-minor axes (a, b)
    - Rotation angle (phi)
    - Center offset (x0, y0)
    - Constant offset (k)
    """

    def __init__(self, q, convert=False, raw=True):
        """Initialize ellipse from data or coefficients.

        Parameters
        ----------
        q : array-like
            If raw=True: array of function values at 9 points, fitted via QuadForm.
            If raw=False: either quadratic coefficients or ellipse parameters.
        convert : bool
            If True and raw=False, input q is quadratic coefficients to convert.
            If False and raw=False, input q is already ellipse parameters.
        raw : bool
            If True, treat q as function values and fit QuadForm first.
            If False, treat q as already-fitted coefficients.
        """
        if raw:
            self.qf = QuadForm(q)
            self.chisq = self.qf.chisq
            self.q = self.convert(self.qf.p)
        else:
            self.q = q if not convert else self.convert(q)
            self.chisq = -1  # N/A when not fitted from data

    def convert(self, p):
        """Convert quadratic coefficients to ellipse parameters.

        Parameters
        ----------
        p : array-like
            Quadratic coefficients [a, b, c, d, e, f] where
            f(x,y) = a*x^2 + b*x + c*y^2 + d*y + e*x*y + f

        Returns
        -------
        list
            Ellipse parameters [a_ell, b_ell, phi, x0, y0, k] where:
            - a_ell, b_ell: semi-major and semi-minor axis lengths
            - phi: rotation angle in radians (-pi/2 to pi/2)
            - x0, y0: center offset
            - k: constant offset

        Raises
        ------
        Exception
            If the quadratic form is not an ellipse (degenerate case).
        """
        p = np.asarray(p, dtype='float64')  # int can mess up trigonometry
        # Check that the quadratic form defines an ellipse.
        if 4 * p[0] * p[2] <= p[4] ** 2:
            raise Exception('Poorly formed quadratic form (not an ellipse)')
        # Ensure positive orientation.
        if p[0] < 0 and p[2] < 0:
            p = -p
        # Compute eigenvalue-related quantities.
        X = p[2] - p[0]
        Y = p[4]
        if abs(X) < 1e-9:
            X = 0
        Z = p[0] + p[2]
        if abs(Y) < 1e-9:
            Y = 0
        D = np.sqrt(X * X + Y * Y)
        if X < 0:
            D = -D
        # Compute rotation angle.
        phi = np.pi / 4 if X == 0 else np.arctan(Y / X) / 2.
        phi = phi - np.pi / 2
        # Compute axis lengths.
        alpha, beta = abs(Z - D) / 2, abs(Z + D) / 2
        a = 1 / np.sqrt(alpha)
        b = 1 / np.sqrt(beta)
        # Ensure a is the semi-major axis; map phi to range (-90, 90) degrees.
        if b > a:
            a, b = b, a
            phi += np.pi / 2
        if phi >= np.pi / 2:
            phi -= np.pi
        if phi < -np.pi / 2:
            phi += np.pi

        # Compute center (x0, y0).
        det = 4 * p[0] * p[2] - p[4] ** 2
        x0 = (p[4] * p[3] - 2 * p[2] * p[1]) / det
        y0 = (p[4] * p[1] - 2 * p[0] * p[3]) / det
        # Compute constant term offset.
        s = -np.sin(phi)
        c = np.cos(phi)
        s, c = c, s
        K = x0**2 * ((c / a) ** 2 + (s / b) ** 2) \
            + y0**2 * ((c / b) ** 2 + (s / a) ** 2) \
            - 2 * c * s * x0 * y0 * (1 / a ** 2 - 1 / b ** 2)

        return [a, b, phi, x0, y0, p[5] - K]

    def __call__(self, r):
        """Evaluate the ellipse at point r (offset from center).

        Parameters
        ----------
        r : tuple
            (x, y) offset from center in rotated frame.

        Returns
        -------
        float
            Ellipse surface value at r.
        """
        x, y = r
        phi = self.q[2]
        s, c = np.sin(-self.q[2]), np.cos(self.q[2])
        s, c = c, s
        dx, dy = x - self.q[3], y - self.q[4]
        return ((c * dx - s * dy) / self.q[0]) ** 2 \
             + ((c * dy + s * dx) / self.q[1]) ** 2 \
             + self.q[5]

    def contour(self, r=1, count=50):
        """Return points tracing an ellipse contour at scaled radius r.

        Parameters
        ----------
        r : float
            Scaling factor (1.0 traces the unit ellipse).
        count : int
            Number of points to generate around the contour.

        Returns
        -------
        tuple[list, list]
            (x, y) coordinate lists for the contour.
        """
        s, c = np.sin(-self.q[2]), np.cos(self.q[2])
        a, b = self.q[0], self.q[1]
        x0, y0 = self.q[3], self.q[4]
        s, c = c, s
        x = []
        y = []
        for t in np.linspace(0, 2 * np.pi, count):
            ct, st = np.cos(t), np.sin(t)
            x.append(r * (a * ct * c - b * st * s) + x0)
            y.append(r * (a * ct * s + b * st * c) + y0)
        return x, y

    def draw(self, data=None, scale=2):
        """Plot the ellipse contour (legacy utility; requires matplotlib).

        Parameters
        ----------
        data : optional
            Not used; placeholder for compatibility.
        scale : float
            Axis scale for the plot.
        """
        import matplotlib.pyplot as plt
        x, y = self.contour()
        plt.plot(x, y, '-')
        plt.plot([self.q[3]], [self.q[4]], '+')
        plt.axis((-scale, scale, -scale, scale))
        plt.axvline(0, color='k')
        plt.axhline(0, color='k')
        plt.grid()

            
def testit(p=[1, 0, 2., 0, 0, 0]):
    """Test quadratic form fitting and ellipse conversion (utility function).

    Parameters
    ----------
    p : array-like
        Quadratic coefficients to test with.
    """
    print('testing with quad pars=', p)
    points = QuadForm.points
    u1 = np.asarray([quadfun(r, p) for r in points])  # generate data
    qf = QuadForm(u1)
    pfit = qf.p  # fit quadratic form coefficients
    check = ((p - pfit) ** 2).sum()
    print('fit chisq, check: %10.1g %10.1g' % (qf.chisq, check))
    if check > 1e-9:
        print('failed to fit: output parameters ', pfit)
    ell = Ellipse(pfit, True, raw=False)
    print(('elliptical pars: ' + 6 * '%10.3f') % tuple(ell.q))
    u2 = np.asarray([ell(r) for r in points])
    check = ((u1 - u2) ** 2).sum()
    print('compare two functions: %10.1g' % check)
    if check > 1e-10:
        print('Failed comparison!')
        print((5 * '%+10s') % ('x   ', 'y   ', 'quad ', 'eliptical', 'diff '))
        for i, (x, y) in enumerate(points):
            print((4 * '%10.3f' + '%10.1g') % (x, y, u1[i], u2[i], u1[i] - u2[i]))

class Localize:
    """Iterative source localization using elliptical TS surface fitting.

    The algorithm:
    1. Evaluate TS at 9 canonical points: the center plus 8 on an octagon ring.
    2. Fit a quadratic form to those TS values.
    3. Convert the quadratic to ellipse parameters.
    4. Shift the source position and uncertainty toward the ellipse center.
    5. Iterate until convergence.

    This method is robust for weak sources and provides local estimates of
    localization uncertainty and solution quality.
    """

    fit_radius = 2.5  # Ring radius in sigma units for TS sampling (modified from 2.0)

    def __init__(self, psl, verbose=True):
        """Initialize localization with a source and initial TS evaluation.

        Parameters
        ----------
        psl : object
            Point source-like object with methods:
            - dir() -> SkyDir: source position
            - errorCircle() -> float: initial position uncertainty (sigma)
            - TSmap(skydir) -> float: TS value at skydir
        verbose : bool
            If True, print iteration progress and diagnostics.
        """
        self.verbose = verbose
        self.psl = psl
        self.dir = psl.dir()
        self.ra, self.dec = self.dir.ra(), self.dir.dec()
        self.sigma = psl.errorCircle()
        self.qual_cache = -1
        if verbose:
            print(('initial: ra,dec, sigma:' + 3 * '%10.4f') % (self.ra, self.dec, self.sigma))

        try:
            self.fit(update=True)
        except:
            if self.verbose:
                print('update failed: center on highest TS and try again')
            self.recenter()
            self.fit(update=True)

    def recenter(self):
        """Find the TS peak on the ring and move the source center there.

        This is a fallback used when the initial ellipse fit fails, typically
        for sources with low TS at the initial position.
        """
        ts = np.array(self.ts)
        tsmax = ts.max()
        if np.isnan(tsmax):
            print('really lost')
            raise Exception('Localize: reallylost')
        idir = np.arange(9)[tsmax == ts][0]
        mdir = self.rcirc[idir]
        if self.verbose:
            print('try ra,dec,ts =', mdir.ra(), mdir.dec(), tsmax)
        self.ra = mdir.ra()
        self.dec = mdir.dec()

    def TS(self, sdir):
        """Evaluate TS at a sky direction.

        Parameters
        ----------
        sdir : SkyDir
            Sky direction.

        Returns
        -------
        float
            TS value at that direction.
        """
        return self.psl.TSmap(sdir)

    def fit(self, update=True):
        """Evaluate TS on the ring and fit an ellipse; optionally update position.

        Parameters
        ----------
        update : bool
            If True, shift position and uncertainty based on the ellipse fit.
            If False, only compute diagnostics.
        """
        verbose = self.verbose
        self.rcirc = self.circle()
        self.qual_cache = -1
        self.ts = [self.TS(r) for r in self.rcirc]
        if verbose:
            print(('ts:   ' + ' '.join(9 * ['%9.2f'])) % tuple(self.ts))
        self.ellipse = Ellipse(self.ts)
        self.chisq = self.ellipse.chisq
        if verbose:
            print(('resid:' + ' '.join(9 * ['%9.2f'])) % tuple(self.ts - self.ellipse.qf.v))
        if verbose:
            print(('fit:  ' + len(self.ellipse.q) * '%9.2f') % tuple(self.ellipse.q))
        if verbose:
            print('chisq: %9.2f' % self.ellipse.chisq)
        radius = Localize.fit_radius
        if update:
            # Shift position and uncertainty based on ellipse center offset and size.
            self.ra += self.ellipse.q[3] * self.sigma * radius
            self.dec += self.ellipse.q[4] * self.sigma * radius

            self.dir = SkyDir(self.ra, self.dec)
            self.sigma = np.sqrt(self.ellipse.q[0] * self.ellipse.q[1]) * self.sigma * radius
            if verbose:
                print(('update:  ra,dec, sigma:' + 3 * '%10.4f') % (self.ra, self.dec, self.sigma))

        # Compute final localization parameters for external use.
        self.par = [
            self.ra,
            self.dec,
            self.ts[0],
            radius * self.sigma * self.ellipse.q[0],  # semi-major uncertainty
            radius * self.sigma * self.ellipse.q[1],  # semi-minor uncertainty
            np.degrees(self.ellipse.q[2]),               # position angle
            self.quality(),
            self.ellipse.chisq,
        ]

    def circle(self):
        """Generate 9 SkyDir points for the TS evaluation ring.

        Returns
        -------
        list[SkyDir]
            The points: center, then 8 on an octagon around it.
        """
        d = 1 / np.sqrt(2.)
        points = [(0, 0), (1, 0), (d, d), (0, 1), (-d, d), (-1, 0), (-d, -d), (0, -1), (d, -d)]
        ddec = Localize.fit_radius * self.sigma
        dra = ddec / np.cos(np.radians(self.dec))
        return [SkyDir(self.ra + x * dra, self.dec + y * ddec) for x, y in points]

    def quality(self, radius=2.5):
        """Compute a quality metric for the fit.

        Evaluates residuals at 8 points on the fitted ellipse boundary,
        relative to the TS at the center. A smaller value indicates a better fit.

        Parameters
        ----------
        radius : float
            Distance (in sigma units) at which to evaluate quality points
            on the ellipse contour.

        Returns
        -------
        float
            RMS of TS residuals at the contour points, relative to center.
        """
        if self.qual_cache > 0:
            return self.qual_cache
        qf = self
        xp, yp = qf.ellipse.contour(qf.fit_radius, 8)  # Get points at standard radius.
        ddec = radius * qf.sigma
        dra = ddec / np.cos(np.radians(qf.dec))
        points = [SkyDir(qf.ra - x * dra, qf.dec + y * ddec) for x, y in zip(xp, yp)]

        tszero = qf.TS(SkyDir(qf.ra, qf.dec)) - radius ** 2
        ts = np.asarray([qf.TS(p) for p in points])  # Evaluate TS at the contour points.
        qual = np.sqrt(((ts - tszero) ** 2).sum())
        self.qual_cache = qual
        return qual


def tests():
    """Run a suite of QuadForm/Ellipse validation tests."""
    testit([1, 0, 2., 0, 1, 0])
    testit([2, 0, 1., 0, 1, 0])
    testit([1, 1, 1, 1, 1, 1])
    testit([9., 0, 9., 0, 0, 0])  # should give sigmas of 1/3


if __name__ == '__main__':
    tests()
