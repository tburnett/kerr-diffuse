# Likelihood Surface Quality Metric

This note documents the fit-quality quantity printed by `FermiFit.fit()` as `quality`.

## Definition

The metric is computed from the local gradient and Hessian of the log-likelihood surface:

\[
q = \frac{1}{4}\, g^T H^{-1} g
\]

where:

- `g` is the current gradient vector of the log-likelihood with respect to free parameters.
- `H` is the current Hessian matrix of the log-likelihood with respect to free parameters.

In code, this is implemented in `FitterSummaryMixin.delta_loglike()`.

## Where It Is Computed

Primary implementation:

- `like3/views.py` in `FitterSummaryMixin.delta_loglike()`

Core lines:

```python
gv = self.gradient()
H = np.array(self.hessian())
return np.dot(np.dot(gv, np.linalg.inv(H)), gv) / 4
```

If the calculation fails (for example, singular Hessian), the method returns `99.0` as a fallback.

## Where It Is Used In Fits

`FermiFit.fit()` uses this quantity in two ways:

- Printed as the `quality` value in the post-fit line:
  - `"%d calls: improvement, quality: ..."`
- Stored into fit diagnostics:
  - `self.fit_info['qual']`

It can also be used before fitting when `tolerance > 0` to skip an optimization that is estimated to provide only small improvement.

## Interpretation

This metric is a local quadratic estimate of "remaining improvable log-likelihood" near the current point.

Practical interpretation:

- `quality` near `0`: locally close to optimum.
- moderate positive `quality`: some remaining improvement likely.
- large `quality`: parameters are likely not near a local optimum.
- fallback value `99.0`: estimation failed (often numerical conditioning/Hessian issues).

Like all quadratic diagnostics, reliability depends on the local surface being well-approximated by a quadratic form.

## Relationship To Improvement

Do not confuse:

- `improvement` printed by `fit()`:
  - actual observed change in log-likelihood during the optimizer run.
- `quality`:
  - model-based local estimate from gradient and Hessian.

They are related but not identical, and may disagree on noisy or ill-conditioned surfaces.

## Notes

- The metric is not a p-value and is not a likelihood-ratio test statistic by itself.
- It is most useful as a convergence/conditioning diagnostic.
- If `quality` remains large after fitting, inspect parameter bounds, scaling, and covariance/Hessian conditioning.
