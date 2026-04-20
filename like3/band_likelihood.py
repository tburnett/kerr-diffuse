from like3 import likelihood as likelihood_mod; reload(likelihood_mod)
Likelihood = likelihood_mod.Likelihood


class BandModel:
    """Adapter exposing the Likelihood interface for a single Band instance.

    Parameters
    ----------
    band : Band
        A Band with `data`, `diffuse`, `pixel_counts`, and `pixel_gradient` populated.

    Notes
    -----
    The active pixel set is the intersection of the data pixels and the pixels
    illuminated by `Band.pixel_counts()` (called without arguments).  Only these
    pixels appear in `counts()`, `count_gradient()`, and `self.data`.
    """
    def __init__(self, band):
        self.band = band
        self.source_model = band.source_model
        self.parameters = self.source_model.parameters

        # Pixels illuminated by the source model
        illum_pix, _ = band.pixel_counts()

        # Restrict to the intersection with band.data pixels
        data_pix, data_counts = band.data
        mask = np.isin(data_pix, illum_pix)
        self._pix = data_pix[mask]
        self.data = data_counts[mask].astype(float)

        # Diffuse contribution aligned to the same restricted pixel set
        if band.diffuse_counts is not None:
            self._diffuse = band.diffuse_counts[mask]
        else:
            self._diffuse = 0.0

    @property
    def parameter_names(self):
        return self.source_model.parameter_names

    def parsubset(self, *select):
        return self.source_model.parsubset(*select)

    def counts(self):
        """Total model counts per active pixel: source + diffuse."""
        _, src_counts = self.band.pixel_counts(self._pix)
        return src_counts + self._diffuse

    def count_gradient(self):
        """Gradient of source counts w.r.t. free params, shape (n_params, n_pixels).

        Diffuse counts are treated as fixed (no gradient contribution).
        Returns (n_params, n_pixels) as expected by Likelihood._evaluate.
        """
        band = self.band
        g = []
        for src in band.source_model:
            grad = src.model.gradient(band.energy)[src.model.free]  # (n_free_for_src,)
            _, v = src.response(band).evaluate(self._pix)           # v: (n_pixels,)
            g.append(v[:, None] * grad[None, :])                    # (n_pixels, n_free_for_src)
        g = np.hstack(g)                                            # (n_pixels, n_params)
        g *= band.exposure_map(self._pix)[:, None] * band.delta_e   # apply exposure scaling
        return g.T                                                   # (n_params, n_pixels)


# --- Build model and data from band b ---
bm = BandModel(b)
show(f"""
**Active pixels**: {len(bm._pix)} of {len(b.data[0])} data pixels  
(restricted to those illuminated by pixel_counts())
""")

L = Likelihood(bm, bm.data)

# Evaluate at current parameter values
x0 = bm.parameters.values.copy()
logl_val, grad_val = L._evaluate(x0)

show(f"""
**Likelihood at initial parameters**  
log-likelihood: `{logl_val:.4f}`  
gradient: `{np.array2string(grad_val, precision=4)}`
""")

# Verify shapes and finiteness
assert grad_val.shape == (len(x0),), f"Unexpected gradient shape {grad_val.shape}"
assert np.isfinite(logl_val), "log-likelihood is not finite"
assert np.all(np.isfinite(grad_val)), "gradient contains non-finite values"
show("All assertions passed.")
