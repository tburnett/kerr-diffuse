import numpy as np



class PSFlookup:
    
    def __init__(self, table_path='files/loc'):
        """ A functor that returns the PSF for a given band, using the same PSF for all pixels in the band.

        Loads :class:`~pylib.psf_func.PSFlist` entries from *table_path* and
        matches each band to the nearest-energy PSF for its event type.
        When *table_path* is a directory, ``fb_psf_table.pkl`` is used for
        FRONT/BACK bands (event types 0-1) and ``psf_psf_table.pkl`` for
        PSF-partition bands (event types 2-5).  If an event type is absent
        from the tables, the FRONT (event type 0) shapes are used as a
        fallback.

        Parameters
        ----------
        table_path : str or Path, optional
            Directory containing ``fb_psf_table.pkl`` and
            ``psf_psf_table.pkl``, or a direct path to a single pickle file.
            Default is ``'files/loc'``.

        Returns
        -------
        PixelTable
            Returns *self* for method chaining.
        """
        from pylib.psf_func import PSFlist
        import copy

        all_psfs = PSFlist(event_type=None, table_path=table_path)
        if not all_psfs:
            print(f'PSFlookup: no PSF entries loaded from {table_path!r}')
            return self

        et_names = PSFlist.PSF.et_name
        ets = sorted({p.event_type for p in all_psfs})
        et_labels = [et_names[e] if e < len(et_names) else str(e) for e in ets]
        # print(f'PSFlookup: {len(all_psfs)} PSF entries '
        #       f'({", ".join(et_labels)}) from {table_path!r}')
        self.psf_list = all_psfs

    def __call__(self, band):
        """Return the PSF for *band*."""
        
        for candidate_psf in self.psf_list:
            if band.event_type != candidate_psf.event_type:
                continue
            if abs(candidate_psf.energy/band.energy-1)<0.1:
                # print(f'PSFlookup: found PSF {candidate_psf} for band {band}')
                return candidate_psf
        raise ValueError(f'PSFlookup: no PSF found for band {band} ')

