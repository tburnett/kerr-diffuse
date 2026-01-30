"""
Docstring for main
"""
from . parameterset import ParameterSet
from . sourcelist import SourceList


class Main(SourceList):

    def __init__(self, sources):
        """
        Set up the bands and a list of variable sources
        """
        super().__init__(sources)
        self.sources = self
        pass

    
    def fit(self, select=None, exclude=None,  summarize=True, setpars=None, **kwargs):
        """ Perform fit, return fitter object to examine errors, or refit
        
        Parameters
        ----------
        select : None, item or list of items, where item is an int or a string
            if not None, it defines a subset of the parameter numbers to select
                to define a projected function to fit
            int:  select the corresponding parameter number
            string: select parameters according to matching rules
                The name of a source (with possible wild cards) to select for fitting
                If initial character is '_', match the rest with parameter names
                if initial character is '_' and last character is '*', treat as wild card
        
        exclude : None, int, or list of int 
            if specified, will remove parameter numbers from selection

        summarize : bool
            if True (default) call summary after succesful fit

        setpars : dict | None
            set a set of parameters by index: the dict has keys that are either the index, 
            or the name of the varialbe, and float values,
            e.g. {1:1e-14, 2:2.1, 'Source_Index': 2.0}
            Note that this uses *internal* variables

        kwargs 
        ------
        ignore_exception : bool
                if set, run the fit in a try block and return None
        update_by : float
            set to zero to not change parameters, or a number between 0 and 1 to make a partial update
        tolerance : float, default 0.0
            If current fit quality, an estimate of potential improvent of the log likelihood, which is
            based on gradient and hessian is less than this, do not fit
        plot : bool
            if set to True, create plots of the parameter fits

            others passed to the fitter minimizer command. defaults are
                estimate_errors = True
                use_gradient = True
                
        """
        if len(self.sources.parameters)==0:
            print('No parameters to fit')
            return
        ignore_exception = kwargs.pop('ignore_exception', False)
        update_by = kwargs.pop('update_by', 1.0)
        tolerance = kwargs.pop('tolerance', 0.0)
        plot = kwargs.pop('plot', False)
       
        if setpars is not None: 
            self.sources.parameters.setitems(setpars, quiet=True)
            
        fit_kw = dict(use_gradient=True, estimate_errors=True)
        fit_kw.update(kwargs)

        with self.fitter_view(select, exclude=exclude) as fv:
            if tolerance>0:
                qual = fv.delta_loglike()
                if qual < tolerance and qual>0:
                    if summarize:
                        print('Not fitting, estimated improvement, %.2f, is less than tolerance= %.1f' % (qual, tolerance))
                        return
            try:
                qual=99.
                fv.maximize(**fit_kw)
                w = fv.log_like()
                self.fmin_ret = fv.fmin_ret
                if summarize:
                    print('%d calls, function value, improvement, quality: %.1f, %.2f, %.2f'\
                        % (fv.calls, w, w - fv.initial_likelihood, fv.delta_loglike()))
                # self.fit_info = dict(
                #     loglike = fv.log_like(),
                #     pars = fv.parameters[:], 
                #     covariance  = fv.covariance,
                #     mask_indeces = np.arange(len(fv.mask))[fv.mask],
                #     qual = fv.delta_loglike(),)
                fv.modify(update_by)
                if fit_kw['estimate_errors']: fv.save_covariance()
                if summarize: fv.summary()
                if plot: fv.plot_all()
                
            except Exception as msg:
                print('Fit Failure %s: quality: %.2f' % (msg, qual))
                fv.summary() # 
                if not ignore_exception: raise
            self.fit_info = dict(
                loglike = fv.log_like(),
                pars = fv.parameters[:], 
                covariance  = getattr(fv, 'covariance',None),
                mask_indeces = np.arange(len(fv.mask))[fv.mask],
                qual = fv.delta_loglike(),)
        return 
  
