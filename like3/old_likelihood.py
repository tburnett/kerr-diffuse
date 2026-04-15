""" 
Manage likelihood analysis
"""
import numpy as np
from . views import (FitterMixin, FitterView, SubsetFitterView)


class Likelihood(FitterMixin) :
    """ Compute the log likelihood as a function of a set of parameters and the pixel data. 

    """
    def __init__(self, model, data,  **kwargs):
        """ 
        Parameters
        ----------
        model : callable
            A model function that manages its parameters and returns expected counts

        data : array-like
            The observed data to compute the likelihood against--
        """
        
        self.model = model  # model function that computes expected counts
        self.parameterset = model # all implents paraamterset interface
        self.data = data
        self.calls =0

    @property
    def sources(self):
        """ Return the source list associated with the parameters
        """
        return self.parameterset.sources
    
    def __repr__(self):
        return f'<Likelihood: {len(self.parameterset)} parameters, {len(self.data)} data points>'
   
    def __getitem__(self, index):
        """ Return the source at the given index
        """
        return self.sources.__getitem__(index)
    
    def get_parameters(self):
        """ Return the current parameter values as a numpy array
        """
        return self.parameterset.values
    
    def set_parameters(self, values):
        """ Set the parameter values from a numpy array
        """
        self.parameterset.set_values(np.atleast_1d(values))
    
    def get_source(self, source_name=None):
        return self.model.find_source(source_name)

    @property
    def bounds(self):
        """ Return the bounds for the parameters as a list of (min, max) tuples
        """
        return self.parameterset.bounds

    def log_like(self, pars=None,*, summed=True): 

        """ Evaluate the log likelihood with the current parameter values
        summed: If False, return array of individual band likelihoods

        """
        if pars is not None:
            self.set_parameters(pars)

        # Compute model predictions
        model_predictions = self.model()

        # Compute log likelihood based on data and model predictions
        ret = self.data * np.log(model_predictions) - model_predictions  # Poisson likelihood
        return np.sum(ret) if summed else ret
    
    def __call__(self, pars):
        
        self.calls+=1
        return -self.log_like(pars)
        
    def gradient(self, pars=None, summed=True, numeric=False):
        """ Compute the gradient of the (-) log likelihood at the given parameter values

        """   
        if pars is None:
            pars = self.get_parameters()   
        else:
            self.set_parameters(np.atleast_1d(pars))
        self.log_like(pars)  # to update internal state?  

        if not numeric:
            d = self.data
            f = self.model()
            df = self.model.gradient()
            return -((d/f-1)*df).sum(axis=1)

        # This is a placeholder implementation; actual gradient computation would depend on the model
        epsilon = 1e-6
        pars = self.get_parameters()
        grad = np.zeros_like(pars)
        for i in range(len(grad)):
            params_up = pars.copy()
            params_down = pars.copy()
            params_up[i] += epsilon
            params_down[i] -= epsilon
            grad[i] = (self.log_like(params_up,summed=summed) - 
                       self.log_like(params_down,summed=summed)) / (2 * epsilon)
        return -grad
    
    def update(self):
        """ Update internal state after parameter changes
        """
        # for source in self.sources:
        #     source.update_model()

    @classmethod
    def simple_test(cls):

        class SimpleModel:
            def __call__(self, parameters):
                return parameters.values[0] * np.ones(10)
        class SimpleParameters:
            def __init__(self):
                self.values = np.array([2.0])
            def set_values(self, values):
                self.values = values
        class SimpleData:
            def __init__(self):
                self.data = np.array([1,2,3,4,5,6,7,8,9,10])
            def __mul__(self, other):
                return np.sum(self.data * other)
        model = SimpleModel()
        parameters = SimpleParameters()
        data = SimpleData()
        ll = cls(model, parameters, data)
        print("Log Likelihood at initial parameters:", ll(parameters.values))
        new_params = np.array([3.0])
        print("Log Likelihood at new parameters:", ll(new_params))

   
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
        if len(self.parameterset)==0:
            print('No parameters to fit')
            return
        ignore_exception = kwargs.pop('ignore_exception', False)
        update_by = kwargs.pop('update_by', 1.0)
        plot = kwargs.pop('plot', False)
        
        if setpars is not None: 
            self.sources.parameters.setitems(setpars, quiet=True)
            
        fit_kw = dict(use_gradient=True, estimate_errors=True)
        fit_kw.update(kwargs)

        self.initial_parameters = self.get_parameters()
        self.initial_likelihood = self.log_like(self.initial_parameters)

        with self.fitter_view(select=select, exclude=exclude) as fv:

            try:
                qual=99.
                self.fit_value, self.fit_pars, _ = fv.maximize(**fit_kw)
                w = fv.log_like()

                if summarize:
                    print('%d calls, log like improvement, quality:  %.2f, %.2f'\
                        % (fv.calls, w - fv.initial_likelihood, fv.delta_loglike()))

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

    def summary(self, select=None, exclude=None, out=None, title=None, gradient=True):
        """ summary table of free parameters, values uncertainties gradient
        
        Parameters:
        ----------
        select : list of integers or string
            integers are indices of parameters
            string is the wildcarded name of a source
        out : open file or None
        title: None or string
        gradient: bool
            set False to not print gradient
            
        """
        if title is not None:
            print(title, file=out)

        fmt, tup = '%-21s%6s%10s%10s', tuple('Name index value error(%)'.split())
        if gradient:
            grad = self.gradient(self.initial_parameters)
            fmt +='%10s'; tup += ('gradient',)
        print(fmt %tup, file=out)
        prev=''
        selected = (select, exclude)
        index_array = np.arange(len(self.model.mask))[self.model.mask]
        for index, (name, value, rsig) in enumerate(zip(self.model.parameter_names, 
                                                        self.model.values, 
                                                        self.model.uncertainties)):
            t = name.split('_')
            pname = t[-1]
            sname = '_'.join(t[:-1])
            if sname==prev: name = len(sname)*' '+'_'+pname
            prev = sname
            fmt = '%-21s%6d%10.4g%10s'
            psig = '%.1f'%(rsig*100) if rsig>0 and not np.isnan(rsig) else '***'
            truncname = name[:20]+'*' if len(name)>20 else name
            tup = (truncname, index_array[index], value,psig)
            if gradient:
                fmt +='%10.1f'; tup += (grad[index],)
            # print(tup, fmt)
            print(fmt % tup, file=out)

    def fitter_view(self, select=None, setpars=None, **kwargs):
        """ return a object to use with a fitter.
            Two versions, one with full set of parameters, other if a subset is specified
        """
        if setpars is not None: 
            self.sources.parameters.setitems(setpars)

        if select is None:
            return FitterView(self, **kwargs)
        return SubsetFitterView(self, select, **kwargs)


    @classmethod
    def test(cls, N=1000, random_state=42):  

        from scipy import stats
        
        class ParameterSet:
            def __init__(self, value):
                self._value = np.array(value, dtype=float)
            @property
            def values(self):
                return self._value
            def set_values(self, value):
                self._value = np.array(value)
            @property
            def bounds(self):
                return [(-5,5) for _ in self._value]
            def __repr__(self) -> str:
                return f'{self._value}'
            
        bins = bins=np.arange(-4,4.01,(binsize:=0.2))
        centers = centers = 0.5*(bins[1:]+bins[:-1])

        pars = ParameterSet([0,1])
        norm_truth = stats.norm(*pars.values)
        x = norm_truth.rvs(size:=N, random_state=random_state)        
        data, _= np.histogram(x, bins=bins)

        model = lambda parset: stats.norm(*parset.values).pdf(centers)*size*binsize    
        self =  cls(model, pars, data)
        self.bins = bins
        self.centers = centers

        # self.maximize(quiet=False,  estimate_errors=True)
        return self

