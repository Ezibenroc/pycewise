from collections import namedtuple, Counter
import itertools
import math
from abc import ABC, abstractmethod
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import TypeVar, Generic, List, Generator, Callable, Union, Tuple, Dict
try:
    import pandas
except ImportError:
    pandas = None
try:
    import statsmodels.formula.api as statsmodels
except ImportError:
    statsmodels = None
try:
    import graphviz  # https://github.com/xflr6/graphviz
except ImportError:
    graphviz = None
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import palettable
except ImportError:
    palettable = None
try:
    import numpy as np
except ImportError:
    numpy = None
else:
    numpy = np  # for this import, mypy complains (see https://github.com/python/mypy/issues/1297)


Number = TypeVar('Number', float, Fraction, Decimal)
ExtNumber = Union[Number, int]


class Config:
    allowed_modes = ('AIC', 'BIC', 'log', 'weighted')

    def __init__(self, mode: str, epsilon: float) -> None:
        if mode not in self.allowed_modes:
            raise ValueError('Unknown mode %s. Authorized modes: %s.' %
                             (mode, ', '.join(self.allowed_modes)))
        self.mode = mode
        self.epsilon = epsilon

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Config):
            return False
        return self is other or (self.mode == other.mode and self.epsilon == other.epsilon)

    def __repr__(self) -> str:
        return '%s(%s, %.2e)' % (self.__class__.__name__, self.mode, self.epsilon)


class IncrementalStat(Generic[Number]):
    '''Represent a collection of numbers. Numbers can be added and removed (see methods add and pop).
    Several aggregated values (e.g., mean and variance) can be obtained in constant time.
    For the algorithms, see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance'''

    def __init__(self, func: Callable[[Number], Number] = lambda x: x) -> None:
        self.values: List[Number] = []
        self.Ex: List[Number] = []
        self.M2: List[Number] = []
        self.func: Callable[[Number], Number] = func

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Generator[Number, None, None]:
        yield from self.values

    def __reviter__(self) -> Generator[Number, None, None]:
        yield from reversed(self.values)

    def add(self, val: Number) -> None:
        '''Add a new element to the collection.'''
        original_value = val
        val = self.func(val)
        if len(self) == 0:
            new_Ex = val
            new_M2 = val.__class__(0)
        else:
            Ex = self.Ex[-1]
            M2 = self.M2[-1]
            n = len(self)+1
            new_Ex = Ex + (val-Ex)/n
            new_M2 = M2 + (val - Ex)*(val - new_Ex)
        self.values.append(original_value)
        self.Ex.append(new_Ex)
        self.M2.append(new_M2)

    @property
    def last(self) -> Number:
        '''Return the last element that was added to the collection.'''
        return self.values[-1]

    @property
    def first(self) -> Number:
        '''Return the first element that was added to the collection.'''
        return self.values[0]

    def pop(self) -> Number:
        '''Remove the last element that was added to the collection and return it.'''
        val = self.values.pop()
        self.Ex.pop()
        self.M2.pop()
        return val

    @property
    def mean(self) -> Number:
        '''Return the mean of all the elements of the collection.'''
        assert len(self) > 0
        return self.Ex[-1]

    @property
    def var(self) -> Number:
        '''Return the variance of all the elements of the collection.'''
        n = len(self)
        assert n > 1
        return self.M2[-1]/(n-1)

    @property
    def std(self) -> float:
        '''Return the standard deviation of all the elements of the collection.'''
        return math.sqrt(self.var)

    @property
    def sum(self) -> Number:
        '''Return the sum of all the elements of the collection.'''
        return self.mean*len(self)


class AbstractReg(ABC, Generic[Number]):
    '''An abstract class factorizing some common methods of Leaf and Node.
    '''
    config: Config

    def __repr__(self) -> str:
        return str(self)

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __iter__(self) -> Generator[Tuple[Number, Number], None, None]:
        pass

    @property
    @abstractmethod
    def breakpoints(self) -> List[Number]:
        pass

    @abstractmethod
    def merge(self):
        pass

    @property
    def null_RSS(self) -> bool:
        return self.RSS <= 0 or math.isclose(self.RSS, 0, abs_tol=self.config.epsilon**2)

    def rss_equal(self, a: Number, b: Number) -> bool:
        return math.isclose(a, b, abs_tol=self.config.epsilon**2)

    def error_equal(self, a: Number, b: Number) -> bool:
        assert self.config.mode in self.config.allowed_modes
        eps = abs(math.log2(self.config.epsilon**2))
        return math.isclose(a, b, abs_tol=eps)

    @property
    @abstractmethod
    def RSS(self) -> Number:
        pass

    @property
    @abstractmethod
    def nb_params(self) -> int:
        pass

    @property
    def error(self) -> float:
        '''Return an error, depending on the chosen mode. Lowest is better.'''
        try:
            if self.config.mode == 'AIC':
                return self.AIC
            elif self.config.mode == 'BIC':
                return self.BIC
            elif self.config.mode == 'log':
                return self.compute_BIClog()
            elif self.config.mode == 'weighted':
                return self.compute_weighted_BIC()
            else:
                assert False
        except (AssertionError, InvalidOperation):
            return float('inf')

    @abstractmethod
    def predict(self, x: Number) -> Number:
        pass

    @property
    def MSE(self) -> Number:
        '''Return the mean squared error (MSE) of the linear regression.'''
        return self.RSS / len(self)

    def information_criteria(self, param_penalty: float) -> float:
        try:
            RSS = float(self.RSS)
        except ZeroDivisionError:
            return float('inf')
        if self.null_RSS:
            # RSS cannot be null or negative
            RSS = math.ldexp(1.0, -1000)
        return param_penalty + len(self)*math.log(RSS/len(self))

    @property
    def AIC(self) -> float:
        '''Return the Akaike information criterion (AIC) of the linear regression.
        See https://en.wikipedia.org/wiki/Akaike_information_criterion#Comparison_with_least_squares'''
        param_penalty = 2*self.nb_params
        return self.information_criteria(param_penalty)

    @property
    def BIC(self) -> float:
        '''Return the bayesian information criterion (BIC) of the linear regression.
        See https://en.wikipedia.org/wiki/Bayesian_information_criterion#Gaussian_special_case'''
        param_penalty = math.log(len(self)) * self.nb_params
        return self.information_criteria(param_penalty)

    def compute_RSS(self) -> ExtNumber:
        '''Actually compute the residual sum of squares from scratch.
        Should be a bit more precise than the RSS property, but O(n) duration.'''
        return sum([(y - self.predict(x))**2 for x, y in self])

    def compute_weighted_RSS(self) -> ExtNumber:
        '''Actually compute the *weighted* residual sum of squares from scratch.
        Warning: this computation has O(n) complexity.'''
        return sum([((y - self.predict(x))/x)**2 for x, y in self])

    def compute_RSSlog(self) -> float:
        '''Warning: this computation has O(n) complexity.'''
        try:
            return sum([(math.log(y) - math.log(self.predict(x)))**2 for x, y in self])
        except ValueError:
            return float('inf')

    def compute_weighted_BIC(self) -> float:
        '''Return a custom error metric based on the weighted RSS.
        Warning: this computation has a O(n) complexity.'''
        N = len(self)
        try:
            param_penalty = math.log(N) * self.nb_params
            WRSS = self.compute_weighted_RSS()
            if WRSS <= 0:
                WRSS = math.ldexp(1., -1000)
            return param_penalty + N*math.log(WRSS/N)
        except ZeroDivisionError:
            return float('inf')

    def compute_BIClog(self) -> float:
        '''Return a custom error metric which is hopefully better suited to exponential scales.
        Warning: this computation has a O(n) complexity.'''
        try:
            N = len(self)
            param_penalty = math.log(N) * self.nb_params
            return param_penalty + N*self.compute_RSSlog()
        except ZeroDivisionError:
            return float('inf')

    def __plot_reg(self, color='red', log=False, use_statsmodels=False):
        # cannot use self.min, only Node objects have it
        min_x = math.floor(min(self)[0])
        max_x = math.ceil(max(self)[0])
        breaks = [min_x, *self.breakpoints, max_x]
        for i in range(len(breaks)-1):
            new_x = []
            start = breaks[i]*(1+1e-3)
            stop = breaks[i+1]*(1-1e-3)
            if log:
                if start <= 0:
                    # cannot plot negative values, capping to 0
                    x_i = math.ldexp(1.0, -1000)
                    if stop <= 0:
                        raise ValueError(
                            'Cannot plot in log scale with negative values.')
                else:
                    x_i = start
                while x_i < stop:
                    new_x.append(x_i)
                    x_i *= 1.5  # TODO find a better factor
            else:
                step = (stop-start)/1000
                x_i = start
                while x_i < stop:
                    new_x.append(x_i)
                    x_i += step
            new_x.append(stop)
            if not use_statsmodels:
                new_y = [self.predict(d) for d in new_x]
            else:
                if statsmodels is None:
                    raise ImportError('Could not import statsmodels')
                else:
                    self.compute_statsmodels_reg()
                    new_y = [self.predict_statsmodels(d) for d in new_x]
            plt.plot(new_x, new_y, '-', color=color)

    def __plot_points(self, alpha, color):
        segments = self.flatify().segments
        if not color:
            colors = ['black']
        else:
            if color is True:
                nbcolors = max(3, min(8, len(segments)))
                if palettable:
                    colors = palettable.colorbrewer.qualitative.__getattribute__('Dark2_%d' % nbcolors).mpl_colors
                else:
                    colors = ['red', 'green', 'blue']
            elif isinstance(color, str):
                colors = [color]
            else:
                colors = color
        for i, ((_, _), leaf) in enumerate(segments):
            data = list(leaf)
            x = [d[0] for d in data]
            y = [d[1] for d in data]
            plt.plot(x, y, 'o', color=colors[i % len(colors)], alpha=alpha)

    def __show_plot(self, log, log_x, log_y):
        if log or log_x:
            plt.xscale('log')
        if log or log_y:
            plt.yscale('log')

    def plot_dataset(self, log=False, log_x=False, log_y=False, alpha=0.5, color=True, plot_merged_reg=False,
                     use_statsmodels=False):
        if plt is None:
            raise ImportError('No module named "matplotlib".')
        plt.figure(figsize=(20, 20))
        plt.subplot(2, 1, 1)
        self.__plot_points(alpha=alpha, color=color)
        if len(self.breakpoints) > 0 and plot_merged_reg:
            self.merge().__plot_reg('red', log=log or log_x, use_statsmodels=use_statsmodels)
        if isinstance(self, Node) and plot_merged_reg:
            xl, yl = zip(*self.left)
            xr, yr = zip(*reversed(list(self.right)))
            Node(Leaf(xl, yl, self.config), Leaf(xr, yr, self.config),
                 no_check=True).__plot_reg('green', log=log or log_x)
        if color:
            self.__plot_reg('black', log=log or log_x)
        else:
            self.__plot_reg('red', log=log or log_x)
        for bp in self.breakpoints:
            plt.axvline(x=bp, color='black', linestyle='dashed', alpha=0.3)
        self.__show_plot(log, log_x, log_y)

    def plot_error(self, log=False, log_x=False, log_y=False, alpha=1):
        if plt is None:
            raise ImportError('No module named "matplotlib".')
        plt.figure(figsize=(20, 20))
        plt.subplot(2, 1, 1)
        x = []
        y = []
        x_min = []
        y_min = []
        min_err = self.errors.minsplit
        for d in self.errors.split:
            if self.error_equal(d[1], min_err):
                x_min.append(d[0])
                y_min.append(d[1])
            else:
                x.append(d[0])
                y.append(d[1])
        plt.plot(x, y, 'o', color='black', alpha=alpha)
        plt.plot(x_min, y_min, 'o', color='red')
        plt.axhline(y=self.errors.nosplit, color='red', linestyle='-')
        for bp in self.breakpoints:
            plt.axvline(x=bp, color='black', linestyle='dashed', alpha=0.3)
        self.__show_plot(log, log_x, log_y)

    @abstractmethod
    def _to_graphviz(self, dot) -> None:
        pass

    def flatify(self):
        all_x = []
        all_y = []
        for x, y in self:
            all_x.append(x)
            all_y.append(y)
        return FlatRegression(all_x, all_y, config=self.config, breakpoints=self.breakpoints)

    def simplify(self, RSSlog=False):
        return self.flatify().simplify(RSSlog=RSSlog)

    def auto_simplify(self, RSSlog=False):
        return self.flatify().auto_simplify(RSSlog=RSSlog)

    def to_pandas(self):
        if pandas is None:
            raise ImportError('No module named "pandas".')
        segments = []
        for (min_x, max_x), leaf in self.flatify().segments:
            segments.append({'min_x': min_x,
                             'max_x': max_x,
                             'intercept': leaf.intercept,
                             'coefficient': leaf.coeff,
                             'RSS': leaf.RSS,
                             'MSE': leaf.MSE,
                             'RSSlog': leaf.compute_RSSlog(),
                             'weighted_RSS': leaf.compute_weighted_RSS(),
                             })
            if statsmodels is not None:
                leaf.compute_statsmodels_reg()
                segments[-1]['statsmodels_intercept'] = leaf.statsmodels_intercept
                segments[-1]['statsmodels_coefficient'] = leaf.statsmodels_coeff
        return pandas.DataFrame(segments)


class Leaf(AbstractReg[Number]):
    '''Represent a collection of pairs (x, y), where x is a control variable and y is a response variable.
    Pairs can be added or removed (see methods add/pop) to the collection in any order.
    Several aggregated values can be obtained in constant time (e.g. covariance, coefficient and intercept
    of the linear regression).
    '''

    def __init__(self, x: List[Number], y: List[Number], config: Config) -> None:
        assert len(x) == len(y)
        self.config = config
        self.__modified = True
        self.x: IncrementalStat[Number] = IncrementalStat()
        self.y: IncrementalStat[Number] = IncrementalStat()
        self.counter_x: Dict[Number, int] = Counter()
        self.cov_sum: IncrementalStat[Number] = IncrementalStat()
        self.xy: IncrementalStat[Number] = IncrementalStat()
        self.x2: IncrementalStat[Number] = IncrementalStat(lambda x: x*x)
        self.y2: IncrementalStat[Number] = IncrementalStat(lambda x: x*x)
        for xx, yy in zip(x, y):
            self.add(xx, yy)

    def __len__(self) -> int:
        return len(self.x)

    def __str__(self) -> str:
        if len(self) <= 1:
            return '⊥'
        return 'y ~ %.3ex + %.3e' % (float(self.coeff), float(self.intercept))

    def _to_graphviz(self, dot):
        dot.node(str(id(self)), str(self))

    def __iter__(self) -> Generator[Tuple[Number, Number], None, None]:
        yield from zip(self.x, self.y)

    def __reviter__(self) -> Generator[Tuple[Number, Number], None, None]:
        yield from zip(self.x.__reviter__(), self.y.__reviter__())

    def __add__(self, other):
        x1 = self.x.values
        y1 = self.y.values
        x2 = other.x.values
        y2 = other.y.values
        if x2[0] > x2[-1]:
            assert y2[0] > y2[-1]
            x2 = list(reversed(x2))
            y2 = list(reversed(y2))
        return self.__class__(x1+x2, y1+y2, config=self.config)

    @property
    def first(self) -> Number:
        '''Return the first element x that was added to the collection.'''
        return self.x.first

    @property
    def last(self) -> Number:
        '''Return the last element x that was added to the collection.'''
        return self.x.last

    @property
    def mean_x(self) -> Number:
        '''Return the mean of the elements x of the collection.'''
        return self.x.mean

    @property
    def mean_y(self) -> Number:
        '''Return the mean of the elements y of the collection.'''
        return self.y.mean

    @property
    def std_x(self) -> float:
        '''Return the standard deviation of the elements x of the collection.'''
        return self.x.std

    @property
    def std_y(self) -> float:
        '''Return the standard deviation of the elements y of the collection.'''
        return self.y.std

    @property
    def cov(self) -> Number:
        '''Return the covariance between the elements x and the elements y.'''
        n = len(self)
        return self.cov_sum.mean * n / (n-1)

    @property
    def corr(self) -> float:
        '''Return the correlation coefficient between the elements x and the elements y.

        See https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#For_a_sample
        '''
        n = len(self)
        return float(self.xy.sum - n * self.mean_x * self.mean_y) / ((n - 1) * self.std_x * self.std_y)

    def _compute_weighted_parameters(self, weights):
        '''Return the tuple (intercept, coefficient) of the linear regression with the given weights.
        Warning: O(n) complexity.
        The formula is based on slide 7-3 from https://ms.mcmaster.ca/canty/teaching/stat3a03/Lectures7.pdf
        '''
        if len(self) <= 1:
            raise ZeroDivisionError
        assert len(self) == len(weights)
        Wtotal, XW, YW = 0, 0, 0
        for w, (x, y) in zip(weights, self):
            Wtotal += w
            XW += x*w
            YW += y*w
        XW /= Wtotal
        YW /= Wtotal
        num = 0
        denom = 0
        for w, (x, y) in zip(weights, self):
            num += (w*(x-XW)*(y-YW))
            denom += w*(x-XW)**2
        coeff = num/denom
        intercept = YW-coeff*XW
        return coeff, intercept

    def compute_weighted_parameters(self):
        if self.__modified:
            weights = [1/x for x in self.x]
            self.__wcoeff, self.__wintercept = self._compute_weighted_parameters(weights)
            self.__modified = False
        return self.__wcoeff, self.__wintercept

    def _compute_log_parameters(self, start_coeff=10, start_intercept=10, eps=1e-12,
                                max_iter=1000, return_search=False,
                                orthogonal_search=11,
                                forbid_negative_intercept=True, forbid_negative_coefficient=True):
        '''Return the tuple (intercept, coefficient) of the linear regression where the error function is logarithmic
        (i.e. we use the BIClog and RSSlog functions instead of BIC and RSS).
        Warning: O(Kn) complexity with K large...
        There is no closed formula for this, so we perform a gradient descent.
        '''
        if numpy is None:
            raise ImportError('No module named "numpy".')

        def deriv(coeff, intercept, x, y):
            '''Compute the value of the derivative of RSSlog in the given point (w.r.t. the intercept and the coefficient).
            '''
            if forbid_negative_intercept and intercept <= 0:
                raise ValueError('Negative intercept')
            if forbid_negative_coefficient and coeff <= 0:
                raise ValueError('Negative coefficient')
            S_intercept = 0
            S_coefficient = 0
            pred = x*coeff + intercept
            A = numpy.log(y) - numpy.log(pred)
            B = 1/(pred)
            S_intercept = A*B
            S_coefficient = (S_intercept*x).sum()
            S_intercept = S_intercept.sum()
            return -2*S_coefficient, -2*S_intercept

        def function(coeff, intercept, x, y):
            '''Compute the value of RSSlog in the given point.'''
            if (forbid_negative_intercept and intercept <= 0) or (forbid_negative_coefficient and coeff <= 0):
                return float('inf')
            return ((numpy.log(y) - numpy.log(x*coeff+intercept))**2).sum()

        def dot(Ax, Ay, Bx, By):
            return Ax*Bx + Ay*By

        def norm(Ax, Ay):
            return dot(Ax, Ay, Ax, Ay)**(1/2)

        def project_vector(Ax, Ay, Bx, By):
            '''Return the length of (Ax,Ay) projected onto (Bx,By).'''
            return dot(Ax, Ay, Bx, By) / norm(Bx, By)

        if len(self) <= 1:
            raise ZeroDivisionError
        x_val = numpy.array(list(self.x))
        y_val = numpy.array(list(self.y))
        coeff = start_coeff
        intercept = start_intercept
        error = function(coeff, intercept, x_val, y_val)
        i = 0
        if return_search:
            search_list = []
            search_list.append({'coefficient': coeff, 'intercept': intercept, 'error': error, 'index': i})
        # Start of the gradient descent loop
        while True:
            i += 1
            D_coefficient, D_intercept = deriv(coeff, intercept, x_val, y_val)
            if i % orthogonal_search == orthogonal_search-2:
                D_coefficient = 0
            elif i % orthogonal_search == orthogonal_search-1:
                D_intercept = 0
            D = norm(D_coefficient, D_intercept)
            if D < eps or i >= max_iter:
                break
            # We have a gradient direction, now we have to find out the distance.
            # We perform a binary search to find the appropriate step size.
            # First, we search for the upper bound of our binary search with an exponential increase.
            step = 1.
            while True:
                delta_coeff = D_coefficient*step
                delta_int = D_intercept*step
                try:
                    new_Deriv = deriv(coeff-delta_coeff, intercept-delta_int, x_val, y_val)
                except ValueError:  # negative log, we went too far
                    break
                if any(numpy.isnan(new_Deriv)):
                    break
                new_D = project_vector(new_Deriv[0], new_Deriv[1], D_coefficient, D_intercept)
                if new_D < 0:
                    break
                step *= 10
            # Then we do the binary search itself.
            interval = [0, step]
            last_good_step = 0
            while True:
                step = (interval[0] + interval[1])/2
                if step == interval[0] or step == interval[1]:
                    break
                delta_coeff = D_coefficient*step
                delta_int = D_intercept*step
                try:
                    new_Deriv = deriv(coeff-delta_coeff, intercept-delta_int, x_val, y_val)
                except ValueError:  # negative log, we went too far
                    interval[1] = step
                    continue
                if any(numpy.isnan(new_Deriv)):
                    interval[1] = step
                    continue
                last_good_step = step
                new_D = project_vector(new_Deriv[0], new_Deriv[1], D_coefficient, D_intercept)
                if abs(new_D) < 1e-50:
                    break
                if new_D > 0:
                    interval[0] = step
                else:
                    interval[1] = step
            # Here we have the distance and the direction, we can compute the next point.
            step = last_good_step
            delta_coeff = D_coefficient*step
            delta_int = D_intercept*step
            coeff -= delta_coeff
            intercept -= delta_int
            new_error = function(coeff, intercept, x_val, y_val)
            if abs(error-new_error) <= eps:
                break
            error = new_error
            if return_search:
                search_list.append({'coefficient': coeff, 'intercept': intercept,
                                    'error': error,
                                    'index': i,
                                    'final_step': step, 'D': D, 'new_D': new_D,
                                    'D_coeff': D_coefficient, 'D_inter': D_intercept})
        if return_search:
            return pandas.DataFrame(search_list)
        return coeff, intercept

    def compute_log_parameters(self):
        if self.__modified:
            self.__lcoeff, self.__lintercept = self._compute_log_parameters(
                    start_coeff=max(1e-300, abs(self._compute_classical_coeff())),
                    start_intercept=max(1e-300, abs(self._compute_classical_intercept())),
                    eps=1e-3)
            self.__modified = False
        return self.__lcoeff, self.__lintercept

    def _compute_classical_coeff(self):
        return self.cov / self.x.var

    def _compute_classical_intercept(self):
        return self.mean_y - self._compute_classical_coeff()*self.mean_x

    @property
    def coeff(self) -> Number:
        '''Return the coefficient α of the linear regression y = αx + β.'''
        if self.config.mode == 'weighted':
            return self.compute_weighted_parameters()[0]
        elif self.config.mode == 'log':
            return self.compute_log_parameters()[0]
        return self._compute_classical_coeff()

    @property
    def intercept(self) -> Number:
        '''Return the intercept β of the linear regression y = αx + β.'''
        if self.config.mode == 'weighted':
            return self.compute_weighted_parameters()[1]
        elif self.config.mode == 'log':
            return self.compute_log_parameters()[1]
        return self._compute_classical_intercept()

    @property
    def rsquared(self) -> float:
        '''Return the value R² of the linear regression y = αx + β.'''
        return self.corr**2

    @property
    def RSS(self) -> Number:
        '''Return the residual sum of squares (RSS) of the linear regression y = αx + β.
        See https://stats.stackexchange.com/a/333431/196336'''
        return self.MSE * len(self)

    @property
    def MSE(self) -> Number:
        '''Return the mean squared error (MSE) of the linear regression.'''
        n = len(self)
        return self.y.var * (n-1) / n - (self.cov * (n-1) / n)**2 / (self.x.var * (n-1) / n)

    @property
    def nb_params(self) -> int:
        '''Return the number of parameters of the model.
            There are only three parameters for a Leaf:
            - the slope,
            - the intercept,
            - the standard deviation of the residuals.
            This is true *if* we assume that the residuals follow a normal distribution fo mean 0.
        '''
        return 3

    def predict(self, x: Number) -> Number:
        '''Return a prediction of y for the variable x by using the linear regression y = αx + β.'''
        return self.coeff*x + self.intercept

    def predict_statsmodels(self, x):
        return self.statsmodels_coeff*x + self.statsmodels_intercept

    def add(self, x: Number, y: Number) -> None:
        '''Add the pair (x, y) to the collection.'''
        self.__modified = True
        if len(self) == 0:
            dx = x
        else:
            dx = x - self.mean_x
        self.x.add(x)
        self.counter_x[x] += 1
        self.y.add(y)
        self.xy.add(x*y)
        self.x2.add(x)
        self.y2.add(y)
        # For the covariance, see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online.
        self.cov_sum.add(dx*(y - self.mean_y))

    def pop(self) -> Tuple[Number, Number]:
        '''Remove and return the last pair (x, y) that was added to the collection.'''
        self.__modified = True
        self.cov_sum.pop()
        self.xy.pop()
        self.x2.pop()
        self.y2.pop()
        x = self.x.pop()
        self.counter_x[x] -= 1
        if self.counter_x[x] == 0:
            del self.counter_x[x]
        return x, self.y.pop()

    def pop_all(self) -> List[Tuple[Number, Number]]:
        '''Remove and return the last set of pairs (x_i, y_i) such that all x_i are equal and there is no more point in
        the dataset that have an x equal to x_i.'''
        result = []
        x, y = self.pop()
        result.append((x, y))
        while self.counter_x[x] > 0:
            x2, y2 = self.pop()
            assert x2 == x
            result.append((x2, y2))
        return result

    @property
    def breakpoints(self) -> List[Number]:
        return []  # no breakpoints

    def merge(self):
        return self  # nothing to do, already a single line

    def compute_statsmodels_reg(self) -> None:
        self.statsmodels_reg = statsmodels.ols(
            formula='y~x', data={'x': [float(x) for x in self.x.values],
                                 'y': [float(y) for y in self.y.values]}).fit()
        self.statsmodels_coeff = self.statsmodels_reg.params['x']
        self.statsmodels_intercept = self.statsmodels_reg.params['Intercept']

    def compute_statsmodels_RSS(self) -> float:
        try:
            self.statsmodels_reg
        except AttributeError:
            self.compute_statsmodels_reg()
        return self.statsmodels_reg.ssr


class Node(AbstractReg[Number]):
    STR_LJUST = 30
    Error = namedtuple('Error', ['nosplit', 'split', 'minsplit'])

    def __init__(self, left_node: AbstractReg, right_node: AbstractReg, *, no_check: bool = False) -> None:
        '''Assumptions:
             - all the x values in left_node are lower than the x values in right_node,
             - values in left_node  are sorted in increasing order (w.r.t. x),
             - values in right_node are sorted in decreasing order (w.r.t. x),
             - left_node.last  is the largest  x value in left_node,
             - right_node.last is the smallest x value in right_node.'''
        self.left = left_node
        self.right = right_node
        assert self.left.config == self.right.config
        self.config = self.left.config
        if len(self.right) == 0:
            self.nosplit = deepcopy(self.left)
            self.left_to_right = True
        else:
            assert no_check or len(self.left) == 0
            self.nosplit = deepcopy(self.right)
            self.left_to_right = False

    def __len__(self) -> int:
        return len(self.left) + len(self.right)

    def __iter__(self) -> Generator[Tuple[Number, Number], None, None]:
        if isinstance(self.right, Leaf):
            yield from itertools.chain(self.left, self.right.__reviter__())
        else:
            yield from itertools.chain(self.left, self.right)

    @property
    def min(self) -> Number:
        '''Return the smallest element of the node (if the assumptions are satisfied.)'''
        if isinstance(self.left, Node):
            return self.left.min
        elif isinstance(self.left, Leaf):
            return self.left.first
        else:
            raise ValueError()

    @property
    def max(self) -> Number:
        '''Return the largest element of the node (if the assumptions are satisfied.)'''
        if isinstance(self.right, Node):
            return self.right.max
        elif isinstance(self.right, Leaf):
            return self.right.first
        else:
            raise ValueError()

    @property
    def RSS(self) -> Number:
        '''Return the residual sum of squares (RSS) of the segmented linear regression.'''
        return self.left.RSS + self.right.RSS

    def compute_statsmodels_reg(self):
        self.left.compute_statsmodels_reg()
        self.right.compute_statsmodels_reg()

    def compute_statsmodels_RSS(self):
        return self.left.compute_statsmodels_RSS() + self.right.compute_statsmodels_RSS()

    @property
    def nb_params(self) -> int:
        '''Return the number of parameters of the model.'''
        return self.left.nb_params + self.right.nb_params + 1  # one additional parameter: the breakpoint

    def move_left_to_right(self) -> None:
        '''Move the last element(s) of the left node to the right node.'''
        assert isinstance(self.left, Leaf)
        assert isinstance(self.right, Leaf)
        for x, y in self.left.pop_all():
            self.right.add(x, y)

    def move_right_to_left(self) -> None:
        '''Move the last element(s) of the right node to the left node.'''
        assert isinstance(self.left, Leaf)
        assert isinstance(self.right, Leaf)
        for x, y in self.right.pop_all():
            self.left.add(x, y)

    def move_forward(self) -> None:
        '''Move element(s) from the node that was full at instantiation to the node that was empty at instantiation.'''
        if self.left_to_right:
            self.move_left_to_right()
        else:
            self.move_right_to_left()

    def move_backward(self) -> None:
        '''Move element(s) from the node that was empty at instantiation to the node that was full at instantiation.'''
        if self.left_to_right:
            self.move_right_to_left()
        else:
            self.move_left_to_right()

    @property
    def can_move(self) -> bool:
        assert isinstance(self.left, Leaf)
        assert isinstance(self.right, Leaf)
        if self.left_to_right:
            return len(self.left.counter_x) > 1
        else:
            return len(self.right.counter_x) > 1

    @property
    def split(self) -> Number:
        '''Return the current split (i.e., the largest element of the left node).'''
        assert len(self.left) > 0
        if isinstance(self.left, Node):
            return self.left.max
        else:
            assert isinstance(self.left, Leaf)
            return self.left.last

    @staticmethod
    def tabulate(string: str, pad: str = '    ', except_first: bool = False) -> str:
        '''Used for the string representation.'''
        substrings = string.split('\n')
        for i, s in enumerate(substrings):
            if i > 0 or not except_first:
                substrings[i] = pad + s
        return '\n'.join(substrings)

    def __str__(self) -> str:
        split = 'x ≤ %.3e?' % float(self.split)
        left_str = str(self.left)
        left_str = self.tabulate(left_str, '│', True)
        left_str = '└──' + left_str
        left_str = self.tabulate(left_str)

        right_str = str(self.right)
        right_str = self.tabulate(right_str, ' ', True)
        right_str = '└──' + right_str
        right_str = self.tabulate(right_str)

        return '%s\n%s\n%s' % (split, left_str, right_str)

    def to_graphviz(self):
        if graphviz is None:
            raise ImportError('No module named "graphviz".')
        dot = graphviz.Digraph()
        self._to_graphviz(dot)
        return dot

    def _to_graphviz(self, dot):
        dot.node(str(id(self)), 'x ≤ %.3e?' % float(self.split), shape='box')
        self.left._to_graphviz(dot)
        self.right._to_graphviz(dot)
        dot.edge(str(id(self)), str(id(self.left)), 'yes')
        dot.edge(str(id(self)), str(id(self.right)), 'no')

    def compute_best_fit(self, depth=0):
        '''Compute recursively the best fit for the dataset of this node, using a greedy algorithm. This can either be:
            - a leaf, representing a single linear regression,
            - a tree of nodes, representing a segmented linear regressions.'''
        lowest_error = self.error
        lowest_index = 0
        new_errors = []
        i = 0
        while self.can_move:
            self.move_forward()
            i += 1
            new_errors.append((self.split, self.error))
            if self.error < lowest_error:
                lowest_error = self.error
                lowest_split = self.split
                lowest_index = i
        # TODO stopping criteria?
        if lowest_error < self.nosplit.error and not self.error_equal(lowest_error, self.nosplit.error):
            while i > lowest_index:
                i -= 1
                self.move_backward()
            assert lowest_split == self.split
            self.left = Node(self.left, Leaf(
                [], [], config=self.config)).compute_best_fit(depth+1)
            self.right = Node(Leaf([], [], config=self.config),
                              self.right).compute_best_fit(depth+1)
            self.errors = self.Error(
                self.nosplit.error, new_errors, lowest_error)
            return self
        else:
            self.nosplit.errors = self.Error(
                self.nosplit.error, new_errors, lowest_error)
            return self.nosplit

    def predict(self, x: Number) -> Number:
        '''Return a prediction of y for the variable x by using the piecewise linear regression.'''
        if x <= self.split:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

    def predict_statsmodels(self, x):
        if x <= self.split:
            return self.left.predict_statsmodels(x)
        else:
            return self.right.predict_statsmodels(x)

    @property
    def breakpoints(self) -> List[Number]:
        return self.left.breakpoints + [self.split] + self.right.breakpoints

    def merge(self):
        return self.left.merge() + self.right.merge()


class FlatRegression(AbstractReg[Number]):
    def __init__(self, x: List[Number], y: List[Number], config: Config, breakpoints: List[Number]) -> None:
        self.config = config
        assert len(x) == len(y)
        assert list(sorted(set(breakpoints))) == breakpoints
        intervals: List[Tuple[Union[float, Number], Union[float, Number]]] = []
        if len(breakpoints) == 0:
            intervals.append((-float('inf'), float('inf')))
        else:
            intervals.append((-float('inf'), breakpoints[0]))
            for i in range(len(breakpoints)-1):
                intervals.append((breakpoints[i], breakpoints[i+1]))
            intervals.append((breakpoints[-1], float('inf')))
        self.segments: List[Tuple[Tuple[Union[float, Number], Union[float, Number]], Leaf[Number]]] = []
        points = list(sorted(zip(x, y)))
        for min_x, max_x in intervals:
            subx, suby = [], []
            for xx, yy in points:
                if min_x < xx <= max_x:
                    subx.append(xx)
                    suby.append(yy)
            self.segments.append(((min_x, max_x), Leaf(subx, suby, config=config)))

    def __repr__(self) -> str:
        result = []
        for (min_x, max_x), reg in self.segments:
            condition = '%.3e < x ≤ %.3e' % (float(min_x), float(max_x))
            result.append('%s\n\t%s' % (condition, str(reg)))
        return '\n'.join(result)

    @property
    def RSS(self) -> Number:
        '''Return the residual sum of squares (RSS) of the segmented linear regression.'''
        assert len(self.segments) > 0
        rss = self.segments[0][1].RSS
        for (_, _), leaf in self.segments[1:]:
            rss += leaf.RSS
        return rss

    def compute_statsmodels_reg(self):
        for _, reg in self.segments:
            reg.compute_statsmodels_reg()

    def __len__(self) -> int:
        return sum(len(leaf) for (_, _), leaf in self.segments)

    def __iter__(self) -> Generator[Tuple[Number, Number], None, None]:
        for (_, _), leaf in self.segments:
            yield from leaf

    def _to_graphviz(self, dot) -> None:
        dot.node(str(id(self)), str(self))

    @property
    def breakpoints(self):  # TODO typing
        result = []
        for (_, max_x), _ in self.segments[:-1]:
            result.append(max_x)
        return result

    @property
    def nb_params(self) -> int:
        total = 0
        for (_, _), leaf in self.segments:
            total += leaf.nb_params + 1
        return total - 1

    def predict(self, x: Number) -> Number:
        '''Return a prediction of y for the variable x by using the piecewise linear regression.'''
        for (min_x, max_x), leaf in self.segments:
            if min_x < x <= max_x:
                break
        return leaf.predict(x)

    def predict_statsmodels(self, x):
        for (min_x, max_x), leaf in self.segments:
            if min_x < x <= max_x:
                break
        return leaf.predict_statsmodels(x)

    def merge(self):
        leaf = Leaf([], [], config=self.config)
        for x, y in self:
            leaf.add(x, y)
        return leaf

    def __simplify(self, RSSlog=False):
        def RSS(x):
            if RSSlog or self.config.mode == 'log':
                return x.compute_RSSlog()
            elif self.config.mode == 'weighted':
                return x.compute_weighted_RSS()
            else:
                return x.RSS
        all_regressions = [self]
        while True:
            min_rss = float('inf')
            min_i = -1
            new_reg = deepcopy(all_regressions[-1])
            if len(new_reg.segments) <= 1:
                break
            for i in range(len(new_reg.segments)-1):
                (left_min, left_max), left_leaf = new_reg.segments[i]
                (right_min, right_max), right_leaf = new_reg.segments[i+1]
                rss_diff = RSS(left_leaf + right_leaf) - (RSS(left_leaf) + RSS(right_leaf))
                if rss_diff < min_rss:
                    min_rss = rss_diff
                    min_i = i
            (left_min, left_max), left_leaf = new_reg.segments[min_i]
            (right_min, right_max), right_leaf = new_reg.segments[min_i+1]
            new_reg.segments.pop(min_i+1)
            new_reg.segments[min_i] = (left_min, right_max), (left_leaf + right_leaf)
            all_regressions.append(new_reg)
        result = [{'regression': reg,
                   'RSS': reg.RSS,
                   'BIC': reg.BIC,
                   'AIC': reg.AIC,
                   'BIClog': reg.compute_BIClog(),
                   'RSSlog': reg.compute_RSSlog(),
                   'weighted_RSS': reg.compute_weighted_RSS(),
                   'weighted_BIC': reg.compute_weighted_BIC(),
                   'nb_breakpoints': len(reg.breakpoints)}
                  for reg in all_regressions]
        return result

    def simplify(self, RSSlog=False):
        result = self.__simplify(RSSlog=RSSlog)
        if pandas is not None:
            return pandas.DataFrame(result)
        else:
            return result

    def auto_simplify(self, RSSlog=False):
        def err(x):
            if RSSlog:
                return x.compute_BIClog()
            else:
                return x.error
        result = self.__simplify(RSSlog=RSSlog)
        min_error = float('inf')
        min_reg = None
        for res in result:
            reg = res['regression']
            new_error = err(reg)
            if min_error > new_error or reg.error_equal(min_error, new_error):
                min_error = new_error
                min_reg = reg
        return min_reg

    def flatify(self):
        return self


def compute_regression(x, y=None, *, breakpoints=None, mode='BIC', epsilon=None):
    '''Compute a segmented linear regression.
    The data can be given either as a tuple of two lists, or a list of tuples (each one of size 2).
    The first values represent the x, the second values represent the y.
    '''
    if y is not None:
        assert len(x) == len(y)
        dataset = list(zip(x, y))
    else:
        dataset = x
    assert all([len(d) == 2 for d in dataset])
    dataset = sorted(dataset)
    x = [d[0] for d in dataset]
    y = [d[1] for d in dataset]
    if epsilon:
        assert epsilon > 0
    else:
        epsilon = min([abs(yy) for yy in y])
    config = Config(mode, epsilon)
    if breakpoints is not None:
        return FlatRegression(x, y, config=config, breakpoints=breakpoints)
    reg = Node(Leaf(x, y, config=config), Leaf(
        [], [], config=config)).compute_best_fit()
    return reg
