from collections import namedtuple, Counter
import itertools
import math
import sys
from abc import ABC, abstractmethod
from copy import deepcopy
from decimal import Decimal
try:
    import statsmodels.formula.api as statsmodels
except ImportError:
    statsmodels = None
    sys.stderr.write('WARNING: no module statsmodels, the tree will not be simplified.')
try:
    import graphviz # https://github.com/xflr6/graphviz
except ImportError:
    graphviz = None
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

class IncrementalStat:
    '''Represent a collection of numbers. Numbers can be added and removed (see methods add and pop).
    Several aggregated values (e.g., mean and variance) can be obtained in constant time.
    For the algorithms, see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance'''

    def __init__(self, func=lambda x: x):
        self.values = []
        self.Ex = [0]
        self.M2 = [0]
        self.func = func

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        yield from self.values

    def __reviter__(self):
        yield from reversed(self.values)

    def add(self, val):
        '''Add a new element to the collection.'''
        original_value = val
        val = self.func(val)
        Ex = self.Ex[-1]
        M2 = self.M2[-1]
        n = len(self)+1
        new_Ex = Ex + (val-Ex)/n
        new_M2 = M2 + (val - Ex)*(val - new_Ex)
        self.values.append(original_value)
        self.Ex.append(new_Ex)
        self.M2.append(new_M2)

    @property
    def last(self):
        '''Return the last element that was added to the collection.'''
        return self.values[-1]

    @property
    def first(self):
        '''Return the first element that was added to the collection.'''
        return self.values[0]

    def pop(self):
        '''Remove the last element that was added to the collection and return it.'''
        val = self.values.pop()
        self.Ex.pop()
        self.M2.pop()
        return val

    @property
    def mean(self):
        '''Return the mean of all the elements of the collection.'''
        if len(self) == 0:
            return 0
        return self.Ex[-1]

    @property
    def var(self):
        '''Return the variance of all the elements of the collection.'''
        if len(self) == 0:
            return 0
        return self.M2[-1]/len(self) # TODO sample variance or population variance?

    @property
    def std(self):
        '''Return the standard deviation of all the elements of the collection.'''
        return math.sqrt(self.var)

    @property
    def sum(self):
        '''Return the sum of all the elements of the collection.'''
        return self.mean*len(self)

class AbstractReg(ABC):
    '''An abstract class factorizing some common methods of Leaf and Node.
    '''
    def __repr__(self):
        return str(self)

    @property
    def null_RSS(self):
        return self.RSS <= 0 or math.isclose(self.RSS, 0, abs_tol=self.config.epsilon**2)

    def error_equal(self, a, b):
        if self.config.mode == 'RSS':
            eps = self.config.epsilon**2
        else:
            assert self.config.mode in ('BIC', 'AIC')
            eps = abs(math.log2(self.config.epsilon**2))
        return math.isclose(a, b, abs_tol=eps)

    @property
    @abstractmethod
    def RSS(self):
        pass

    @property
    @abstractmethod
    def nb_params(self):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @property
    def MSE(self):
        '''Return the mean squared error (MSE) of the linear regression.'''
        return self.RSS / len(self)

    def information_criteria(self, param_penalty):
        RSS = self.RSS
        if self.null_RSS:
            RSS = RSS.__class__(math.ldexp(1.0, -1074)) # RSS cannot be null or negative
        try:
            logrss = RSS.ln() # works only for Decimal
        except AttributeError:
            logrss = math.log(RSS)
            param_penalty = float(param_penalty) # cannot add Decimal and float
        return param_penalty + len(self)*logrss

    @property
    def AIC(self):
        '''Return the Akaike information criterion (AIC) of the linear regression.
        See https://en.wikipedia.org/wiki/Akaike_information_criterion#Comparison_with_least_squares'''
        param_penalty = 2*self.nb_params
        return self.information_criteria(param_penalty)

    @property
    def BIC(self):
        '''Return the bayesian information criterion (BIC) of the linear regression.
        See https://en.wikipedia.org/wiki/Bayesian_information_criterion#Gaussian_special_case'''
        param_penalty = Decimal(len(self))
        param_penalty = param_penalty.ln()
        param_penalty *= self.nb_params
        return self.information_criteria(param_penalty)

    def compute_RSS(self):
        '''Actually compute the residual sum of squares from scratch.
        Should be a bit more precise than the RSS property, but O(n) duration.'''
        rss = 0
        for x, y in self:
            rss += (y - self.predict(x))**2
        return rss

    def __plot_reg(self, color='red', log=False):
        min_x = math.floor(min(self)[0]) # cannot use self.min, only Node objects have it
        max_x = math.ceil(max(self)[0])
        breaks = [min_x, *self.breakpoints, max_x]
        for i in range(len(breaks)-1):
            new_x = []
            start = breaks[i]*(1+1e-3)
            stop = breaks[i+1]*(1-1e-3)
            if log:
                x_i = start
                while x_i < stop:
                    new_x.append(x_i)
                    x_i *= 1.5 # TODO find a better factor
            else:
                step = (stop-start)/1000
                x_i = start
                while x_i < stop:
                    new_x.append(x_i)
                    x_i += step
            new_x.append(stop)
            new_y = [self.predict(d) for d in new_x]
            plt.plot(new_x, new_y, '-', color=color)

    def __show_plot(self, log, log_x, log_y):
        axes = plt.gca()
        if log or log_x:
            plt.xscale('log')
        if log or log_y:
            plt.yscale('log')
        plt.show()

    def plot_dataset(self, log=False, log_x=False, log_y=False, alpha=0.5):
        data = list(self)
        x = [d[0] for d in data]
        y = [d[1] for d in data]
        plt.figure(figsize=(20,20))
        plt.subplot(2,1,1)
        plt.plot(x, y, 'o', color='blue', alpha=alpha)
        if len(self.breakpoints) > 0:
            self.merge().__plot_reg('black', log=log or log_x)
        if len(self.breakpoints) > 1:
            xl, yl = zip(*self.left)
            xr, yr = zip(*reversed(list(self.right)))
            Node(Leaf(xl, yl, self.config), Leaf(xr, yr, self.config), no_check=True).__plot_reg('green', log=log or log_x)
        self.__plot_reg()
        for bp in self.breakpoints:
            plt.axvline(x=bp, color='black', linestyle='dashed', alpha=0.3)
        self.__show_plot(log, log_x, log_y)

    def plot_error(self, log=False, log_x=False, log_y=False, alpha=1):
        plt.figure(figsize=(20,20))
        plt.subplot(2,1,1)
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
        plt.plot(x, y, 'o', color='blue', alpha=alpha)
        plt.plot(x_min, y_min, 'o', color='red')
        plt.axhline(y=self.errors.nosplit, color='red', linestyle='-')
        for bp in self.breakpoints:
            plt.axvline(x=bp, color='black', linestyle='dashed', alpha=0.3)
        self.__show_plot(log, log_x, log_y)

class Leaf(AbstractReg):
    '''Represent a collection of pairs (x, y), where x is a control variable and y is a response variable.
    Pairs can be added or removed (see methods add/pop) to the collection in any order.
    Several aggregated values can be obtained in constant time (e.g. covariance, coefficient and intercept
    of the linear regression).
    '''

    def __init__(self, x, y, config):
        assert len(x) == len(y)
        self.config = config
        self.x = IncrementalStat()
        self.y = IncrementalStat()
        self.counter_x = Counter()
        self.cov_sum = IncrementalStat()
        self.xy = IncrementalStat()
        self.x2 = IncrementalStat(lambda x: x**2)
        self.y2 = IncrementalStat(lambda x: x**2)
        for xx, yy in zip(x, y):
            self.add(xx, yy)

    def __len__(self):
        return len(self.x)

    def __str__(self):
        if len(self) <= 1:
            return '⊥'
        return 'y ~ %.3ex + %.3e' % (self.coeff, self.intercept)

    def _to_graphviz(self, dot):
        dot.node(str(id(self)), str(self))

    def __iter__(self):
        yield from zip(self.x, self.y)

    def __reviter__(self):
        yield from zip(self.x.__reviter__(), self.y.__reviter__())

    def __add__(self, other):
        x1 = self.x.values
        y1 = self.y.values
        x2 = other.x.values
        y2 = other.y.values
        return self.__class__(x1+list(reversed(x2)), y1+list(reversed(y2)), config=self.config)

    def __eq__(self, other):
        '''Return True if the two Leaf instances are not *significantly* different.
        They are significantly different if (at least) one of the conditions holds:
            - One and only one of the two intercepts   is significant.
            - One and only one of the two coefficients is significant.
            - Both intercepts   are significant and their confidence intervals do not overlap.
            - Both coefficients are significant and their confidence intervals do not overlap.
        '''
        if not isinstance(other, self.__class__):
            return False
        if len(self) <= 5 or len(other) <= 5: # too few points anyway...
            return True
        pvalues_thresh = 1e-3
        reg1 = statsmodels.ols(formula='y~x', data={'x': self.x.values,  'y': self.y.values }).fit()
        confint1 = reg1.conf_int(alpha=0.05, cols=None)
        reg2 = statsmodels.ols(formula='y~x', data={'x': other.x.values, 'y': other.y.values}).fit()
        confint2 = reg2.conf_int(alpha=0.05, cols=None)
        signif_intercept1 = reg1.pvalues.Intercept < pvalues_thresh
        signif_intercept2 = reg2.pvalues.Intercept < pvalues_thresh
        signif_slope1 = reg1.pvalues.x < pvalues_thresh
        signif_slope2 = reg2.pvalues.x < pvalues_thresh
        if signif_intercept1 and signif_intercept2: # intercept is significant for both
            if confint1[1].Intercept < confint2[0].Intercept-1e-3 or confint1[0].Intercept-1e-3 > confint2[1].Intercept: # non-overlapping C.I. for intercept
                return False
        elif signif_intercept1 or signif_intercept2: # intercept is significant for only one
            return False
        if signif_slope1 and signif_slope2: # slope is significant for both
            if confint1[1].x < confint2[0].x-1e-3 or confint1[0].x-1e-3 > confint2[1].x: # non-overlapping C.I. for slope
                return False
        elif signif_slope1 or signif_slope2:
            return False
        return True

    @property
    def first(self):
        '''Return the first element x that was added to the collection.'''
        return self.x.first

    @property
    def last(self):
        '''Return the last element x that was added to the collection.'''
        return self.x.last

    @property
    def mean_x(self):
        '''Return the mean of the elements x of the collection.'''
        return self.x.mean

    @property
    def mean_y(self):
        '''Return the mean of the elements y of the collection.'''
        return self.y.mean

    @property
    def std_x(self):
        '''Return the standard deviation of the elements x of the collection.'''
        return self.x.std

    @property
    def std_y(self):
        '''Return the standard deviation of the elements y of the collection.'''
        return self.y.std

    @property
    def cov(self):
        '''Return the covariance between the elements x and the elements y.'''
        return self.cov_sum.mean

    @property
    def corr(self):
        '''Return the correlation coefficient between the elements x and the elements y.'''
        return self.cov / (self.std_x * self.std_y)

    @property
    def coeff(self):
        '''Return the coefficient α of the linear regression y = αx + β.'''
        return self.cov / self.x.var

    @property
    def intercept(self):
        '''Return the intercept β of the linear regression y = αx + β.'''
        return self.mean_y - self.coeff*self.mean_x

    @property
    def rsquared(self):
        '''Return the value R² of the linear regression y = αx + β.'''
        return self.corr**2

    @property
    def RSS(self):
        '''Return the residual sum of squares (RSS) of the linear regression y = αx + β.
        See https://stats.stackexchange.com/a/333431/196336'''
        try:
            a  = self.coeff
            b  = self.intercept
            x  = self.x.sum
            y  = self.y.sum
            x2 = self.x2.sum
            y2 = self.y2.sum
            xy = self.xy.sum
            return (+y2\
                    -2*(a*xy + b*y)\
                    +(a**2*x2 + 2*a*b*x + len(self)*b**2)
            )
        except ZeroDivisionError:
            return float('inf')

    @property
    def nb_params(self):
        '''Return the number of parameters of the model.'''
        return 3 # only three parameters: slope, intercept and standard deviation of the residuals (we assume they follow a normal distribution of mean 0)

    @property
    def error(self):
        '''Return an error, depending on the chosen mode. Lowest is better.'''
        if self.std_x == 0:
            return float('inf')
        if self.config.mode == 'AIC':
            return self.AIC
        if self.config.mode == 'BIC':
            return self.BIC
        if self.config.mode == 'RSS':
            if self.MSE < 0:
                return 0
            else:
                return self.MSE** (1/2)

    def predict(self, x):
        '''Return a prediction of y for the variable x by using the linear regression y = αx + β.'''
        return self.coeff*x + self.intercept

    def add(self, x, y):
        '''Add the pair (x, y) to the collection.'''
        dx = x - self.mean_x
        self.x.add(x)
        self.counter_x[x] += 1
        self.y.add(y)
        self.xy.add(x*y)
        self.x2.add(x)
        self.y2.add(y)
        # For the covariance, see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online.
        self.cov_sum.add(dx*(y - self.mean_y))

    def pop(self):
        '''Remove and return the last pair (x, y) that was added to the collection.'''
        self.cov_sum.pop()
        self.xy.pop()
        self.x2.pop()
        self.y2.pop()
        x = self.x.pop()
        self.counter_x[x] -= 1
        if self.counter_x[x] == 0:
            del self.counter_x[x]
        return x, self.y.pop()

    def pop_all(self):
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

    def simplify(self):
        '''Does nothing.'''
        return self # nothing to do

    @property
    def breakpoints(self):
        return [] # no breakpoints

    def merge(self):
        return self # nothing to do, already a single line

    def compute_statsmodels_reg(self):
        self.statsmodels_reg = statsmodels.ols(formula='y~x', data={'x': self.x.values,  'y': self.y.values }).fit()

    def compute_statsmodels_RSS(self):
        try:
            self.statsmodels_reg
        except AttributeError:
            self.compute_statsmodels_reg()
        return self.statsmodels_reg.ssr

class Node(AbstractReg):
    STR_LJUST = 30
    Error = namedtuple('Error', ['nosplit', 'split', 'minsplit'])
    def __init__(self, left_node, right_node, *, no_check=False):
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

    def __len__(self):
        return len(self.left) + len(self.right)

    def __iter__(self):
        try:
            yield from itertools.chain(self.left, self.right.__reviter__())
        except AttributeError: # right is *not* a leaf
            yield from itertools.chain(self.left, self.right)

    @property
    def min(self):
        '''Return the smallest element of the node (if the assumptions are satisfied.)'''
        try:
            return self.left.min
        except AttributeError: # left is a leaf
            return self.left.first

    @property
    def max(self):
        '''Return the largest element of the node (if the assumptions are satisfied.)'''
        try:
            return self.right.max
        except AttributeError: # right is a leaf
            return self.right.first

    @property
    def RSS(self):
        '''Return the residual sum of squares (RSS) of the segmented linear regression.'''
        return self.left.RSS + self.right.RSS

    def compute_statsmodels_RSS(self):
        return self.left.compute_statsmodels_RSS() + self.right.compute_statsmodels_RSS()

    @property
    def nb_params(self):
        '''Return the number of parameters of the model.'''
        return self.left.nb_params + self.right.nb_params + 1 # one additional parameter: the breakpoint

    @property
    def error(self):
        '''Return an error, depending on the chosen mode. Lowest is better.'''
        if len(self.left) <= 1 or len(self.right) <= 1:
            return float('inf')
        if self.config.mode == 'AIC':
            return self.AIC
        if self.config.mode == 'BIC':
            return self.BIC
        if self.config.mode == 'RSS':
            return len(self.left)/len(self)*self.left.error + len(self.right)/len(self)*self.right.error

    def move_left_to_right(self):
        '''Move the last element(s) of the left node to the right node.'''
        for x, y in self.left.pop_all():
            self.right.add(x, y)

    def move_right_to_left(self):
        '''Move the last element(s) of the right node to the left node.'''
        for x, y in self.right.pop_all():
            self.left.add(x, y)

    def move_forward(self):
        '''Move element(s) from the node that was full at instantiation to the node that was empty at instantiation.'''
        if self.left_to_right:
            self.move_left_to_right()
        else:
            self.move_right_to_left()

    def move_backward(self):
        '''Move element(s) from the node that was empty at instantiation to the node that was full at instantiation.'''
        if self.left_to_right:
            self.move_right_to_left()
        else:
            self.move_left_to_right()

    @property
    def can_move(self):
        if self.left_to_right:
            return len(self.left.counter_x) > 1
        else:
            return len(self.right.counter_x) > 1

    @property
    def split(self):
        '''Return the current split (i.e., the largest element of the left node).'''
        assert len(self.left) > 0
        try:
            return self.left.max
        except AttributeError: # left is a leaf
            return self.left.last

    @staticmethod
    def tabulate(string, pad='    ', except_first=False):
        '''Used for the string representation.'''
        substrings = string.split('\n')
        for i, s in enumerate(substrings):
            if i > 0 or not except_first:
                substrings[i] = pad + s
        return '\n'.join(substrings)

    def __str__(self):
        split = 'x ≤ %.3e?' % self.split
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
        dot.node(str(id(self)), 'x ≤ %.3e?' % self.split, shape='box')
        self.left._to_graphviz(dot)
        self.right._to_graphviz(dot)
        dot.edge(str(id(self)), str(id(self.left)), 'yes')
        dot.edge(str(id(self)), str(id(self.right)), 'no')

    def compute_best_fit(self, depth=0):
        '''Compute recursively the best fit for the dataset of this node, using a greedy algorithm. This can either be:
            - a leaf, representing a single linear regression,
            - a tree of nodes, representing a segmented linear regressions.'''
        lowest_error  = self.error
        lowest_index  = 0
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
        if lowest_error < self.nosplit.error and not self.error_equal(lowest_error, self.nosplit.error): # TODO stopping criteria?
            while i > lowest_index:
                i -= 1
                self.move_backward()
            assert lowest_split == self.split
            self.left = Node(self.left, Leaf([], [], config=self.config)).compute_best_fit(depth+1)
            self.right = Node(Leaf([], [], config=self.config), self.right).compute_best_fit(depth+1)
            self.errors = self.Error(self.nosplit.error, new_errors, lowest_error)
            return self
        else:
            self.nosplit.errors = self.Error(self.nosplit.error, new_errors, lowest_error)
            return self.nosplit

    def predict(self, x):
        '''Return a prediction of y for the variable x by using the piecewise linear regression.'''
        if x <= self.split:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

    def simplify(self):
        '''Recursively simplify the tree (if possible).
        If two leaves of a same node are considered equal (see Leaf.__eq__ method), then they are merged:
        this node becomes a Leaf containing the union of the two leaves.'''
        left = self.left.simplify()
        right = self.right.simplify()
        if type(self.right) != type(right): # keeping the property that right leaves are in reverse order
            right.x.values = list(reversed(right.x.values))
            right.y.values = list(reversed(right.y.values))
        if isinstance(left, Leaf) and isinstance(right, Leaf):
            merge = left + right
            if left == right or merge == left or merge == right:
                result = merge
            else:
                result = Node(left, right)
        else:
            result = Node(left, right)
        result.errors = self.errors
        return result

    @property
    def breakpoints(self):
        return self.left.breakpoints + [self.split] + self.right.breakpoints

    def merge(self):
        return self.left.merge() + self.right.merge()

class Config:
    allowed_modes = ('AIC', 'BIC', 'RSS')
    def __init__(self, mode, epsilon):
        if mode not in self.allowed_modes:
            raise ValueError('Unknown mode %s. Authorized modes: %s.' % (mode, ', '.join(self.allowed_modes)))
        self.mode = mode
        self.epsilon = epsilon

    def __eq__(self, other):
        return self is other or (self.mode == other.mode and self.epsilon == other.epsilon)

def compute_regression(x, y=None, *, simplify=False, mode='BIC', epsilon=None):
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
    reg = Node(Leaf(x, y, config=config), Leaf([], [], config=config)).compute_best_fit()
    if statsmodels and simplify:
        reg = reg.simplify()
    return reg
