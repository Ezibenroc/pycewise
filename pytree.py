import numpy
from copy import deepcopy

class IncrementalStat:
    '''See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Computing_shifted_data'''
    def __init__(self):
        self.values = []
        self.K = 0
        self.Ex = 0
        self.Ex2 = 0

    def __len__(self):
        return len(self.values)

    def add(self, val):
        if len(self) == 0:
            self.K = val
        self.values.append(val)
        self.Ex  += val - self.K
        self.Ex2 += (val - self.K)**2

    @property
    def last(self):
        return self.values[-1]

    @property
    def first(self):
        return self.values[0]

    def pop(self):
        val = self.values.pop()
        self.Ex  -= val - self.K
        self.Ex2 -= (val - self.K)**2
        return val

    @property
    def mean(self):
        if len(self) == 0:
            return 0
        return self.K + self.Ex/len(self)

    @property
    def var(self):
        if len(self) == 0:
            return 0
        return (self.Ex2 - (self.Ex**2)/len(self)) / len(self)

    @property
    def std(self):
        return self.var ** (1/2)

    @property
    def sum(self):
        return self.Ex + self.K*len(self)

    @property
    def sum_square(self):
        return self.Ex2 + 2*self.K*self.sum - len(self)*self.K**2

class Leaf:
    def __init__(self, x, y):
        '''Assume the values in x are sorted in increasing order.'''
        assert len(x) == len(y)
        self.x = IncrementalStat()
        self.y = IncrementalStat()
        self.cov_sum = IncrementalStat()
        self.xy = IncrementalStat()
        for xx, yy in zip(x, y):
            self.add(xx, yy)

    def __len__(self):
        return len(self.x)

    def __str__(self):
        if len(self) <= 1:
            return '😢'
        return 'y ~ %.3fx + %.3f' % (self.coeff, self.intercept)

    @property
    def first(self):
        return self.x.first

    @property
    def last(self):
        return self.x.last

    @property
    def mean_x(self):
        return self.x.mean

    @property
    def mean_y(self):
        return self.y.mean

    @property
    def std_x(self):
        return self.x.std

    @property
    def std_y(self):
        return self.y.std

    @property
    def cov(self):
        return self.cov_sum.mean

    @property
    def corr(self):
        return self.cov / (self.std_x * self.std_y)

    @property
    def coeff(self):
        return self.corr * self.std_y / self.std_x

    @property
    def intercept(self):
        return self.mean_y - self.coeff*self.mean_x

    @property
    def rsquared(self):
        return self.corr**2

    @property
    def MSE(self):
        a  = self.coeff
        b  = self.intercept
        x  = self.x.sum
        y  = self.y.sum
        x2 = self.x.sum_square
        y2 = self.y.sum_square
        xy = self.xy.sum
        return (+y2\
                -2*(a*xy + b*y)\
                +(a**2*x2 + 2*a*b*x + len(self)*b**2)
        )/len(self)

    @property
    def error(self):
        if len(self) <= 1:
            return float('inf')
        if self.MSE < 0:
            return 0
        else:
            return self.MSE** (1/2)

    def predict(self, x):
        return self.coeff*x + self.intercept

    def add(self, x, y):
        '''For the covariance, see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online.'''
        dx = x - self.mean_x
        self.x.add(x)
        self.y.add(y)
        self.xy.add(x*y)
        self.cov_sum.add(dx*(y - self.mean_y))

    def pop(self):
        self.cov_sum.pop()
        self.xy.pop()
        return self.x.pop(), self.y.pop()

class Node:
    STR_LJUST = 30
    def __init__(self, left_node, right_node):
        '''Assumptions:
             - all the x values in left_node are lower than the x values in right_node,
             - values in left_node  are sorted in increasing order (w.r.t. x),
             - values in right_node are sorted in decreasing order (w.r.t. x),
             - left_node.last  is the largest  x value in left_node,
             - right_node.last is the smallest x value in right_node.'''
        self.left = left_node
        self.right = right_node

    def __len__(self):
        return len(self.left) + len(self.right)

    @property
    def first(self):
        return self.left.first

    @property
    def last(self):
        return self.right.first

    @property
    def error(self):
        if len(self.left) <= 1 or len(self.right) <= 1:
            return float('inf')
        return len(self.left)/len(self)*self.left.error + len(self.right)/len(self)*self.right.error

    def left_to_right(self):
        x, y = self.left.pop()
        self.right.add(x, y)

    def right_to_left(self):
        x, y = self.right.pop()
        self.left.add(x, y)

    @property
    def split(self):
        return self.left.last

    @staticmethod
    def tabulate(string, pad='    ', except_first=False):
        substrings = string.split('\n')
        for i, s in enumerate(substrings):
            if i > 0 or not except_first:
                substrings[i] = pad + s
        return '\n'.join(substrings)

    def __str__(self):
        split = 'x ≤ %.3f?' % self.split
        left_str = str(self.left)
        left_str = self.tabulate(left_str, '│', True)
        left_str = '└──' + left_str
        left_str = self.tabulate(left_str)

        right_str = str(self.right)
        right_str = self.tabulate(right_str, ' ', True)
        right_str = '└──' + right_str
        right_str = self.tabulate(right_str)

        return '%s\n%s\n%s' % (split, left_str, right_str)

    def compute_best_fit(self):
        if len(self.right) == 0:
            nosplit = deepcopy(self.left)
            left_to_right = True
        else:
            assert len(self.left) == 0
            nosplit = deepcopy(self.right)
            left_to_right = False
        lowest_error  = self.error
        lowest_index  = 0
        for i in range(len(self)-1):
            if left_to_right:
                self.left_to_right()
            else:
                self.right_to_left()
            if self.error < lowest_error:
                lowest_error = self.error
                lowest_index = i
        if lowest_error + 1e-5 < nosplit.error: # TODO epsilon?
            while i > lowest_index:
                i -= 1
                if left_to_right:
                    self.right_to_left()
                else:
                    self.left_to_right()
            assert abs(lowest_error - self.error) < 1e-3
            self.left = Node(self.left, Leaf([], [])).compute_best_fit()
            self.right = Node(Leaf([], []), self.right).compute_best_fit()
            return self
        else:
            return nosplit

    def predict(self, x):
        raise NotImplementedError()

def compute_regression(x, y=None):
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
    return Node(Leaf(x, y), Leaf([], [])).compute_best_fit()
