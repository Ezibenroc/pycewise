import numpy

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
        return 'y ~ %.3fx + %.3f' % (self.coeff, self.intercept)

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
        return self.MSE ** (1/2)

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
    def __init__(self, x, y):
        '''Assume the values in x are sorted in increasing order.'''
        assert len(x) == len(y)
        self.left = Leaf(x, y)
        self.right = Leaf([], [])

    def __len__(self):
        return len(self.left) + len(self.right)

    @property
    def error(self):
        return len(self.left)/len(self)*self.left.error + len(self.right)/len(self)*self.right.error

    def left_to_right(self):
        x, y = self.left.pop()
        self.right.add(x, y)

    def right_to_left(self):
        x, y = self.right.pop()
        self.left.add(x, y)

    @property
    def split(self):
        return self.left.x.last

    @staticmethod
    def tabulate(string):
        return '\n'.join('── ' + s for s in string.split('\n'))

    def __str__(self):
        split = 'x ≤ %.3f?' % self.split
        split     = '%s| error = %.3f' % (split.ljust(self.STR_LJUST+1), self.error)
        left_str  = '%s| error = %.3f' % (self.tabulate(str(self.left)).ljust(self.STR_LJUST), self.left.error)
        right_str = '%s| error = %.3f' % (self.tabulate(str(self.right)).ljust(self.STR_LJUST), self.right.error)
        return '%s\n└%s\n└%s' % (split, left_str, right_str)
