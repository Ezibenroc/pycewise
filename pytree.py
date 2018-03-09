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
        return self.K + self.Ex/len(self)

    @property
    def var(self):
        return (self.Ex2 - (self.Ex**2)/len(self)) / len(self)

    @property
    def std(self):
        return self.var ** (1/2)

class Node:
    def __init__(self, x, y):
        '''Assume the values in x are sorted in increasing order.'''
        assert len(x) == len(y)
        self.x = IncrementalStat()
        self.y = IncrementalStat()
        self.cov_sum = IncrementalStat()
        for xx, yy in zip(x, y):
            self.add(xx, yy)

    def __len__(self):
        return len(self.x)

    def __str__(self):
        return 'y ~ %3.fx + %.3f | R^2 = %.3f' % (self.beta, self.alpha, self.rsquared)

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
    def beta(self):
        return self.corr * self.std_y / self.std_x

    @property
    def alpha(self):
        return self.mean_y - self.beta*self.mean_x

    @property
    def rsquared(self):
        return self.corr**2

    def add(self, x, y):
        self.x.add(x)
        self.y.add(y)
        self.cov_sum.add((x-self.mean_x)*(y-self.mean_y))

    def pop(self):
        x = self.x.last()
        y = self.y.last()
        self.cov_sum.pop()
        self.x.pop()
        self.y.pop()
        return x, y
