import numpy

class Node:
    def __init__(self, x, y):
        '''Assume the values in x are sorted in increasing order.'''
        assert len(x) == len(y)
        self.x = x
        self.y = y
        self.compute_values()

    def __len__(self):
        return len(self.x)

    def __str__(self):
        return 'y ~ %3.fx + %.3f | R^2 = %.3f' % (self.beta, self.alpha, self.rsquared)

    @staticmethod
    def square_sum(values, mean_val):
        s = 0
        for v in values:
            s += (v - mean_val)**2
        return s

    def compute_values(self):
        self.sum_x = sum(self.x)
        self.sum_y = sum(self.y)
        self.square_sum_x = self.square_sum(self.x, self.mean_x)
        self.square_sum_y = self.square_sum(self.y, self.mean_y)
        self.cov_sum = 0
        for x, y in zip(self.x, self.y):
            self.cov_sum += (x-self.mean_x)*(y-self.mean_y)

    @property
    def mean_x(self):
        return self.sum_x / len(self)

    @property
    def mean_y(self):
        return self.sum_y / len(self)

    @property
    def std_x(self):
        return (self.square_sum_x/len(self))**(1/2)

    @property
    def std_y(self):
        return (self.square_sum_y/len(self))**(1/2)

    @property
    def cov(self):
        return self.cov_sum / len(self)

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
        assert x >= self.x[-1]
        self.x.append(x)
        self.y.append(y)
        self.sum_x += x
        self.sum_y += y
        self.square_sum_x += (x-self.mean_x)**2
        self.square_sum_y += (y-self.mean_y)**2
        self.cov_sum += (x-self.mean_x)*(y-self.mean_y)

    def pop(self):
        x = self.x[-1]
        y = self.y[-1]
        self.cov_sum -= (x-self.mean_x)*(y-self.mean_y)
        self.square_sum_y -= (y-self.mean_y)**2
        self.square_sum_x -= (x-self.mean_x)**2
        self.sum_y -= y
        self.sum_x -= x
        self.x.pop()
        self.y.pop()
        return x, y
