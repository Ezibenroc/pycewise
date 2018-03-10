import unittest
import random
import numpy
from pytree import Node, IncrementalStat

class IncrementalStatTest(unittest.TestCase):
    def test_basic(self):
        size = random.randint(50, 100)
        values = []
        stats = IncrementalStat()
        for _ in range(size):
            val = random.uniform(0, 100)
            stats.add(val)
            values.append(val)
        for _ in range(size-2): # don't do the last two ones
            val = stats.last
            self.assertEqual(stats.pop(), val)
            self.assertEqual(values.pop(), val)
            self.assertAlmostEqual(numpy.mean(values), stats.mean)
            self.assertAlmostEqual(numpy.var(values),  stats.var)
            self.assertAlmostEqual(numpy.std(values),  stats.std)


class NodeTest(unittest.TestCase):
    def setUp(self):
        self.coeff = random.uniform(0, 100)
        self.intercept = random.uniform(0, 100)
        self.size = random.randint(50, 100)
        self.data = []
        for _ in range(self.size):
            x = random.uniform(0, 100)
            y = self.coeff * x + self.intercept
            self.data.append((x, y))
        self.data.sort()

    def perform_tests(self, x, y, node, noisy):
        delta = 1e-10
        self.assertAlmostEqual(node.mean_x,   numpy.mean(x),              delta=delta)
        self.assertAlmostEqual(node.mean_y,   numpy.mean(y),              delta=delta)
        self.assertAlmostEqual(node.std_x,    numpy.std(x),               delta=delta)
        self.assertAlmostEqual(node.std_y,    numpy.std(y),               delta=delta)
        self.assertAlmostEqual(node.corr,     numpy.corrcoef(x, y)[1,0],  delta=delta)
        if noisy:
            delta = max(*x, *y)/100 # TODO better delta ?
        self.assertAlmostEqual(node.beta,     self.coeff,                 delta=delta)
        self.assertAlmostEqual(node.alpha,    self.intercept,             delta=delta)
        self.assertAlmostEqual(node.rsquared, 1,                          delta=delta)

    def test_init(self):
        for noise in [0, 1, 2, 4, 8]:
            x = [d[0] for d in self.data]
            y = [d[1] + random.gauss(0, noise) for d in self.data]
            node = Node(x, y)
            self.perform_tests(x, y, node, noise > 0)


    def test_add(self):
        for noise in [0, 1, 2, 4, 8]:
            x = [d[0] for d in self.data]
            y = [d[1] + random.gauss(0, noise) for d in self.data]
            limit = self.size // 3
            new_x = x[:limit]
            new_y = y[:limit]
            node = Node(list(new_x), list(new_y))
            self.perform_tests(new_x, new_y, node, noise > 0)
            for xx, yy in zip(x[limit:], y[limit:]):
                node.add(xx, yy)
                new_x.append(xx)
                new_y.append(yy)
                self.perform_tests(new_x, new_y, node, noise > 0)


if __name__ == "__main__":
    unittest.main()
