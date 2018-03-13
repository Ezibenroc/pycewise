import unittest
import random
import numpy
from pytree import Node, Leaf, IncrementalStat, compute_regression

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
            self.assertAlmostEqual(sum(values),        stats.sum)
            self.assertAlmostEqual(sum([val**2 for val in values]), stats.sum_square)


class LeafTest(unittest.TestCase):
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
        self.assertAlmostEqual(node.coeff,     self.coeff,      delta=delta)
        self.assertAlmostEqual(node.intercept, self.intercept,  delta=delta)
        self.assertAlmostEqual(node.rsquared, 1,                delta=delta)
        MSE = 0
        for xx, yy in zip(x, y):
            MSE += (yy - node.predict(xx))**2
        self.assertAlmostEqual(node.MSE, MSE/len(x))

    def test_init(self):
        for noise in [0, 1, 2, 4, 8]:
            x = [d[0] for d in self.data]
            y = [d[1] + random.gauss(0, noise) for d in self.data]
            node = Leaf(x, y)
            self.perform_tests(x, y, node, noise > 0)


    def test_add_remove(self):
        for noise in [0, 1, 2, 4, 8]:
            x = [d[0] for d in self.data]
            y = [d[1] + random.gauss(0, noise) for d in self.data]
            limit = self.size // 3
            new_x = x[:limit]
            new_y = y[:limit]
            node = Leaf(list(new_x), list(new_y))
            self.perform_tests(new_x, new_y, node, noise > 0)
            for xx, yy in zip(x[limit:], y[limit:]):
                node.add(xx, yy)
                new_x.append(xx)
                new_y.append(yy)
                self.perform_tests(new_x, new_y, node, noise > 0)
            for _ in range(2*limit):
                xx, yy = node.pop()
                self.assertEqual(xx, new_x.pop())
                self.assertEqual(yy, new_y.pop())
                self.perform_tests(new_x, new_y, node, noise > 0)

class NodeTest(unittest.TestCase):

    @staticmethod
    def generate_dataset(intercept, coeff, size, min_x, max_x):
        dataset = []
        for _ in range(size):
            x = random.uniform(min_x, max_x)
            y = x*coeff + intercept
            dataset.append((x, y))
        return dataset

    def test_nosplit(self):
        intercept = random.uniform(0, 100)
        coeff = random.uniform(0, 100)
        dataset = self.generate_dataset(intercept=intercept, coeff=coeff, size=50, min_x=0, max_x=100)
        reg = compute_regression(dataset)
        self.assertIsInstance(reg, Leaf)
        self.assertAlmostEqual(reg.intercept, intercept)
        self.assertAlmostEqual(reg.coeff, coeff)
        self.assertAlmostEqual(reg.error, 0)

    def test_singlesplit(self):
        intercept_1 = random.uniform(0, 100)
        coeff_1 = random.uniform(0, 100)
        intercept_2 = random.uniform(0, 100)
        coeff_2 = random.uniform(0, 100)
        split = random.uniform(30, 60)
        dataset1 =  self.generate_dataset(intercept=intercept_1, coeff=coeff_1, size=50, min_x=0, max_x=split)
        dataset2 = self.generate_dataset(intercept=intercept_2, coeff=coeff_2, size=50, min_x=split, max_x=100)
        dataset = dataset1 + dataset2
        random.shuffle(dataset)
        reg = compute_regression(dataset)
        self.assertIsInstance(reg, Node)
        self.assertAlmostEqual(reg.error, 0, delta=1e-4)
        self.assertIsInstance(reg.left, Leaf)
        self.assertAlmostEqual(reg.left.intercept, intercept_1)
        self.assertAlmostEqual(reg.left.coeff, coeff_1)
        self.assertIsInstance(reg.right, Leaf)
        self.assertAlmostEqual(reg.right.intercept, intercept_2)
        self.assertAlmostEqual(reg.right.coeff, coeff_2)
        self.assertEqual(reg.split, max(dataset1)[0])

if __name__ == "__main__":
    unittest.main()
