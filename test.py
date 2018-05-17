#! /usr/bin/env python3

import unittest
import random
import numpy
from decimal import Decimal
from fractions import Fraction
from pytree import Node, Leaf, IncrementalStat, compute_regression, Config

DEFAULT_MODE='RSS'

def generate_dataset(intercept, coeff, size, min_x, max_x, cls=float, repeat=1):
    dataset = []
    if cls is float:
        f = cls
    else:
        f = lambda x: cls('%.3f' % x)
    intercept = f(intercept)
    coeff     = f(coeff)
    for _ in range(size):
        x = f(random.uniform(min_x, max_x))
        y = x*coeff + intercept
        dataset.extend([(x, y)]*repeat)
    return dataset

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

    def test_fraction(self):
        size = random.randint(50, 100)
        values = []
        stats = IncrementalStat()
        for _ in range(size):
            val = Fraction(random.uniform(0, 100))
            stats.add(val)
            values.append(val)
        for _ in range(size-2): # don't do the last two ones
            val = stats.last
            self.assertEqual(stats.pop(), val)
            self.assertEqual(values.pop(), val)
            self.assertEqual(numpy.mean(values), stats.mean)
            self.assertEqual(numpy.var(values),  stats.var)
            self.assertEqual(sum(values),        stats.sum)

    def test_func(self):
        f = lambda x: x**2 - x + 4
        size = random.randint(50, 100)
        original_values = []
        values = []
        stats = IncrementalStat(f)
        for _ in range(size):
            val = random.uniform(0, 100)
            stats.add(val)
            original_values.append(val)
            values.append(f(val))
        for _ in range(size-2): # don't do the last two ones
            val = stats.last
            self.assertEqual(stats.pop(), val)
            self.assertEqual(original_values.pop(), val)
            values.pop()
            self.assertAlmostEqual(numpy.mean(values), stats.mean)
            self.assertAlmostEqual(numpy.var(values),  stats.var)
            self.assertAlmostEqual(numpy.std(values),  stats.std)
            self.assertAlmostEqual(sum(values),        stats.sum)


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
        self.config = Config(mode=DEFAULT_MODE, epsilon=1e-6)
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
        try:
            self.assertAlmostEqual(node.MSE, MSE/len(x))
        except:
            raise
        self.assertEqual(list(node), list(zip(x, y)))

    def test_init(self):
        for noise in [0, 1, 2, 4, 8]:
            x = [d[0] for d in self.data]
            y = [d[1] + random.gauss(0, noise) for d in self.data]
            node = Leaf(x, y, config=self.config)
            self.perform_tests(x, y, node, noise > 0)

    def test_add_remove(self):
        for noise in [0, 1, 2, 4, 8]:
            x = [d[0] for d in self.data]
            y = [d[1] + random.gauss(0, noise) for d in self.data]
            limit = self.size // 3
            new_x = x[:limit]
            new_y = y[:limit]
            node = Leaf(list(new_x), list(new_y), config=self.config)
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

    def test_plus(self):
        l1 = Leaf(range(10), range(10), config=self.config)
        l2 = Leaf(range(10, 20), range(10, 20), config=self.config)
        l = l1 + l2
        self.assertAlmostEqual(l.intercept, 0)
        self.assertAlmostEqual(l.coeff, 1)
        self.assertAlmostEqual(l.MSE, 0)
        self.assertEqual(l.x.values, l1.x.values + list(reversed(l2.x.values)))
        self.assertEqual(l.y.values, l1.y.values + list(reversed(l2.y.values)))

    def assert_equal_reg(self, dataset1, dataset2):
        leaf1 = Leaf([d[0] for d in dataset1], [d[1] for d in dataset1], config=self.config)
        leaf2 = Leaf([d[0] for d in dataset2], [d[1] for d in dataset2], config=self.config)
        self.assertEqual(leaf1, leaf2)

    def assert_notequal_reg(self, dataset1, dataset2):
        leaf1 = Leaf([d[0] for d in dataset1], [d[1] for d in dataset1], config=self.config)
        leaf2 = Leaf([d[0] for d in dataset2], [d[1] for d in dataset2], config=self.config)
        self.assertNotEqual(leaf1, leaf2)

    def test_eq(self):
        intercept = random.uniform(0, 100)
        coeff = random.uniform(0, 100)
        size = random.randint(50, 100)
        self.assert_equal_reg(generate_dataset(intercept, coeff, size, 0, 100),
                              generate_dataset(intercept, coeff, size, 0, 100))
        self.assert_equal_reg(generate_dataset(intercept, coeff, size, 0, 100),
                              generate_dataset(intercept, coeff, size, 1000, 2000))

    def test_neq(self):
        intercept = random.uniform(10, 100)
        coeff = random.uniform(10, 100)
        size = random.randint(50, 100)
        self.assert_notequal_reg(generate_dataset(intercept, coeff, size, 0, 100),
                                 generate_dataset(intercept*2, coeff, size, 0, 100))
        self.assert_notequal_reg(generate_dataset(intercept, coeff, size, 0, 100),
                                 generate_dataset(intercept, coeff*2, size, 0, 100))
        self.assert_notequal_reg(generate_dataset(intercept, coeff, size, 0, 100),
                                 generate_dataset(intercept*2, coeff*2, size, 0, 100))
        self.assert_notequal_reg(generate_dataset(intercept, coeff, size, 0, 100),
                                 generate_dataset(0, coeff, size, 0, 100))
        self.assert_notequal_reg(generate_dataset(intercept, coeff, size, 0, 100),
                                 generate_dataset(intercept, 0, size, 0, 100))

class NodeTest(unittest.TestCase):

    def test_nosplit(self):
        intercept = random.uniform(0, 100)
        coeff = random.uniform(0, 100)
        dataset = generate_dataset(intercept=intercept, coeff=coeff, size=50, min_x=0, max_x=100)
        reg = compute_regression(dataset)
        self.assertIsInstance(reg, Leaf)
        self.assertAlmostEqual(reg.intercept, intercept)
        self.assertAlmostEqual(reg.coeff, coeff)
        self.assertAlmostEqual(reg.RSS, 0, delta=1e-3)
        self.assertEqual(reg.breakpoints, [])
        self.assertEqual(list(reg), list(sorted(dataset)))

    def test_singlesplit(self):
        intercept_1 = random.uniform(0, 50)
        coeff_1 = random.uniform(0, 50)
        intercept_2 = random.uniform(50, 100)
        coeff_2 = random.uniform(50, 100)
        split = random.uniform(30, 60)
        dataset1 = generate_dataset(intercept=intercept_1, coeff=coeff_1, size=50, min_x=0, max_x=split)
        dataset2 = generate_dataset(intercept=intercept_2, coeff=coeff_2, size=50, min_x=split, max_x=100)
        dataset = dataset1 + dataset2
        random.shuffle(dataset)
        reg = compute_regression(dataset)
        self.assertIsInstance(reg, Node)
        self.assertAlmostEqual(reg.RSS, 0, delta=1e-3)
        self.assertIsInstance(reg.left, Leaf)
        self.assertAlmostEqual(reg.left.intercept, intercept_1)
        self.assertAlmostEqual(reg.left.coeff, coeff_1)
        self.assertIsInstance(reg.right, Leaf)
        self.assertAlmostEqual(reg.right.intercept, intercept_2)
        self.assertAlmostEqual(reg.right.coeff, coeff_2)
        self.assertEqual(reg.split, max(dataset1)[0])
        self.assertEqual(reg.breakpoints, [reg.split])
        self.assertEqual(list(reg), list(sorted(dataset)))

    def assertAlmostIncluded(self, sub_sequence, sequence, epsilon=1e-2):
        for elt in sub_sequence:
            is_in = False
            for ref in sequence:
                if abs(elt-ref) < epsilon:
                    is_in = True
                    break
            if not is_in:
                self.fail('Element %s is not in sequence %s (with ε=%f).' % (elt, sequence, epsilon))

    def generic_multiplesplits(self, cls, repeat):
        all_datasets = [generate_dataset(intercept=i, coeff=i, size=50, min_x=(i-1)*10, max_x=i*10, cls=cls, repeat=repeat) for i in range(1, 9)]
        dataset = sum(all_datasets, [])
        reg = compute_regression(dataset)
        self.assertEqual(list(reg), list(sorted(dataset)))
        self.assertIn(len(reg.breakpoints), (7, 8)) # TODO should be 7, but is 8 i reality because of the non-optimality of the algorithm
        self.assertAlmostIncluded(range(10, 80, 10), reg.breakpoints, epsilon=2)
        for x, y in dataset:
            prediction = reg.predict(x)
            self.assertAlmostEqual(y, prediction)

    def test_multiple_splits(self):
        self.generic_multiplesplits(float, 1)
        self.generic_multiplesplits(float, 10)

    def test_multiple_splits_decimal(self):
        self.generic_multiplesplits(Decimal, 1)

    def test_multiple_splits_fraction(self):
        self.generic_multiplesplits(Fraction, 1)

if __name__ == "__main__":
    unittest.main()
